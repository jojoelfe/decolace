import typer
from pathlib import Path
from typing import List, Optional

app = typer.Typer()

@app.command()
def reset_montage(
    ctx: typer.Context,
):
    for aa in ctx.obj.acquisition_areas:
        aa.montage_image = None
        aa.montage_star = None
        aa.initial_tile_star = None
        aa.refined_tile_star = None
    ctx.obj.project.write()

@app.command()
def run_montage(
    ctx: typer.Context,
    iterations: int = typer.Option(3, help="Number of iterations"),
    num_procs: int = typer.Option(10, help="Number of processors to use"),
    bin_to: float = typer.Option(10.0, help="Pixel size to bin to"),
    erode_mask_shifts: int = typer.Option(50, help="Erode mask by this many pixels before calculating cross-correlation"),
    erode_mask_montage: int = typer.Option(50, help="Erode mask by this many pixels before assembling montage"), 
    filter_cutoff_frequency_ratio: float = typer.Option(0.02, help="Cutoff ratio of the butterworth filter"), 
    filter_order: float = typer.Option(4.0, help="Order of the butterworth filter"), 
    mask_size_cutoff: int = typer.Option(100, help="If the mask size is smaller than this, the pair will be discarded"),
    overlap_ratio: float = typer.Option(0.2, help="Overlap ratio parameter for masked crosscorrelation"),
    redo: bool = typer.Option(False, help="Redo the montage even if it already exists"),
    redo_montage: bool = typer.Option(False, help="Redo only the creatin of the montage even if it already exists"),
    max_mean_density: Optional[float] = typer.Option(None, help="Maximum mean density of the tiles"),
    cc_cutoff_as_fraction_of_median: float = typer.Option(0.5, help="Cutoff for the cross-correlation as a fraction of the median cross-correlation"),
):
    from rich.console import Console
    import starfile
    from numpy.linalg import LinAlgError
    import pandas as pd

    console = Console()
    
    from decolace.processing.decolace_processing import read_data_from_cistem, read_decolace_data, create_tile_metadata, find_tile_pairs, calculate_shifts, calculate_refined_image_shifts, calculate_refined_intensity, create_montage_metadata, create_montage, prune_bad_shifts, prune_tiles
    import numpy as np
    for i, aa in enumerate(ctx.obj.acquisition_areas):
        if aa.montage_image is not None and not redo and not redo_montage:
            continue
        output_directory = ctx.obj.project.project_path / "Montages" / aa.area_name
        typer.echo(f"Running montage for {aa.area_name}")
        if aa.initial_tile_star is None or redo:
            cistem_data = read_data_from_cistem(aa.cistem_project)
            typer.echo(
                f"Read data about {len(cistem_data)} tiles."
            )
            # Read decolace data
            try:
                decolace_data = read_decolace_data(aa.decolace_session_info_path)
            except TypeError:
                decolace_data = read_decolace_data(aa.decolace_acquisition_area_info_path)
            # Create tile metadata
            
            output_directory.mkdir(parents=True, exist_ok=True)
            output_path = output_directory / f"{aa.area_name}_tile_metadata.star"
            if type(decolace_data) is not dict:
                decolace_data = dict(decolace_data)
            tile_metadata_result = create_tile_metadata(cistem_data, decolace_data, output_path)
            tile_metadata = tile_metadata_result["tiles"]
            tile_metadata, message = prune_tiles(tile_metadata, max_mean_density=max_mean_density)
            console.log(message) 
            aa.initial_tile_star = output_path
            typer.echo(f"Wrote tile metadata to {output_path}.")
        else:
            tile_metadata_result = starfile.read(aa.initial_tile_star,always_dict=True)
            tile_metadata = tile_metadata_result["tiles"]
            typer.echo(f"Read tile metadata from {aa.initial_tile_star}.")
        
        # I should sort by acquisition time here
        if aa.refined_tile_star is None or redo:
            estimated_distance_threshold = np.median(
                tile_metadata["tile_x_size"] * tile_metadata["tile_pixel_size"]
            )
            refinement_failed = False
            for iteration in range(iterations):
                console.log(f"Starting iteration {iteration+1} of {iterations}.")
                # Calculate tile paris
                tile_pairs = find_tile_pairs(tile_metadata, estimated_distance_threshold)
                console.log(
                    f"In {len(tile_metadata)} tiles {len(tile_pairs)} tile pairs were found using distance threshold of {estimated_distance_threshold:.2f} A."
                )
                # I should sort by image_order here
                # Calculate shifts of tile pairs
                shifts = calculate_shifts(tile_pairs, num_procs, erode_mask=erode_mask_shifts, filter_cutoff_frequency_ratio=filter_cutoff_frequency_ratio, filter_order=filter_order, mask_size_cutoff=mask_size_cutoff, overlap_ratio=overlap_ratio)
                console.log(
                    f"Shifts were adjusted. Mean: {np.mean(shifts['add_shift']):.2f} A, Median: {np.median(shifts['add_shift']):.2f} A, Std: {np.std(shifts['add_shift']):.2f} A. Min: {np.min(shifts['add_shift']):.2f} A, Max: {np.max(shifts['add_shift']):.2f} A."
                )
                # TODO: prune bad shifts and then tiles with no shifts
                shifts, message = prune_bad_shifts(shifts,cc_cutoff_as_fraction_of_media=cc_cutoff_as_fraction_of_median)
                shifts = shifts.copy()
                console.log(message)

                # Prune out tiles without any shifts
                init_len = len(tile_metadata)
                tile_metadata["filename_index"] = tile_metadata["tile_filename"]
                tile_metadata.set_index("filename_index", inplace=True)
                tile_metadata = tile_metadata.loc[pd.concat([shifts["image_1"],shifts["image_2"]]).unique()].copy()
                console.log(f"Pruned out {init_len - len(tile_metadata)} tiles without any shifts.")
                output_path = output_directory / f"{aa.area_name}_cc_shifts_{iteration}.star"
                starfile.write(shifts, output_path, overwrite=True)
                # Optimize tile positions
                try:
                    calculate_refined_image_shifts(tile_metadata, shifts)
                # Optimize intensities
                    calculate_refined_intensity(tile_metadata, shifts)
                except LinAlgError:
                    console.log("Optimization failed.")
                    refinement_failed = True
                    break
            if refinement_failed:
                break
            # Write new tile data

            output_path = output_directory / f"{aa.area_name}_refined_tile_metadata.star"
            starfile.write(tile_metadata, output_path, overwrite=True)
            aa.refined_tile_star = output_path
            console.log(f"Wrote refined tile metadata to {output_path}.")
        else:
            tile_metadata = starfile.read(aa.refined_tile_star)
            console.log(f"Read refined tile metadata from {aa.refined_tile_star}.")
        
        # Create montage
        output_path_metadata = output_directory / f"{aa.area_name}_montage_metadata.star"
        output_path_montage = output_directory / f"{aa.area_name}_montage.mrc"
        montage_metadata = create_montage_metadata(
            tile_metadata, output_path_metadata, bin_to / tile_metadata["tile_pixel_size"].iloc[0], output_path_montage
        )
        console.log(f"Wrote montage metadata to {output_path_metadata}.")
        create_montage(montage_metadata, output_path_montage, erode_mask=erode_mask_montage)
        console.log(
            f"Wrote {montage_metadata['montage']['montage_x_size'].values[0]}x{montage_metadata['montage']['montage_y_size'].values[0]} montage to {output_path_montage}."
        )
        aa.montage_star = output_path_metadata
        aa.montage_image = output_path_montage
        ctx.obj.project.write()


from enum import Enum
class KnotsOptions(str, Enum):
    uniform = "uniform"
    quantile = "quantile"

@app.command()
def filter_montage(
    ctx: typer.Context,
    n_knots: int = typer.Option(5, help="Number of knots for spline"),
    knots: KnotsOptions = typer.Option(KnotsOptions.uniform, help="Knots for spline"),
):
    for aa in ctx.obj.acquisition_areas:
        if aa.montage_image is None:
            continue
        typer.echo(f"Filtering montage for {aa.area_name}")
        output_directory = ctx.obj.project.project_path / "Montages" / aa.area_name
        output_path = output_directory / f"{aa.area_name}_montage_filtered.mrc"
        from decolace.processing.montage_filtering import subtract_linear_background_model
        #highpassfilter_montage(aa.montage_image, output_path)
        subtract_linear_background_model(aa.montage_image, output_path, n_knots=n_knots, knots=knots)
