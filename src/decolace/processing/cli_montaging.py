import typer
from pathlib import Path
from typing import List

app = typer.Typer()

@app.command()
def reset_montage(
    project_main: Path = typer.Option(None, help="Path to wanted project file"),
):
    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)

    for aa in project.acquisition_areas:
        aa.montage_image = None
        aa.montage_star = None
        aa.initial_tile_star = None
        aa.refined_tile_star = None
    project.write()

@app.command()
def run_montage(
    project_main: Path = typer.Option(None, help="Path to wanted project file"),
    acquisition_areas: List[int] = typer.Option(None, help="List of acquisition areas to run montage for"),
    iterations: int = typer.Option(2, help="Number of iterations"),
    num_procs: int = typer.Option(10, help="Number of processors to use"),
    bin_to: float = typer.Option(10.0, help="Pixel size to bin to"),
    erode_mask_shifts: int = typer.Option(0, help="Erode mask by this many pixels before calculating cross-correlation"),
    erode_mask_montage: int = typer.Option(0, help="Erode mask by this many pixels before assembling montage"), 
    filter_cutoff_frequency_ratio: float = typer.Option(0.02, help="Cutoff ratio of the butterworth filter"), 
    ilter_order: float = typer.Option(4.0, help="Order of the butterworth filter"), 
    mask_size_cutoff: int = typer.Option(100, help="If the mask size is smaller than this, the pair will be discarded"),
    overlap_ratio: float = typer.Option(0.1, help="Overlap ratio parameter for masked crosscorrelation"),
):
    from rich.console import Console
    import starfile
    from numpy.linalg import LinAlgError
    import glob
    from decolace.processing.project_managment import ProcessingProject


    console = Console()
    
    from decolace.processing.decolace_processing import read_data_from_cistem, read_decolace_data, create_tile_metadata, find_tile_pairs, calculate_shifts, calculate_refined_image_shifts, calculate_refined_intensity, create_montage_metadata, create_montage
    import numpy as np
    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)
    print(acquisition_areas)
    for i, aa in enumerate(project.acquisition_areas):
        if aa.montage_image is not None:
            continue
        if len(acquisition_areas) > 0 and i not in acquisition_areas:
            continue
        output_directory = project.project_path / "Montages" / aa.area_name
        typer.echo(f"Running montage for {aa.area_name}")
        if aa.initial_tile_star is None:
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
            aa.initial_tile_star = output_path
            typer.echo(f"Wrote tile metadata to {output_path}.")
        else:
            tile_metadata_result = starfile.read(aa.initial_tile_star,always_dict=True)
            tile_metadata = tile_metadata_result["tiles"]
            typer.echo(f"Read tile metadata from {aa.initial_tile_star}.")

        if aa.refined_tile_star is None:
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
                # Calculate shifts of tile pairs
                shifts = calculate_shifts(tile_pairs, num_procs, erode_mask=erode_mask_shifts, filter_cutoff_frequency_ratio=filter_cutoff_frequency_ratio, filter_order=ilter_order, mask_size_cutoff=mask_size_cutoff, overlap_ratio=overlap_ratio)
                console.log(
                    f"Shifts were adjusted. Mean: {np.mean(shifts['add_shift']):.2f} A, Median: {np.median(shifts['add_shift']):.2f} A, Std: {np.std(shifts['add_shift']):.2f} A. Min: {np.min(shifts['add_shift']):.2f} A, Max: {np.max(shifts['add_shift']):.2f} A."
                )
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
        project.write()

@app.command()
def filter_montage(
    ctx: typer.Context,
):
    for aa in ctx.obj.acquisition_areas:
        if aa.montage_image is None:
            continue
        output_directory = ctx.obj.project.project_path / "Montages" / aa.area_name
        output_path = output_directory / f"{aa.area_name}_montage_filtered.mrc"
        from decolace.processing.montage_filtering import highpassfilter_montage
        highpassfilter_montage(aa.montage_image, output_path)
