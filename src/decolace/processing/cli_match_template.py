import typer
import logging
from rich.logging import RichHandler
from rich import print
from pathlib import Path


app = typer.Typer()

@app.command()
def run_matchtemplate(
    ctx: typer.Context,
    template: Path = typer.Option(None, help="Path to wanted template file"),
    angular_step: float = typer.Option(3.0, help="Angular step for template matching"),
    in_plane_angular_step: float = typer.Option(2.0, help="In plane angular step for template matching"),
    defocus_step: float = typer.Option(0.0, help="Defocus step for template matching"),
    defocus_range: float = typer.Option(0.0, help="Defocus range for template matching"),
    save_mip: bool = typer.Option(False, help="Save MIP of template"),
):
    """Runs match template on all images in the acquisition areas"""
    from decolace.processing.project_managment import MatchTemplateRun
    from pycistem.programs import match_template, generate_gpu_prefix, generate_num_procs
    import pycistem
    pycistem.set_cistem_path("/groups/elferich/cistem_binaries/")
    from pycistem.database import get_already_processed_images
    import pandas as pd
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[
            RichHandler(),
            #logging.FileHandler(current_output_directory / "log.log")
        ]
    )
    run_profile = {
            "prague":8,
            "budapest":8,
            "helsinki":8,
            "palermo":8,
        "warsaw": 8,
        "bucharest": 8,
        "istanbul": 8,
        "kyiv": 8,
        "barcelona": 8,
        "milano": 8,
        "sofia": 8,
        "manchester": 8,
    }   
    
    new_mtr = True
    if ctx.obj.match_template_job is None:
        if len(ctx.obj.project.match_template_runs) == 0:
            match_template_job_id = 1
        else:
            match_template_job_id = max([mtr.run_id  for mtr in ctx.obj.project.match_template_runs]) + 1
        if template is None:
            typer.echo("No template file or template match job id given")
            return
    else:
        new_mtr = False
        # Get position in array
        template = Path(ctx.obj.match_template_job.template_path)
        angular_step = ctx.obj.match_template_job.angular_step
        in_plane_angular_step = ctx.obj.match_template_job.in_plane_angular_step
        defocus_step = ctx.obj.match_template_job.defocus_step
        defocus_range = ctx.obj.match_template_job.defocus_range
        match_template_job_id = ctx.obj.match_template_job.run_id
        typer.echo(f"Template match job id already exists, continuing job {ctx.obj.match_template_job.run_name}")
        typer.echo(f"template_filename={template.absolute().as_posix()}, angular_step={angular_step}, in_plane_angular_step={in_plane_angular_step} defous_step={defocus_step}, defocus_range={defocus_range}, decolace=True)")

    if new_mtr:
        ctx.obj.project.match_template_runs.append(
            MatchTemplateRun(
                run_name=f"matchtemplate_{match_template_job_id}",
                run_id=match_template_job_id,
                template_path=template.absolute().as_posix(),
                angular_step=angular_step,
                in_plane_angular_step=in_plane_angular_step,
                defocus_step=defocus_step,
                defocus_range=defocus_range,
            )
        )
        ctx.obj.project.write()
            

    all_image_info=[]
    for aa in ctx.obj.acquisition_areas:
        typer.echo(f"Running matchtemplate for {aa.area_name}")
        image_info = match_template.parameters_from_database(aa.cistem_project, template_filename=template.absolute().as_posix(), match_template_job_id=match_template_job_id, decolace=True)
        orig_num= len(image_info)
        for par in image_info["PARAMETERS"]:
            par.angular_step = angular_step
            par.in_plane_angular_step = in_plane_angular_step
            par.defocus_step = defocus_step
            par.defocus_search_range = defocus_range
            par.max_threads = 2
            if save_mip:
                par.mip_output_file = par.scaled_mip_output_file.replace("_scaled_mip.mrc", "_mip.mrc")
        
        already_processed_images = get_already_processed_images(aa.cistem_project, match_template_job_id)
        image_info = image_info[~image_info['IMAGE_ASSET_ID'].isin(already_processed_images['IMAGE_ASSET_ID'])]
        all_image_info.append(image_info)
        typer.echo(f"{len(image_info)} tiles out of {orig_num} still to process")
        if len(image_info) == 0:
            typer.echo(f"All tiles already processed")
            continue
    all_image_info = pd.concat(all_image_info)
    typer.echo(f"Total of {len(all_image_info)} tiles to process")

    res = match_template.run(all_image_info,num_procs=generate_num_procs(run_profile),cmd_prefix=list(generate_gpu_prefix(run_profile)),cmd_suffix='"', sleep_time=1.0, write_directly_to_db=True)
        #typer.echo(f"Writing results for {aa.area_name}")

        #match_template.write_results_to_database(aa.cistem_project,pars,res,image_info)
        #typer.echo(f"Done with {aa.area_name}")
        #project.write()

@app.command()
def create_tmpackage(
    ctx: typer.Context,
):
    """Creates a Template-Matches Package for each acquisition area (i.e. a star file containing all matches)"""
    from pycistem.database import get_num_already_processed_images, write_match_template_to_starfile, insert_tmpackage_into_db, get_num_images
    

    for aa in ctx.obj.acquisition_areas:
        if get_num_already_processed_images(aa.cistem_project, ctx.obj.match_template_job.run_id) != get_num_images(aa.cistem_project):
            typer.echo(f"No images processed for {aa.area_name}")
            continue
        
        typer.echo(f"Creating tm package for {aa.area_name}")
        output_filename = Path(aa.cistem_project).parent / "Assets" / "TemplateMatching" / f"{aa.area_name}_{ctx.obj.match_template_job.run_id}_tm_package.star"
        if output_filename.exists():
            typer.echo(f"Package already exists for {aa.area_name}")
            continue
        write_match_template_to_starfile(aa.cistem_project, ctx.obj.match_template_job.run_id, output_filename)  
        insert_tmpackage_into_db(aa.cistem_project, f"DeCOoLACE_{aa.area_name}_TMRun_{ctx.obj.match_template_job.run_id}", output_filename)
        
@app.command()
def run_refinetemplate(
    ctx: typer.Context,
):
    from pycistem.programs import refine_template_dev
    from pycistem.config import set_cistem_path
    set_cistem_path("/groups/cryoadmin/software/CISTEM/je_dev/")
   
    for aa in ctx.obj.acquisition_areas:
        tm_package_file = Path(aa.cistem_project).parent / "Assets" / "TemplateMatching" / f"{aa.area_name}_{ctx.obj.match_template_job.run_id}_tm_package.star"
        if not tm_package_file.exists():
            typer.echo(f"No tm package for {aa.area_name}")
            continue
        typer.echo(f"Running refine template for {aa.area_name}")
        par = refine_template_dev.RefineTemplateDevParameters(
            input_starfile=tm_package_file.as_posix(),
            output_starfile=tm_package_file.with_suffix('').as_posix()+'_refined.star',
            input_template=Path(ctx.obj.match_template_job.template_path).as_posix(),
            num_threads=10
        )
        if Path(par.output_starfile).exists():
            typer.echo(f"Refined tm package already exists for {aa.area_name}")
            continue
        refine_template_dev.run(par,save_output=True,save_output_path="/tmp/output")

@app.command()
def assemble_matches(
    ctx: typer.Context,
    use_filtered: str = typer.Option("", help="Use this set of filtered matches"),
    use_filtered_montage: bool = typer.Option(False, help="Use the filtered montage"),
):
    """Maps matches from different tiles into the montage. Must run refine temaplte before"""
    import starfile
    import decolace.processing.decolace_processing as dp
    if ctx.obj.match_template_job is None:
        typer.echo("No match template job given")
        raise typer.Exit()
    for aa in ctx.obj.acquisition_areas:
        if use_filtered != "":
            refined_matches_file = Path(aa.cistem_project).parent / "Assets" / "TemplateMatching" / f"{aa.area_name}_{ctx.obj.match_template_job.run_id}_tm_package_filtered_{use_filtered}.star"
        else:
            refined_matches_file = Path(aa.cistem_project).parent / "Assets" / "TemplateMatching" / f"{aa.area_name}_{ctx.obj.match_template_job.run_id}_tm_package_refined.star"
        if not refined_matches_file.exists():
            typer.echo(f"No refined matches for {aa.area_name}")
            continue
        if aa.montage_star is None:
            typer.echo(f"No montage for {aa.area_name}")
            continue
        typer.echo(f"Assembling matches for {aa.area_name}")
        output_dir = Path(ctx.obj.project.project_path) / "Matches"
        output_dir.mkdir(parents=True, exist_ok=True)
        if use_filtered != "":
            output_star_path = output_dir / f"{aa.area_name}_{ctx.obj.match_template_job.run_name}_{ctx.obj.match_template_job.run_id}_filtered.star"
        else:
            output_star_path = output_dir / f"{aa.area_name}_{ctx.obj.match_template_job.run_name}_{ctx.obj.match_template_job.run_id}.star"
        montage_data = starfile.read(aa.montage_star)

        # Read matches data
        matches_data = starfile.read(refined_matches_file)
        if len(matches_data) < 10:
            typer.echo(f"Not enough matches for {aa.area_name}")
            continue
        result = dp.assemble_matches(montage_data, matches_data)
        if use_filtered_montage:
            result["cisTEMOriginalImageFilename"] = result["cisTEMOriginalImageFilename"].str.replace(".mrc", "_filtered.mrc")
        starfile.write(result, output_star_path, overwrite=True)

@app.command()
def precompute_filters(
    ctx: typer.Context,
    binning_boxsize: int = typer.Option(256, help="Binning boxsize"),
):
    """ Precalculates values used to filter out potential false-positives or otherwise problematic matches"""
    from decolace.processing.match_filtering import get_distance_to_edge
    import starfile
    import mrcfile
    import numpy as np
    from functools import partial
    import concurrent.futures
    if ctx.obj.match_template_job is None:
        typer.echo("No match template job given")
        raise typer.Exit()
    for aa in ctx.obj.acquisition_areas:
        refined_matches_starfile = Path(aa.cistem_project).parent / "Assets" / "TemplateMatching" / f"{aa.area_name}_{ctx.obj.match_template_job.run_id}_tm_package_refined.star"
        filtered_matches_starfile = Path(aa.cistem_project).parent / "Assets" / "TemplateMatching" / f"{aa.area_name}_{ctx.obj.match_template_job.run_id}_tm_package_filtervalues.star"
        if not refined_matches_starfile.exists():
            typer.echo(f"No refined matches for {aa.area_name}")
            continue
        if filtered_matches_starfile.exists():
            typer.echo(f"Filter values for matches already exist for {aa.area_name}")
            continue
        refined_matches = starfile.read(refined_matches_starfile)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            fn = partial(get_distance_to_edge, refined_matches=refined_matches, binning_boxsize=binning_boxsize)
            executor.map(fn, refined_matches["cisTEMOriginalImageFilename"].unique())

        starfile.write(refined_matches, filtered_matches_starfile, overwrite=True)  


@app.command()
def filter_matches(
    ctx: typer.Context,
    filterset_name: str = typer.Argument(..., help="Name to use for this filtered set"),
    refined_score_cutoff: float = typer.Option(8.0, help="Cutoff for refined score"),
    distance_from_beamedge_cutoff: float = typer.Option(1.0, help="Cutoff for distance from beam in pixels"),
    variance_after_binning_cutoff: float = typer.Option(0.3, help="Cutoff for variance after binning")
):
    """Uses a mixture of strategies to remove false positive matches"""

    import starfile
    import mrcfile
    import numpy as np
    from scipy.ndimage import binary_erosion

    if ctx.obj.match_template_job is None:
        typer.echo("No match template job given")
        raise typer.Exit()
    for aa in ctx.obj.acquisition_areas:
        refined_matches_starfile = Path(aa.cistem_project).parent / "Assets" / "TemplateMatching" / f"{aa.area_name}_{ctx.obj.match_template_job.run_id}_tm_package_filtervalues.star"
        if not refined_matches_starfile.exists():
            typer.echo(f"No refined matches for {aa.area_name}")
            continue
        refined_matches = starfile.read(refined_matches_starfile)
        print(f"Starting with {len(refined_matches)} matches")
        refined_matches = refined_matches[refined_matches["cisTEMScore"] >= refined_score_cutoff]
        print(f"After {refined_score_cutoff} score criterion {len(refined_matches)} matches")

        refined_matches = refined_matches[refined_matches["LACEBeamEdgeDistance"] >= distance_from_beamedge_cutoff]
        print(f"After {distance_from_beamedge_cutoff} distance from beam criterion {len(refined_matches)} matches")

        refined_matches = refined_matches[refined_matches["LACEVarianceAfterBinning"] <= variance_after_binning_cutoff]
        print(f"After {variance_after_binning_cutoff} variance after binning criterion {len(refined_matches)} matches")
        
        filtered_matches_starfile = Path(aa.cistem_project).parent / "Assets" / "TemplateMatching" / f"{aa.area_name}_{ctx.obj.match_template_job.run_id}_tm_package_filtered_{filterset_name}.star"
        starfile.write(refined_matches, filtered_matches_starfile, overwrite=True)
        print(f"Wrote {len(refined_matches)} matches to {filtered_matches_starfile}")

@app.command()
def join_matches(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Name to use for this set of combined matches"),
    use_filtered: str = typer.Option("", help="Use this set of filtered matches"),
    use_other_images: str = typer.Option("", help="Use other images"),
    use_different_pixel_size: float = typer.Option(None, help="Use different pixel size"),
):
    import starfile 
    import pandas as pd
    from pycistem.database import create_project, insert_tmpackage_into_db

    matches = []
    if ctx.obj.match_template_job is None:
        typer.echo("No match template job given")
        raise typer.Exit()
    
    for aa in ctx.obj.acquisition_areas:
        filtered_matches_starfile = Path(aa.cistem_project).parent / "Assets" / "TemplateMatching" / f"{aa.area_name}_{ctx.obj.match_template_job.run_id}_tm_package_filtered_{use_filtered}.star"
        if filtered_matches_starfile.exists():
            new_matches = starfile.read(filtered_matches_starfile)
            if len(new_matches) > 0:
                matches.append(new_matches)
    combined_matches = pd.concat(matches)
    combined_matches_starfile = Path(ctx.obj.project.project_path) / "Matches" / f"combined_{ctx.obj.match_template_job.run_name}_{ctx.obj.match_template_job.run_id}_{name}.star"
    if use_other_images != "":
        combined_matches_starfile = combined_matches_starfile.with_name(combined_matches_starfile.stem + f"_{use_other_images}.star")
        combined_matches['cisTEMOriginalImageFilename'].str.replace(".mrc", f"_{use_other_images}.mrc")
        if use_different_pixel_size is not None:
            combined_matches['cisTEMPixelSize'] = use_different_pixel_size
    starfile.write(combined_matches, combined_matches_starfile, overwrite=True)

    database = create_project(f"processing_{name}", ctx.obj.project.project_path.absolute() / "cistem_projects")
    print(f"{combined_matches_starfile}")
    insert_tmpackage_into_db(database, f"DeCOoLACE_combind_TMRun_{ctx.obj.match_template_job.run_name}_{ctx.obj.match_template_job.run_id}_{name}_{len(combined_matches)}", combined_matches_starfile)


@app.command()
def calculate_changes_during_refine_template(
    ctx: typer.Context
):

    import starfile

    if ctx.obj.match_template_job is None:
        typer.echo("No match template job given")
        raise typer.Exit()
    for aa in ctx.obj.acquisition_areas:
        original_matches_starfile = Path(aa.cistem_project).parent / "Assets" / "TemplateMatching" / f"{aa.area_name}_{ctx.obj.match_template_job.run_id}_tm_package.star"
        refined_matches_starfile = Path(aa.cistem_project).parent / "Assets" / "TemplateMatching" / f"{aa.area_name}_{ctx.obj.match_template_job.run_id}_tm_package_refined.star"
        original_matches = starfile.read(original_matches_starfile)
        refined_matches = starfile.read(refined_matches_starfile)
        # Add columns to refined matches
        refined_matches["psi_change"] = 0
        refined_matches["theta_change"] = 0
        refined_matches["phi_change"] = 0
        refined_matches["defocus_change"] = 0
        refined_matches["x_change"] = 0
        refined_matches["y_change"] = 0
        for (i,original_match), (j,refined_match) in zip(original_matches.iterrows(), refined_matches.iterrows()):
            refined_matches.loc[j, "psi_change"] = original_match["cisTEMAnglePsi"] - refined_match["cisTEMAnglePsi"]
            refined_matches.loc[j, "theta_change"] = original_match["cisTEMAngleTheta"] - refined_match["cisTEMAngleTheta"]
            refined_matches.loc[j, "phi_change"] = original_match["cisTEMAnglePhi"] - refined_match["cisTEMAnglePhi"]
            refined_matches.loc[j, "defocus_change"] = original_match["cisTEMDefocus1"] - refined_match["cisTEMDefocus1"]
            refined_matches.loc[j, "x_change"] = original_match["cisTEMOriginalXPosition"] - refined_match["cisTEMOriginalXPosition"]
            refined_matches.loc[j, "y_change"] = original_match["cisTEMOriginalYPosition"] - refined_match["cisTEMOriginalYPosition"]
        filtered_matches_starfile = Path(aa.cistem_project).parent / "Assets" / "TemplateMatching" / f"{aa.area_name}_{ctx.obj.match_template_job.run_id}_tm_package_filtered.star"
        starfile.write(refined_matches, filtered_matches_starfile, overwrite=True)
