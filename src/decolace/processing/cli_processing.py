from pathlib import Path
from typing import Optional, Union, List
from types import SimpleNamespace
import typer
import glob
from rich import print
#import numpy as np
#import pandas as pd
import logging
from rich.logging import RichHandler

from decolace.processing.project_managment import ProcessingProject, AcquisitionAreaPreProcessing

#from decolace.processing.project_managment import ProcessingProject, AcquisitionAreaPreProcessing, MatchTemplateRun
#from decolace.acquisition.session import session

#from decolace.processing.decolace_processing import read_data_from_cistem, read_decolace_data


app = typer.Typer()


@app.command()
def create_project(
    name: str = typer.Argument(..., help="Name of project"),
    directory: Optional[Path] = typer.Option(
        None, help="Directory to create project in"
    ),
):

    if directory is None:
        directory = Path.cwd()
    project = ProcessingProject(project_name = name, project_path= directory)
    project.write()


@app.command()
def add_acquisition_area(
    project_main: Path = typer.Option(None, help="Path to wanted project file"),
    area_name: str = typer.Argument(..., help="Name of the area"),
    session_name: str = typer.Argument(..., help="Name of session"),
    grid_name: str = typer.Argument(..., help="Name of grid"),
    area_directory: Path = typer.Argument(..., help="Path to area directory"),

):
    from decolace.acquisition.acquisition_area import AcquisitionAreaSingle
    import numpy as np

    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)

    aa = AcquisitionAreaSingle(area_name, area_directory.as_posix())
    aa.load_from_disk()
    if np.sum(aa.state['positions_acquired']) == 0:
                print(f"{aa.name}: No Data")
    aa_pre = AcquisitionAreaPreProcessing(
                area_name = f"{session_name}_{grid_name}_{aa.name}",
                decolace_acquisition_area_info_path = sorted(Path(aa.directory).glob(f"{aa.name}*.npy"))[-1],
                decolace_grid_info_path = None,
                decolace_session_info_path = None,
                frames_folder = aa.frames_directory,
            )
    project.acquisition_areas.append(aa_pre)
    project.write()
    typer.echo(f"Added acquisition area {aa.name} ")


@app.command()
def add_session(
    project_main: Path = typer.Option(None, help="Path to wanted project file"),
    session_name: str = typer.Argument(..., help="Name of session"),
    session_directory: Path = typer.Argument(..., help="Path to session directory"),
    ignore_grids: List[str] = typer.Option(["cross"], help="Grids to ignore"),
):
    from decolace.acquisition.session import session
    import numpy as np
    
    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)

    my_session = session(session_name, session_directory.as_posix())
    session_path = sorted(Path(my_session.directory).glob(f"{my_session.name}*.npy"))[-1]
    my_session.load_from_disk()
    num_aa = 0
    num_grids = 0
    for grid in my_session.grids:
        if grid.name in ignore_grids:
            continue
        grid_path = sorted(Path(grid.directory).glob(f"{grid.name}*.npy"))[-1]
        num_grids += 1
        for aa in grid.acquisition_areas:
            if np.sum(aa.state['positions_acquired']) == 0:
                print(f"{grid.name} - {aa.name}: No Data")
                continue
            aa_pre = AcquisitionAreaPreProcessing(
                area_name = f"{my_session.name}_{grid.name}_{aa.name}",
                decolace_acquisition_area_info_path = sorted(Path(aa.directory).glob(f"{aa.name}*.npy"))[-1],
                decolace_grid_info_path = grid_path,
                decolace_session_info_path = session_path,
                frames_folder = aa.frames_directory,
            )
            project.acquisition_areas.append(aa_pre)
            num_aa += 1
    typer.echo(f"Added {num_aa} acquisition areas from {num_grids} grids to {project.project_name}")
    project.write()



@app.command()
def generate_cistem_projects(
    project_main: Path = typer.Option(None, help="Path to wanted project file"),
    pixel_size: float = typer.Option(...,help="Movie pixelsize"),
    exposure_dose: float = typer.Option(...,help="Dose in e/A/frame")
):
    from decolace.processing.create_cistem_projects_for_session import create_project_for_area 

    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)

    for aa in project.acquisition_areas:
        if aa.cistem_project is not None:
            continue
        typer.echo(f"Creating cistem project for {aa.area_name}")
        cistem_project_path = create_project_for_area(aa.area_name, project_path.parent.absolute() / "cistem_projects", aa.frames_folder, pixel_size=pixel_size, exposure_dose=exposure_dose, bin_to=project.processing_pixel_size)
        aa.cistem_project = cistem_project_path
        typer.echo(f"Saved as {cistem_project_path}")
    project.write()
   
@app.command()
def run_unblur(
    project_main: Path = typer.Option(None, help="Path to wanted project file"),
    cistem_path: str = typer.Option("/scratch/paris/elferich/cisTEM/build/je_combined_Intel-gpu-debug-static/src/", help="Path to cistem binaries"),
    num_cores: int = typer.Option(10, help="Number of cores to use"),
    cmd_prefix: str = typer.Option("", help="Prefix of run command"),
    cmd_suffix: str = typer.Option("", help="Suffix of run command")
):
    from pycistem.programs import unblur
    import pycistem
    pycistem.set_cistem_path(cistem_path)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[
            RichHandler(),
            #logging.FileHandler(current_output_directory / "log.log")
        ]
    )

    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)

    for aa in project.acquisition_areas:
        if aa.unblur_run:
            continue
        typer.echo(f"Running unblur for {aa.area_name}")
        pars = unblur.parameters_from_database(aa.cistem_project,decolace=True)

        res = unblur.run(pars,num_procs=num_cores,cmd_prefix=cmd_prefix,cmd_suffix=cmd_suffix,save_output=True,save_output_path="/tmp/output")

        unblur.write_results_to_database(aa.cistem_project,pars,res)
        aa.unblur_run = True
        project.write()


def process_experimental_conditions(acquisition_areas: List[AcquisitionAreaPreProcessing]):
    from collections import defaultdict
    experimental_conditions_column = [aa.experimental_condition for aa in acquisition_areas]
    unique_conditions = defaultdict(dict)
    for i, ec_line in enumerate(experimental_conditions_column):
        if ":" not in ec_line:
            continue
        for ec in ec_line.split(";"):
            key, value = ec.split(":")
            unique_conditions[key][i] = value
    return((list(unique_conditions.keys()), experimental_conditions_column))

    


@app.command()
def status(
    project_main: Path = typer.Option(None, help="Path to wanted project file")
):
    from pycistem.database import get_num_movies, get_num_images, get_num_already_processed_images
    from rich.table import Table

    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)

    process_experimental_conditions(project.acquisition_areas)

    table = Table(title="Preprocessing Status")
    table.add_column("AA")
    table.add_column("cisTEM")
    table.add_column("Unblur")
    table.add_column("Ctffind")
    table.add_column("Montage")

    for aa in project.acquisition_areas:
        table.add_row(
            aa.area_name,
            f"✓ {get_num_movies(aa.cistem_project)}" if aa.cistem_project is not None else ":x:",
            f"✓ {get_num_images(aa.cistem_project)}" if aa.unblur_run else ":x:",
            "✓" if aa.ctffind_run else ":x:",
            "✓" if aa.montage_image is not None else ":x:",
        )
    print(table)

    match_template_table = Table(title="Match Template Status")
    match_template_table.add_column("AA")
    
    for mtm in project.match_template_runs:
        match_template_table.add_column(mtm.run_name)
        
    for aa in project.acquisition_areas:
        mtm_status = []
        for mtm in project.match_template_runs:        
            mtm_status.append(f"{get_num_already_processed_images(aa.cistem_project, mtm.run_id)}/{get_num_images(aa.cistem_project)}")
        
        match_template_table.add_row(
            aa.area_name, *mtm_status)
    
    print(match_template_table)

@app.command()
def update_database(
    project_main: Path = typer.Option(None, help="Path to wanted project file")
):
    from pycistem.core import Project

    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)
    
    for aa in project.acquisition_areas:
        pro = Project()
        pro.OpenProjectFromFile(Path(aa.cistem_project).as_posix())
        pro.database.CheckandUpdateSchema()
        pro.Close(True,True)

@app.command()
def redo_projects(
    project_main: Path = typer.Option(None, help="Path to wanted project file")
):
   
    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)

    for aa in project.acquisition_areas:
        aa.cistem_project = None
    
    project.write()

@app.command()
def redo_unblur(
    project_main: Path = typer.Option(None, help="Path to wanted project file")
):
   
    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)

    for aa in project.acquisition_areas:
        aa.unblur_run = False
    
    project.write()

@app.command()
def redo_ctffind(
    project_main: Path = typer.Option(None, help="Path to wanted project file")
):
   
    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)

    for aa in project.acquisition_areas:
        aa.ctffind_run = False
    
    project.write()

@app.command()
def run_ctffind(
    project_main: Path = typer.Option(None, help="Path to wanted project file"),
    num_cores: int = typer.Option(10, help="Number of cores to use"),
):
    from pycistem.programs import ctffind
    import pycistem
    pycistem.set_cistem_path("/groups/cryoadmin/software/CISTEM/je_dev/")

    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)

    for aa in project.acquisition_areas:
        if aa.ctffind_run:
            continue
        typer.echo(f"Running ctffind for {aa.area_name}")
        pars, image_info = ctffind.parameters_from_database(aa.cistem_project,decolace=True)

        res = ctffind.run(pars,num_procs=num_cores)

        ctffind.write_results_to_database(aa.cistem_project,pars,res,image_info)
        aa.ctffind_run = True
        project.write()

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
def run_matchtemplate(
    template: Path = typer.Option(None, help="Path to wanted template file"),
    match_template_job_id: int = typer.Option(None, help="ID of template match job"),
    angular_step: float = typer.Option(3.0, help="Angular step for template matching"),
    in_plane_angular_step: float = typer.Option(2.0, help="In plane angular step for template matching"),
    defocus_step: float = typer.Option(0.0, help="Defocus step for template matching"),
    defocus_range: float = typer.Option(0.0, help="Defocus range for template matching"),
    project_main: Path = typer.Option(None, help="Path to wanted project file")
):
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
    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)
    new_mtr = True
    if match_template_job_id is None:
        if len(project.match_template_runs) == 0:
            match_template_job_id = 1
        else:
            match_template_job_id = max([mtr.run_id  for mtr in project.match_template_runs]) + 1
        if template is None:
            typer.echo("No template file or template match job id given")
            return
    else:
        if match_template_job_id in [mtr.run_id  for mtr in project.match_template_runs]:
            new_mtr = False
            # Get position in array
            array_position = [mtr.run_id  for mtr in project.match_template_runs].index(match_template_job_id)
            template = Path(project.match_template_runs[array_position].template_path)
            angular_step = project.match_template_runs[array_position].angular_step
            in_plane_angular_step = project.match_template_runs[array_position].in_plane_angular_step
            defocus_step = project.match_template_runs[array_position].defocus_step
            defocus_range = project.match_template_runs[array_position].defocus_range

            typer.echo(f"Template match job id already exists, continuing job {project.match_template_runs[array_position].run_name}")
            typer.echo(f"template_filename={template.absolute().as_posix()}, angular_step={angular_step}, in_plane_angular_step={in_plane_angular_step} defous_step={defocus_step}, defocus_range={defocus_range}, decolace=True)")

    if new_mtr:
        project.match_template_runs.append(
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
        project.write()
            

    all_image_info=[]
    for aa in project.acquisition_areas:
        typer.echo(f"Running matchtemplate for {aa.area_name}")
        image_info = match_template.parameters_from_database(aa.cistem_project, template_filename=template.absolute().as_posix(), match_template_job_id=match_template_job_id, decolace=True)
        orig_num= len(image_info)
        for par in image_info["PARAMETERS"]:
            par.angular_step = angular_step
            par.in_plane_angular_step = in_plane_angular_step
            par.defocus_step = defocus_step
            par.defocus_search_range = defocus_range
            par.max_threads = 2
        
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
    project_main: Path = typer.Option(None, help="Path to wanted project file"),
    match_template_job_id: int = typer.Option(None, help="ID of template match job"),
):
    from pycistem.database import get_num_already_processed_images, write_match_template_to_starfile, insert_tmpackage_into_db, get_num_images
    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)

    for aa in project.acquisition_areas:
        if get_num_already_processed_images(aa.cistem_project, match_template_job_id) != get_num_images(aa.cistem_project):
            typer.echo(f"No images processed for {aa.area_name}")
            continue
        
        typer.echo(f"Creating tm package for {aa.area_name}")
        output_filename = Path(aa.cistem_project).parent / "Assets" / "TemplateMatching" / f"{aa.area_name}_{match_template_job_id}_tm_package.star"
        if output_filename.exists():
            typer.echo(f"Package already exists for {aa.area_name}")
            continue
        write_match_template_to_starfile(aa.cistem_project, match_template_job_id, output_filename)  
        insert_tmpackage_into_db(aa.cistem_project, f"DeCOoLACE_{aa.area_name}_TMRun_{match_template_job_id}", output_filename)
        
@app.command()
def run_refinetemplate(
    project_main: Path = typer.Option(None, help="Path to wanted project file"),
    match_template_job_id: int = typer.Option(None, help="ID of template match job"),
):
    from pycistem.programs import refine_template_dev
    from pycistem.config import set_cistem_path
    set_cistem_path("/groups/cryoadmin/software/CISTEM/je_dev/")
    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)

    array_position = [mtr.run_id  for mtr in project.match_template_runs].index(match_template_job_id)
    mtr = project.match_template_runs[array_position]
    for aa in project.acquisition_areas:
        tm_package_file = Path(aa.cistem_project).parent / "Assets" / "TemplateMatching" / f"{aa.area_name}_{match_template_job_id}_tm_package.star"
        if not tm_package_file.exists():
            typer.echo(f"No tm package for {aa.area_name}")
            continue
        typer.echo(f"Running refine template for {aa.area_name}")
        par = refine_template_dev.RefineTemplateDevParameters(
            input_starfile=tm_package_file.as_posix(),
            output_starfile=tm_package_file.with_suffix('').as_posix()+'_refined.star',
            input_template=Path(mtr.template_path).as_posix(),
            num_threads=40
        )
        if Path(par.output_starfile).exists():
            typer.echo(f"Refined tm package already exists for {aa.area_name}")
            continue
        refine_template_dev.run(par,save_output=True,save_output_path="/tmp/output")

@app.command()
def assemble_matches(
    ctx: typer.Context,
    use_filtered: bool = typer.Option(False, help="Use filtered matches"),
):
    import starfile
    import decolace.processing.decolace_processing as dp
    if ctx.obj.match_template_job is None:
        typer.echo("No match template job given")
        raise typer.Exit()
    for aa in ctx.obj.acquisition_areas:
        if use_filtered:
            refined_matches_file = Path(aa.cistem_project).parent / "Assets" / "TemplateMatching" / f"{aa.area_name}_{ctx.obj.match_template_job.run_id}_tm_package_filtered.star"
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
        if use_filtered:
            output_star_path = output_dir / f"{aa.area_name}_{ctx.obj.match_template_job.run_name}_{ctx.obj.match_template_job.run_id}_filtered.star"
        else:
            output_star_path = output_dir / f"{aa.area_name}_{ctx.obj.match_template_job.run_name}_{ctx.obj.match_template_job.run_id}.star"
        montage_data = starfile.read(aa.montage_star)

        # Read matches data
        matches_data = starfile.read(refined_matches_file)

        result = dp.assemble_matches(montage_data, matches_data)

        starfile.write(result, output_star_path, overwrite=True)


@app.command()
def filter_matches(
    ctx: typer.Context,
    erode_mask: int = 128
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
        refined_matches_starfile = Path(aa.cistem_project).parent / "Assets" / "TemplateMatching" / f"{aa.area_name}_{ctx.obj.match_template_job.run_id}_tm_package_refined.star"
        if not refined_matches_starfile.exists():
            typer.echo(f"No refined matches for {aa.area_name}")
            continue
        refined_matches = starfile.read(refined_matches_starfile)
        print(f"Starting with {len(refined_matches)} matches")
        refined_matches = refined_matches[refined_matches["cisTEMScore"] >= 8.0]
        print(f"After 8.0 score criterion {len(refined_matches)} matches")
        refined_matches["tile_mask_filename"] = refined_matches["cisTEMOriginalImageFilename"].str.replace(".mrc", "_mask.mrc")
        refined_matches["tile_mask_filename"].str.replace("'", "")
        for mask_filename in refined_matches["tile_mask_filename"].unique():
        # Open mask and convert to 0-1 floats
            if mask_filename is None:
                continue
            with mrcfile.open(mask_filename.strip("'")) as mask:
                mask_data = np.copy(mask.data[0])
                mask_data.dtype = np.uint8
                mask_float = mask_data / 255.0
                if erode_mask > 0:
                    binary_mask = mask_float > 0.5
                    binary_mask = binary_erosion(binary_mask, iterations=erode_mask)
                    mask_float = binary_mask * 1.0
            #
            peak_indices = refined_matches.loc[
                refined_matches["tile_mask_filename"] == mask_filename
            ].index.tolist()
            for i in peak_indices:
                pixel_position_x = int(refined_matches.loc[i, "cisTEMOriginalXPosition"] / refined_matches.loc[i, "cisTEMPixelSize"])
                pixel_position_y = int(refined_matches.loc[i, "cisTEMOriginalYPosition"] / refined_matches.loc[i, "cisTEMPixelSize"])
                try:
                    refined_matches.loc[i, "mask_value"] = mask_float[
                        pixel_position_y,
                        pixel_position_x,
                    ]
                    if pixel_position_x < 128 or pixel_position_y < 128:
                        refined_matches.loc[i, "display"] = False
                        continue
                    if pixel_position_x > mask_float.shape[1] - 128 or pixel_position_y > mask_float.shape[0] - 128:
                        refined_matches.loc[i, "display"] = False
                        continue
                except IndexError:
                    refined_matches.loc[i, "mask_value"] = 0
                if refined_matches.loc[i, "mask_value"] > 0.7:
                    refined_matches.loc[i, "display"] = True
        refined_matches = refined_matches[refined_matches["display"] == True]
        print(f"After masked area removal {len(refined_matches)} matches")
        filtered_matches_starfile = Path(aa.cistem_project).parent / "Assets" / "TemplateMatching" / f"{aa.area_name}_{ctx.obj.match_template_job.run_id}_tm_package_filtered.star"
        starfile.write(refined_matches, filtered_matches_starfile, overwrite=True)

@app.command()
def join_matches(
    ctx: typer.Context,
):
    import starfile 
    import pandas as pd
    from pycistem.database import create_project, insert_tmpackage_into_db

    matches = []
    if ctx.obj.match_template_job is None:
        typer.echo("No match template job given")
        raise typer.Exit()
    
    for aa in ctx.obj.acquisition_areas:
        filtered_matches_starfile = Path(aa.cistem_project).parent / "Assets" / "TemplateMatching" / f"{aa.area_name}_{ctx.obj.match_template_job.run_id}_tm_package_filtered.star"
        if filtered_matches_starfile.exists():
            matches.append(starfile.read(filtered_matches_starfile))
    combined_matches = pd.concat(matches)
    combined_matches_starfile = Path(ctx.obj.project.project_path) / "Matches" / f"combined_{ctx.obj.match_template_job.run_name}_{ctx.obj.match_template_job.run_id}_filtered.star"
    starfile.write(combined_matches, combined_matches_starfile, overwrite=True)

    database = create_project("processing", ctx.obj.project.project_path.absolute() / "cistem_projects")
    print(database)
    insert_tmpackage_into_db(database, f"DeCOoLACE_combind_TMRun_{ctx.obj.match_template_job.run_name}_{ctx.obj.match_template_job.run_id}_{len(combined_matches)}", combined_matches_starfile)


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

@app.callback()
def main(
    ctx: typer.Context,
    project: Path = typer.Option(None, help="Path to wanted project file"),
    acquisition_areas: List[int] = typer.Option(None, help="List of acquisition areas to process"),
    match_template_job_id: int = typer.Option(None, help="ID of template match job"),

):
    """DeCoLACE processing pipeline"""
    if project is None:
        potential_projects = glob.glob("*.decolace")
        if len(potential_projects) == 0:
            typer.echo("No project file found")
            raise typer.Exit()
        project = Path(potential_projects[0])
    project_object = ProcessingProject.read(project)
    if len(acquisition_areas) > 0:
        aas_to_process = [aa for i, aa in enumerate(project_object.acquisition_areas) if i in acquisition_areas]
    else:
        aas_to_process = project_object.acquisition_areas
    if match_template_job_id is not None:
        array_position = [mtr.run_id  for mtr in project_object.match_template_runs].index(match_template_job_id)
        mtr = project_object.match_template_runs[array_position]
    else:
        mtr = None
    ctx.obj = SimpleNamespace(project = project_object, acquisition_areas = aas_to_process, match_template_job = mtr)
