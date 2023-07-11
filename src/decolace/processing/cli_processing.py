from pathlib import Path
from typing import Optional, Union, List

import typer
import glob
from rich import print
from rich.table import Table
import numpy as np
import pandas as pd
import logging
from rich.logging import RichHandler

from decolace.processing.project_managment import ProcessingProject, AcquisitionAreaPreProcessing, MatchTemplateRun
from decolace.acquisition.session import session

from decolace.processing.decolace_processing import read_data_from_cistem, read_decolace_data


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
    acquisition_area: Path = typer.Argument(
        ..., help="Path to wanted acquisition area file"
    ),
    project: Path = typer.Option(..., help="Path to wanted project file"),
):
    open_project(project)

    typer.echo("Added acquisition area to ")


@app.command()
def add_session(
    project_main: Path = typer.Option(None, help="Path to wanted project file"),
    session_name: str = typer.Argument(..., help="Name of session"),
    session_directory: Path = typer.Argument(..., help="Path to session directory"),
    ignore_grids: List[str] = typer.Option(["cross"], help="Grids to ignore"),
):
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

@app.command()
def status(
    project_main: Path = typer.Option(None, help="Path to wanted project file")
):
    from pycistem.database import get_num_movies, get_num_images

    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)

    table = Table(title="Project Status")
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
        pro.OpenProjectFromFile(aa.cistem_project.as_posix())
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
    project_main: Path = typer.Option(None, help="Path to wanted project file")
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

        res = ctffind.run(pars,num_procs=40)

        ctffind.write_results_to_database(aa.cistem_project,pars,res,image_info)
        aa.ctffind_run = True
        project.write()

@app.command()
def run_montage(
    project_main: Path = typer.Option(None, help="Path to wanted project file"),
    iterations: int = typer.Option(2, help="Number of iterations"),
    num_procs: int = typer.Option(10, help="Number of processors to use"),
    binning: int = typer.Option(10, help="Binning factor"),
):
    from rich.console import Console
    import starfile

    console = Console()
    
    from decolace.processing.decolace_processing import create_tile_metadata, find_tile_pairs, calculate_shifts, calculate_refined_image_shifts, calculate_refined_intensity, create_montage_metadata, create_montage

    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)

    for aa in project.acquisition_areas:
        if aa.montage_image is not None:
            continue
        output_directory = project.project_path / "Montages" / aa.area_name
        typer.echo(f"Running montage for {aa.area_name}")
        if aa.initial_tile_star is None:
            cistem_data = read_data_from_cistem(aa.cistem_project)
            typer.echo(
                f"Read data about {len(cistem_data)} tiles."
            )
            # Read decolace data
            decolace_data = read_decolace_data(aa.decolace_session_info_path)
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
            for iteration in range(iterations):
                console.log(f"Starting iteration {iteration+1} of {iterations}.")
                # Calculate tile paris
                tile_pairs = find_tile_pairs(tile_metadata, estimated_distance_threshold)
                console.log(
                    f"In {len(tile_metadata)} tiles {len(tile_pairs)} tile pairs were found using distance threshold of {estimated_distance_threshold:.2f} A."
                )
                # Calculate shifts of tile pairs
                shifts = calculate_shifts(tile_pairs, num_procs)
                console.log(
                    f"Shifts were adjusted by an average of {np.mean(shifts['add_shift']):.2f} pixels."
                )
                # Optimize tile positions
                calculate_refined_image_shifts(tile_metadata, shifts)
                # Optimize intensities
                calculate_refined_intensity(tile_metadata, shifts)
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
            tile_metadata, output_path_metadata, binning, output_path_montage
        )
        console.log(f"Wrote montage metadata to {output_path_metadata}.")
        create_montage(montage_metadata, output_path_montage)
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
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[
            RichHandler(),
            #logging.FileHandler(current_output_directory / "log.log")
        ]
    )
    run_profile = {
        "warsaw": 8,
        "bucharest": 8,
        "istanbul": 8,
        "kyiv": 8,
        "barcelona": 8,
        "milano": 8,
        "sofia": 8,
        "manchester": 8,
        "zamor1": 8,
        "zamor2": 8,
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
        
        already_processed_images = get_already_processed_images(aa.cistem_project, match_template_job_id)
        image_info = image_info[~image_info['IMAGE_ASSET_ID'].isin(already_processed_images['IMAGE_ASSET_ID'])]
        all_image_info.append(image_info)
        typer.echo(f"{len(image_info)} tiles out of {orig_num} still to process")
        if len(image_info) == 0:
            typer.echo(f"All tiles already processed")
            continue
    all_image_info = pd.concat(all_image_info)
    typer.echo(f"Total of {len(all_image_info)} tiles to process")

    res = match_template.run(all_image_info,num_procs=generate_num_procs(run_profile),cmd_prefix=list(generate_gpu_prefix(run_profile)),cmd_suffix='"', sleep_time=2.0, write_directly_to_db=True)
        #typer.echo(f"Writing results for {aa.area_name}")

        #match_template.write_results_to_database(aa.cistem_project,pars,res,image_info)
        #typer.echo(f"Done with {aa.area_name}")
        #project.write()