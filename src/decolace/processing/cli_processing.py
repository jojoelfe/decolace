from pathlib import Path
from typing import Optional, Union, List

import typer
import glob
from rich import print
from rich.table import Table
import numpy as np
import pandas as pd
from decolace.processing.project_managment import ProcessingProject, AcquisitionAreaPreProcessing
from decolace.acquisition.session import session

from decolace.processing.create_cistem_projects_for_session import create_project_for_area 

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
):
    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)

    for aa in project.acquisition_areas:
        if aa.cistem_project is not None:
            continue
        typer.echo(f"Creating cistem project for {aa.area_name}")
        cistem_project_path = create_project_for_area(aa.area_name, project_path.parent.absolute() / "cistem_projects", aa.frames_folder)
        aa.cistem_project = cistem_project_path
        typer.echo(f"Saved as {cistem_project_path}")
    project.write()
   
@app.command()
def run_unblur(
    project_main: Path = typer.Option(None, help="Path to wanted project file")
):
    from pycistem.programs import unblur
    import pycistem
    pycistem.set_cistem_path("/scratch/paris/elferich/cisTEM/build/je_combined_Intel-gpu-debug-static/src/")

    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)

    for aa in project.acquisition_areas:
        if aa.unblur_run:
            continue
        typer.echo(f"Running unblur for {aa.area_name}")
        pars = unblur.parameters_from_database(aa.cistem_project,decolace=True)

        res = unblur.run(pars,num_procs=40)

        unblur.write_results_to_database(aa.cistem_project,pars,res)
        aa.unblur_run = True
        project.write()

@app.command()
def status(
    project_main: Path = typer.Option(None, help="Path to wanted project file")
):
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
            "✓" if aa.cistem_project is not None else ":x:",
            "✓" if aa.unblur_run else ":x:",
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
    pycistem.set_cistem_path("/scratch/paris/elferich/cisTEM/build/je_combined_Intel-gpu-debug-static/src/")

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