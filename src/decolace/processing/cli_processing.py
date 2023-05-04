from pathlib import Path
from typing import Optional, Union

import typer
import glob
from rich import print
import numpy as np
import pandas as pd
from decolace.processing.project_managment import new_project, open_project, read_project, acquisition_area_schema
from decolace.acquisition.session import session
app = typer.Typer()


@app.command()
def create_project(
    name: str = typer.Argument(..., help="Name of project"),
    directory: Optional[Path] = typer.Argument(
        ..., help="Directory to create project in"
    ),
):

    if directory is None:
        directory = Path.cwd()
    new_project(name, directory)


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
    project_main: Path = typer.Argument(None, help="Path to wanted project file"),
    session_name: str = typer.Argument(..., help="Name of session"),
    session_directory: Path = typer.Argument(..., help="Path to session directory"),
):
    _, aa_info_existing = read_project(open_project(project_main))
    
    my_session = session(session_name, session_directory.as_posix())
    my_session.load_from_disk()
    data = []
    for grid in my_session.grids:
        for aa in grid.acquisition_areas:
            to_append = {key:default for key, _, default in acquisition_area_schema}
            to_append["session"] = my_session.name
            to_append["grid"] = grid.name
            to_append["name"] = aa.name
            to_append["decolace_acquisition_info_path"] = aa.directory
            to_append["frames_folder"] = aa.frames_directory
            data.append(tuple(to_append.values()))

    aa_info_new = pd.DataFrame(np.array(data))
    aa_info = pd.concat([aa_info_existing, aa_info_new])
    print(aa_info_new)