from pathlib import Path
from typing import Optional, Union

import typer

from decolace.processing.project_managment import new_project, open_project

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
    project: Union[str, Path] = typer.Option(..., help="Path to wanted project file"),
):
    open_project(project)

    typer.echo("Added acquisition area to ")


@app.command()
def add_grid(
    project_main: Path = typer.Argument(..., help="Path to wanted project file"),
):
    typer.echo(f"Added grid to {project_main}")
