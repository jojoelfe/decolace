import glob
import os
from pathlib import Path

import typer

from .session import session

app = typer.Typer()


def load_session(name, directory):
    if directory is None:
        directory = Path(os.getcwd()).as_posix()
    else:
        directory = Path(directory).as_posix()
    if name is None:
        potential_files = glob.glob(os.path.join(directory, "*.npy"))
        if len(potential_files) < 1:
            raise (FileNotFoundError("Couldn't find saved files"))
        most_recent = sorted(potential_files)[-1]
        name = os.path.basename(most_recent).split("_")[0]
    session_o = session(name, directory)
    session_o.load_from_disk()
    return session_o


@app.command()
def new_session(
    name: str = typer.Argument(..., help="Name of the session"),
    directory: str = typer.Option(None, help="Directory to save session in"),
):
    if directory is None:
        directory = os.getcwd()
    else:
        directory = Path(directory).as_posix()
    session_o = session(name, directory)
    session_o.write_to_disk()
    typer.echo(f"Created new session {name} in {directory}")


@app.command()
def save_microscope_settings(
    name: str = typer.Option(..., help="Name of the session"),
    directory: str = typer.Option(..., help="Directory to save session in"),
):

    session_o = load_session(name, directory)
    session_o.add_current_setting()
    session_o.write_to_disk()
    typer.echo(f"Saved microscope settings for session {session_o.name}")


@app.command()
def new_grid(
    name: str = typer.Argument(..., help="Name of the grid"),
    session_name: str = typer.Option(None, help="Name of the session"),
    directory: str = typer.Option(None, help="Directory the session is saved in"),
):
    session_o = load_session(session_name, directory)
    session_o.add_grid(name)
    session_o.write_to_disk()
    typer.echo(f"Created new grid {name} for session {session_o.name}")


@app.command()
def euc_and_nice_view(
    session_name: str = typer.Option(None, help="Name of the session"),
    directory: str = typer.Option(None, help="Directory to save session in"),
):
    session_o = load_session(session_name, directory)
    session_o.active_grid.eucentric()
    session_o.active_grid.nice_view()
    session_o.active_grid.write_to_disk()


@app.command()
def status(
    session_name: str = typer.Option(None, help="Name of the session"),
    directory: str = typer.Option(None, help="Directory to save session in"),
):
    session_o = load_session(session_name, directory)
    typer.echo(f"Session: {session_o.name} contains {len(session_o.grids)} grids")
    typer.echo(f"Active grid: {session_o.active_grid.name}")
    for grid in session_o.grids:
        typer.echo(
            f"Grid: {grid.name} contains {len(grid.acquisition_areas)} acquisition areas"
        )


@app.command()
def new_map(
    session_name: str = typer.Option(..., help="Name of the session"),
    directory: str = typer.Option(..., help="Directory to save session in"),
):
    session_o = load_session(session_name, directory)
    session_o.active_grid.take_map()
    session_o.write_to_disk()
    typer.echo(f"Created new map for grid {session_o.active_grid.name}")


@app.command()
def add_acquisition_area():
    typer.echo("Added acquisition area")


@app.command()
def image_lamella():
    typer.echo("Image lamella")


@app.command()
def acquire():
    typer.echo("Acquire")
