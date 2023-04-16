import sqlite3
import pandas as pd
import typer
from pathlib import Path

app = typer.Typer()

@app.command()
def new_sesion(name: str = typer.Argument(..., help="Name of the session")):
    typer.echo(f"Created new session {name}")


@app.command()
def new_grid(name: str = typer.Argument(..., help="Name of the grid")):
    typer.echo(f"Created new grid {name}")

@app.command()
def add_acquisition_area():
    typer.echo(f"Added acquisition area")

@app.command()
def image_lamella():
    typer.echo(f"Image lamella")

@app.command()
def acquire():
    typer.echo(f"Acquire")