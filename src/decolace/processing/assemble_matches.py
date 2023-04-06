import typer
from rich.console import Console
from rich.progress import track
from pathlib import Path
import starfile
import numpy as np
from decolace_processing import 
console = Console()

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
    montage_star_path: Path = typer.Argument(
        ..., help="Path to the DeCoLACE montage star file"
    ),
    matches_star_path: Path = typer.Argument(
        ..., help="Path to the cisTEM TM package star file"
    )
):
    # Read decolace data
    montage_data = starfile.read(montage_star_path)
    # CHeck if tile_data is a dict
    if isinstance(montage_data, dict):
        montage_data = montage_data['montage']
    # Read matches data
    matches_data = starfile.read(matches_star_path)
    
    