import typer
from rich.console import Console
from rich.progress import track
from pathlib import Path
import starfile
import numpy as np
from decolace_processing import assemble_matches
console = Console()

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
    montage_star_path: Path = typer.Argument(
        ..., help="Path to the DeCoLACE montage star file"
    ),
    matches_star_path: Path = typer.Argument(
        ..., help="Path to the cisTEM TM package star file"
    ),
    output_star_path: Path = typer.Argument(
        ..., help="Path to the output star file"
    )
):
    # Read decolace data
    montage_data = starfile.read(montage_star_path)

    # Read matches data
    matches_data = starfile.read(matches_star_path)

    result = assemble_matches(montage_data, matches_data)

    starfile.write(result, output_star_path, overwrite=True)

if __name__ == "__main__":
    typer.run(main)
    
    