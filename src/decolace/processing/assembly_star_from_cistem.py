from pathlib import Path

import typer
from decolace_processing import (
    create_tile_metadata,
    read_data_from_cistem,
    read_decolace_data,
)
from rich.console import Console

console = Console()


def main(
    cistem_data_path: Path = typer.Argument(..., help="Path to the CisTEM database"),
    decolace_data_path: Path = typer.Argument(
        ..., help="Path to the Decolace data file"
    ),
    output_path: Path = typer.Argument(..., help="Path to the output star file"),
):
    # Read data from Cistem
    cistem_data = read_data_from_cistem(cistem_data_path)
    console.log(
        f"Read data about {len(cistem_data)} tiles from CisTEM project {cistem_data_path}."
    )
    # Read decolace data
    decolace_data = read_decolace_data(decolace_data_path)
    # Create tile metadata
    create_tile_metadata(cistem_data, decolace_data, output_path)
    console.log(f"Wrote tile metadata to {output_path}.")


if __name__ == "__main__":
    typer.run(main)
