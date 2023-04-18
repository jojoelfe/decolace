from pathlib import Path

import starfile
import typer
from decolace_processing import (
    adjust_metadata_for_matches,
    create_montage,
    read_matches_data,
)
from rich.console import Console

console = Console()


def main(
    cistem_project_path: Path = typer.Argument(
        ..., help="Path to the Cistem project directory"
    ),
    montage_star_path: Path = typer.Argument(
        ..., help="Path to the DeCoLACE tile star file"
    ),
    output_path_montage: Path = typer.Argument(
        ..., help="Path to the output image file"
    ),
    tm_job_id: int = typer.Option(1, help="TM job ID"),
):
    # Read decolace data
    montage_data = starfile.read(montage_star_path)
    # Match data
    match_data = read_matches_data(cistem_project_path, tm_job_id)
    # Create montage metadata
    matches_montage_metadata = adjust_metadata_for_matches(montage_data, match_data)
    # console.log(f"Wrote montage metadata to {output_path_metadata}.")
    create_montage(matches_montage_metadata, output_path_montage)
    console.log(
        f"Wrote {matches_montage_metadata['montage']['montage_x_size'].values[0]}x{matches_montage_metadata['montage']['montage_y_size'].values[0]} montage to {output_path_montage}."
    )


if __name__ == "__main__":
    typer.run(main)
