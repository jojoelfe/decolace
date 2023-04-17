from pathlib import Path

import numpy as np
import starfile
import typer
from decolace_processing import (
    calculate_refined_image_shifts,
    calculate_refined_intensity,
    calculate_shifts,
    find_tile_pairs,
)
from rich.console import Console

console = Console()

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    tile_star_path: Path = typer.Argument(
        ..., help="Path to the DeCoLACE tile star file"
    ),
    iterations: int = typer.Option(2, help="Number of iterations"),
    num_procs: int = typer.Option(10, help="Number of processes to use"),
    output_path_metadata: Path = typer.Argument(
        ..., help="Path to the output star file"
    ),
):
    # Read decolace data
    tile_data = starfile.read(tile_star_path)
    # CHeck if tile_data is a dict
    if isinstance(tile_data, dict):
        tile_data = tile_data["tiles"]
    # Estimate distance threshold
    estimated_distance_threshold = np.median(
        tile_data["tile_x_size"] * tile_data["tile_pixel_size"]
    )
    for iteration in range(iterations):
        console.log(f"Starting iteration {iteration+1} of {iterations}.")
        # Calculate tile paris
        tile_pairs = find_tile_pairs(tile_data, estimated_distance_threshold)
        console.log(
            f"In {len(tile_data)} tiles {len(tile_pairs)} tile pairs were found using distance threshold of {estimated_distance_threshold:.2f} A."
        )
        # Calculate shifts of tile pairs
        shifts = calculate_shifts(tile_pairs, num_procs)
        console.log(
            f"Shifts were adjusted by an average of {np.mean(shifts['add_shift']):.2f} pixels."
        )
        # Optimize tile positions
        calculate_refined_image_shifts(tile_data, shifts)
        # Optimize intensities
        calculate_refined_intensity(tile_data, shifts)
    # Write new tile data
    starfile.write(tile_data, output_path_metadata, overwrite=True)
    console.log(f"Wrote refined tile metadata to {output_path_metadata}.")


if __name__ == "__main__":
    app()
