from decolace_processing import create_montage_metadata, create_montage
import typer
from rich.console import Console
from pathlib import Path
import starfile
console = Console()

def main(
    tile_star_path: Path = typer.Argument(
        ..., help="Path to the DeCoLACE tile star file"
    ),
    output_path_metadata: Path = typer.Argument(
        ..., help="Path to the output star file"
    ),
    output_path_montage: Path = typer.Argument(
        ..., help="Path to the output image file"
    ),
    binning: int = typer.Option(10, help="Binning factor"),
):
    # Read decolace data
    tile_data = starfile.read(tile_star_path)
    # Create montage metadata
    montage_metadata = create_montage_metadata(tile_data, output_path_metadata, binning, output_path_montage)
    console.log(f"Wrote montage metadata to {output_path_metadata}.")
    create_montage(montage_metadata, output_path_montage)
    console.log(f"Wrote {montage_metadata['montage']['montage_x_size'].values[0]}x{montage_metadata['montage']['montage_y_size'].values[0]} montage to {output_path_montage}.")

if __name__ == "__main__":
    typer.run(main)
