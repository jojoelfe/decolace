import glob
from pathlib import Path

import mdocfile
import typer
from pycistem.core import Project
from rich.console import Console

from decolace.session import session

console = Console()


def main(
    session_name: str = typer.Argument(..., help="Session name"),
    session_directory: Path = typer.Argument(..., help="Session directory"),
    output_dir: Path = typer.Argument(
        ..., help="Directory where cisTEM projects will be created"
    ),
):
    my_session = session(session_name, session_directory.as_posix())
    my_session.load_from_disk()

    for grid in my_session.grids:
        for aa in grid.acquisition_areas:
            project_name = f"{session_name}_{grid.name}_{aa.name}"
            project = Project()
            Path(output_dir, project_name).mkdir(parents=True, exist_ok=True)
            success = project.CreateNewProject(
                Path(output_dir, project_name, f"{project_name}.db").as_posix(),
                Path(output_dir, project_name).as_posix(),
                project_name,
            )
            if not success:
                project.OpenProjectFromFile(
                    Path(output_dir, project_name, f"{project_name}.db").as_posix()
                )
            num_movies = project.database.ReturnSingleLongFromSelectCommand(
                "SELECT COUNT(*) FROM MOVIE_ASSETS;"
            )
            if num_movies == 0:
                movie_filenames = sorted(
                    glob.glob(Path(aa.directory, "frames", "*.tif").as_posix())
                )
                gain = sorted(
                    glob.glob(Path(aa.directory, "frames", "*.dm4").as_posix())
                )[-1]
                metadata_entries = []
                for i, movie in enumerate(movie_filenames):
                    metadata = mdocfile.read(movie + ".mdoc")
                    metadata_entries.append(metadata.iloc[0])
                    # Insert data in MOVIE_ASSETS_METADATA using sqlite3
                    project.database.ExecuteSQL(
                        f"INSERT INTO MOVIE_ASSETS_METADATA "
                        f"(MOVIE_ASSET_ID,"
                        f"METADATA_SOURCE,"
                        f"CONTENT_JSON,"
                        f"TILT_ANGLE,"
                        f"STAGE_POSITION_X,"
                        f"STAGE_POSITION_Y,"
                        f"STAGE_POSITION_Z,"
                        f"IMAGE_SHIFT_X,"
                        f"IMAGE_SHIFT_Y,"
                        f"EXPOSURE_DOSE,"
                        f"ACQUISITION_TIME)"
                        f"VALUES ({i+1},"
                        f"'serialem_frames_mdoc',"
                        f"'{metadata.iloc[0].to_json(default_handler=str)}',"
                        f" {metadata.loc[0,'TiltAngle']},"
                        f" {metadata.loc[0,'StagePosition'][0]},"
                        f" {metadata.loc[0,'StagePosition'][1]},"
                        f" {metadata.loc[0,'StageZ']},"
                        f" {metadata.loc[0,'ImageShift'][0]},"
                        f" {metadata.loc[0,'ImageShift'][1]},"
                        f" {metadata.loc[0,'ExposureDose']}, 0);"
                    )

                project.database.BeginMovieAssetInsert()
                for i, movie in enumerate(movie_filenames):

                    project.database.AddNextMovieAsset(
                        i + 1,
                        Path(movie).name,
                        movie,
                        0,
                        11520,
                        8184,
                        34,
                        300,
                        0.53,
                        0.8,
                        2.7,
                        gain,
                        "",
                        3.774,
                        0,
                        0,
                        1.0,
                        1.0,
                        0,
                        25,
                        1,
                    )
                project.database.EndMovieAssetInsert()

            project.database.Close(True)


if __name__ == "__main__":
    typer.run(main)
