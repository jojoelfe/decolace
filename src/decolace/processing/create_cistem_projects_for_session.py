from decolace.session import session
import typer
from rich.console import Console
from pathlib import Path
console = Console()
from pycistem.core import Project, Database
import glob
import mdocfile

def main(
    session_name: str = typer.Argument(
        ..., help="Session name"
    ),
    session_directory: Path = typer.Argument(
        ..., help="Session directory"
    ),
    output_dir: Path = typer.Argument(
        ..., help="Directory where cisTEM projects will be created"
    )):
        my_session = session(session_name, session_directory.as_posix())
        my_session.load_from_disk()

        for grid in my_session.grids:
            for aa in grid.acquisition_areas:
                project_name = f"{session_name}_{grid.name}_{aa.name}"
                project = Project()
                Path(output_dir, project_name).mkdir(parents=True, exist_ok=True)
                success = project.CreateNewProject(Path(output_dir, project_name,f"{project_name}.db").as_posix(),Path(output_dir, project_name).as_posix(),project_name)
                if not success:
                    project.OpenProjectFromFile(Path(output_dir, project_name,f"{project_name}.db").as_posix())
                num_movies = project.database.ReturnSingleLongFromSelectCommand("SELECT COUNT(*) FROM MOVIE_ASSETS;")
                if num_movies == 0:
                    movie_filenames = sorted(glob.glob(Path(aa.directory, "frames","*.tif").as_posix()))
                    gain = sorted(glob.glob(Path(aa.directory, "frames","*.dm4").as_posix()))[-1]
                    metadata_entries = []
                    for i, movie in enumerate(movie_filenames):
                        metadata = mdocfile.read(movie+".mdoc")
                        metadata_entries.append(metadata.iloc[0])
                        # Insert data in MOVIE_ASSETS_METADATA using sqlite3
                        project.database.ExecuteSQL(
                             f"INSERT INTO MOVIE_ASSETS_METADATA (MOVIE_ASSET_ID, METADATA_SOURCE, CONTENT_JSON, TILT_ANGLE, STAGE_POSITION_X, STAGE_POSITION_Y, STAGE_POSITION_Z, IMAGE_SHIFT_X, IMAGE_SHIFT_Y, EXPOSURE_DOSE, ACQUISITION_TIME) VALUES ({i+1}, 'serialem_frames_mdoc', '{metadata.iloc[0].to_json(default_handler=str)}', {metadata.loc[0,'TiltAngle']}, {metadata.loc[0,'StagePosition'][0]}, {metadata.loc[0,'StagePosition'][1]}, {metadata.loc[0,'StageZ']}, {metadata.loc[0,'ImageShift'][0]}, {metadata.loc[0,'ImageShift'][1]}, {metadata.loc[0,'ExposureDose']}, 0);"
                        )

                    project.database.BeginMovieAssetInsert()
                    for i, movie in enumerate(movie_filenames):
                        
                        project.database.AddNextMovieAsset(i+1,Path(movie).name, movie, 0, 11520, 8184, 34, 300, 0.53, 0.8, 2.7, gain, "", 3.774, 0 ,0, 1.0, 1.0, 0, 25, 1)
                    project.database.EndMovieAssetInsert()
                    
                         
                project.database.Close(True)
                        

            
        

if __name__ == "__main__":
    typer.run(main)
