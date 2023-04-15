import sqlite3
import pandas as pd
import typer
from pathlib import Path

app = typer.Typer()


@app.command()
def create_project(project_main: Path = typer.Argument(
        ..., help="Path to wanted project file"
    )):

    # Connect to database
    conn = sqlite3.connect(project_main)

    # Create row with overall project information
    project_info = pd.row({'project_name': project_main.stem, 'project_path': project_main.parent})

    # Create empty dataframe with columns for acquisition_areas
    aa_info = pd.DataFrame(columns=['name', 
                                    'decolace_acquisition_info_path', 
                                    'frames_folder', 
                                    'cisTEM project', 
                                    'initial_tile_star', ])
    aa_info.to_sql('acquisition_areas', conn, if_exists='fail', index=False)
    project_info.to_sql('project_info', conn, if_exists='fail', index=False)
    conn.close()
    typer.echo(f"Created project file {project_main}")