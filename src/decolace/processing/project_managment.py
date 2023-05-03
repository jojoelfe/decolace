import sqlite3
from pathlib import Path
from typing import Union, List

import pandas as pd
import numpy as np
from pydantic import BaseModel

project_info_schema = [
    ("project_name", str, "project"),
    ("project_path", str, "/tmp/project"),
    ("processing_pixel_size", float, 2.0),
    ("unblur_arguments", object, {"align_cropped_area": True, "BB": False}),
    ("ctffind_arguments", object, {}),
]

class AcquisitionArea(BaseModel):
    name: str
    decolace_acquisition_info_path: Path
    frames_folder: Path
    cistem_project: Path
    initial_tile_star: Path
    cistem_project_created: bool = False
    unblur_run: bool = False
    ctffind_run: bool = False
    tile_star_created: bool = False
    tile_positions_refined: bool = False
    mts_run: dict = {}

class ProcessingProject(BaseModel):
    name: str
    path: Path
    processing_pixel_size: float = 2.0
    unblur_arguments: dict = {"align_cropped_area": True, "BB": False}
    ctffind_arguments: dict = {}
    acquisition_areas = List[AcquisitionArea]
    

acquisition_area_schema = [
    ("name", str, "acquisition_area"),
    ("grid", str, "grid"),
    ("session", str, "session"),
    ("decolace_acquisition_info_path", str, ""),
    ("frames_folder", str, ""),
    ("cisTEM project", str, ""),
    ("initial_tile_star", str, ""),
    ("refined_tile_star", str, ""),

    ("unblur_run", bool, False),
    ("ctffind_run", bool, False),
    ("match_template_runs", object, {}),
]



def new_project(name:str, directory: Path):

def new_project(name: str, directory: Path):
    # Connect to database
    engine = create_engine(f"sqlite:///{directory / f'{name}.decolace'}", echo=False)
    sqlite_connection = engine.connect()
    # Create row with overall project information
    project_info = {key: value for key, _, value in project_info_schema}
    project_info["project_name"] = name
    project_info["project_path"] = str(directory)
    project_info = pd.DataFrame.from_dict(
        {0: [value for value in project_info.values()]},
        orient="index",
        columns=[key for key, _, _ in project_info_schema],
    )

    # Create empty dataframe with columns for acquisition_areas
    dtypes = np.dtype([(key, dtype) for key, dtype, _ in acquisition_area_schema])
    aa_info = pd.DataFrame(np.empty(0, dtype=dtypes))
    print(aa_info)
    aa_info.to_sql(
        "acquisition_areas", sqlite_connection, if_exists="fail", index=False,
        dtype={
            key: sqlalchemy.types.JSON
            for key, t, default in acquisition_area_schema
            if type(default) == dict
        }
    )
    project_info.to_sql(
        "project_info",
        sqlite_connection,
        if_exists="fail",
        index=False,
        dtype={
            key: sqlalchemy.types.JSON
            for key, _, default in project_info_schema
            if type(default) == dict
        },
    )
    sqlite_connection.close()


def open_project(filename: Union[str, Path]):

    if filename is None:
        directory = Path.cwd()
        # Find project file in directory
        for file in directory.iterdir():
            if file.suffix == ".decolace":
                if filename is not None:
                    raise ValueError(
                        "Multiple project files found in directory. Please specify which one to load."
                    )
                filename = file

    if isinstance(filename, str):
        filename = Path(filename)

    if not filename.exists():
        if filename.suffix != ".decolace":
            filename = filename.with_suffix(".decolace")
        if not filename.exists():
            raise FileNotFoundError(f"Project {filename} can't be found.")

    engine = create_engine(f"sqlite:///{filename}", echo=False)
    sqlite_connection = engine.connect()
    return sqlite_connection

def read_project(sqlite_connection):
    # Read in project_info table
    project_info = pd.read_sql("project_info", sqlite_connection)
    # Read in acquisition_areas table
    aa_info = pd.read_sql("acquisition_areas", sqlite_connection)
    return project_info, aa_info

def update_project(sqlite_connection):
    # Check if the columns in the database are up to date
    # If not, add them
    # Start with project_info_schema
    # Read in project_info table

    project_info = pd.read_sql("project_info", sqlite_connection)
    for key, dtype, default in project_info_schema:
        if key not in project_info.columns:
            project_info[key] = default
    project_info.to_sql("project_info", sqlite_connection, if_exists="replace")

    # Then do the same for acquisition_areas
    aa_info = pd.read_sql("acquisition_areas", sqlite_connection)
    for key, dtype, default in acquisition_area_schema:
        if key not in aa_info.columns:
            aa_info[key] = default
    aa_info.to_sql("acquisition_areas", sqlite_connection, if_exists="replace")

    return()

    pass
