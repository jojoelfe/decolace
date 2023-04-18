import sqlite3
from pathlib import Path
from typing import Union

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

project_info_schema = [
    ("project_name", str, "project"),
    ("project_path", str, "/tmp/project"),
    ("processing_pixel_size", float, 2.0),
    ("unblur_arguments", object, {"align_cropped_area": True, "BB": False}),
    ("ctffind_arguments", object, {}),
]

acquisition_area_schema = [
    ("name", str, "acquisition_area"),
    ("decolace_acquisition_info_path", str, "/tmp/acquisition_info"),
    ("frames_folder", str, "/tmp/frames"),
    ("cisTEM project", str, "/tmp/cistem.db"),
    ("initial_tile_star", str, "/tmp/initial_tile.star"),
    ("cistem_project_created", bool, False),
    ("unblur_run", bool, False),
    ("ctffind_run", bool, False),
    ("tile_star_created", bool, False),
    ("tile_positions_refined", bool, False),
    ("mts_run", dict, {}),
]


def new_project(name: str, directory: Path):
    # Connect to database
    engine = create_engine(f"sqlite:///{directory / f'{name}.decolace'}", echo=True)
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
    aa_info = pd.DataFrame(
        columns=[
            "name",
            "decolace_acquisition_info_path",
            "frames_folder",
            "cisTEM project",
            "initial_tile_star",
        ]
    )
    aa_info.to_sql(
        "acquisition_areas", sqlite_connection, if_exists="fail", index=False
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

    conn = sqlite3.connect(filename)
    return conn


def update_project(conn: sqlite3.Connection):
    pass
