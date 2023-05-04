from pathlib import Path
from typing import List

import pandas as pd
import starfile
from pydantic import BaseModel


class AcquisitionAreaProcessing(BaseModel):
    area_name: str
    decolace_acquisition_info_path: Path = None
    frames_folder: Path = None
    view_frames_path: Path = None
    view_image_path: Path = None
    cistem_project: Path = None
    unblur_run: bool = False
    ctffind_run: bool = False
    initial_tile_star: Path = None
    refined_tile_star: Path = None
    montage_star: Path = None
    montage_image: Path = None
    mts_run: dict = {}


class ProcessingProject(BaseModel):
    project_name: str
    project_path: Path
    processing_pixel_size: float = 2.0
    unblur_arguments: dict = {"align_cropped_area": True, "BB": False}
    ctffind_arguments: dict = {}
    acquisition_areas: List[AcquisitionAreaProcessing] = []

    def write(self):

        filename = self.project_path / f"{self.project_name}.decolace"
        dict_repr = self.dict()
        # Remove acquisition_areas from dict
        acquisition_areas = dict_repr.pop("acquisition_areas")
        star_data = {}
        star_data["project_info"] = pd.DataFrame([dict_repr])
        if len(acquisition_areas) > 0:
            star_data["acquisition_areas"] = pd.DataFrame(
                [aa for aa in acquisition_areas]
            )
        starfile.write(star_data, filename)

    @classmethod
    def read(cls, filename: Path):
        star_data = starfile.read(filename)
        project_info = star_data["project_info"]
        acquisition_areas = star_data["acquisition_areas"]
        acquisition_areas = [
            AcquisitionAreaProcessing(**aa)
            for aa in acquisition_areas.to_dict("records")
        ]
        return cls(**project_info, acquisition_areas=acquisition_areas)
