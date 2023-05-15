from pathlib import Path
from typing import List, Union

import pandas as pd
import starfile
from pydantic import BaseModel, validator


class AcquisitionAreaPreProcessing(BaseModel):
    area_name: str
    decolace_acquisition_info_path: Union[Path, str] = None
    frames_folder: Union[Path, str] = None
    view_frames_path: Union[Path, str] = None
    view_image_path: Union[Path, str] = None
    cistem_project: Union[Path, str] = None
    unblur_run: bool = False
    ctffind_run: bool = False
    initial_tile_star: Union[Path, str] = None
    refined_tile_star: Union[Path, str] = None
    montage_star: Union[Path, str] = None
    montage_image: Union[Path, str] = None

    @validator("*")
    def enforce_none(cls, v):
        if v == "None" or v == "nan":
            return None
        return v


class ProcessingProject(BaseModel):
    project_name: str
    project_path: Path
    processing_pixel_size: float = 2.0
    acquisition_areas: List[AcquisitionAreaPreProcessing] = []

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
        starfile.write(star_data, filename, na_rep="None", overwrite=True)

    @classmethod
    def read(cls, filename: Path):
        star_data = starfile.read(filename)
        project_info = star_data["project_info"].to_dict("records")[0]
        acquisition_areas = star_data["acquisition_areas"]
        acquisition_areas = [
            AcquisitionAreaPreProcessing(**aa)
            for aa in acquisition_areas.to_dict("records")
        ]
        return cls(**project_info, acquisition_areas=acquisition_areas)
