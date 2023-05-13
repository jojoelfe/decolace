from pathlib import Path
from typing import List, Optional

import pandas as pd
import starfile
from pydantic import BaseModel, validator


class AcquisitionAreaPreProcessing(BaseModel):
    area_name: str
    decolace_acquisition_info_path: Optional[Path] = None
    frames_folder: Optional[Path] = None
    view_frames_path: Optional[Path] = None
    view_image_path: Optional[Path] = None
    cistem_project: Optional[Path] = None
    unblur_run: bool = False
    ctffind_run: bool = False
    initial_tile_star: Optional[Path] = None
    refined_tile_star: Optional[Path] = None
    montage_star: Optional[Path] = None
    montage_image: Optional[Path] = None

    @validator('*')
    def change_nan_to_none(cls, v, field):
        print(field.outer_type_)
        if field.outer_type_ is Optional[Path] and isnan(v):
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
        starfile.write(star_data, filename)

    @classmethod
    def read(cls, filename: Path):
        star_data = starfile.read(filename)
        project_info = star_data["project_info"]
        acquisition_areas = star_data["acquisition_areas"]
        print(acquisition_areas.to_dict("records"))
        acquisition_areas = [
            AcquisitionAreaPreProcessing(**aa)
            for aa in acquisition_areas.to_dict("records")
        ]
        return cls(**project_info, acquisition_areas=acquisition_areas)
