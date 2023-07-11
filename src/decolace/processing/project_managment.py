from pathlib import Path
from typing import List, Union, Optional

import pandas as pd
import starfile
from pydantic import BaseModel, validator


class AcquisitionAreaPreProcessing(BaseModel):
    area_name: str
    decolace_acquisition_area_info_path: Union[Path, str] = None
    decolace_grid_info_path: Union[Path, str] = None
    decolace_session_info_path: Union[Path, str] = None
    frames_folder: Union[Path, str] = None
    view_frames_path: Union[Path, str] = None
    view_image_path: Union[Path, str] = None
    cistem_project: Union[Path, str] = None
    unblur_run: bool = False
    ctffind_run: bool = False
    initial_tile_star: Union[Path, str] = None
    refined_tile_star: Union[Path, str] = None
    montage_star: Optional[Union[Path, str]] = None
    montage_image: Optional[Union[Path, str]] = None
    experimental_condition: str = "Test what"
    notes: str = "Test"

    @validator("*")
    def enforce_none(cls, v):
        if v == "None" or v == "nan":
            return None
        return v

class MatchTemplateRun(BaseModel):
    run_name: str
    run_id: int
    template_path: Union[Path, str]
    template_size: Optional[float] = None
    template_bfm: Optional[float] = None
    threshold_offset: float = 0.0
    angular_step: float = 3.0
    in_plane_angular_step: float = 2.0
    defocus_step: float = 0.0
    defocus_range: float = 0.0



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
    match_template_runs: List[MatchTemplateRun] = []

    def write(self):
        filename = self.project_path / f"{self.project_name}.decolace"
        dict_repr = self.dict()
        # Remove acquisition_areas from dict
        acquisition_areas = dict_repr.pop("acquisition_areas")
        match_template_runs = dict_repr.pop("match_template_runs")
        star_data = {}
        star_data["project_info"] = pd.DataFrame([dict_repr])
        if len(acquisition_areas) > 0:
            star_data["acquisition_areas"] = pd.DataFrame(
                [aa for aa in acquisition_areas]
            )
        if len(match_template_runs) > 0:
            star_data["match_template_runs"] = pd.DataFrame(
                [mtr for mtr in match_template_runs]
            )
        if filename.exists():
            filename.rename(filename.with_suffix(".decolace.bak"))
        starfile.write(star_data, filename, na_rep="None", overwrite=True, )

    @classmethod
    def read(cls, filename: Path):
        star_data = starfile.read(filename, always_dict=True)
        project_info = star_data["project_info"].to_dict("records")[0]
        acquisition_areas = []
        if "acquisition_areas" in star_data:
            acquisition_areas = star_data["acquisition_areas"]
            acquisition_areas = [
                AcquisitionAreaPreProcessing(**aa)
                for aa in acquisition_areas.to_dict("records")
            ]
        match_template_runs = []
        if "match_template_runs" in star_data:
            match_template_runs = star_data["match_template_runs"]
            match_template_runs = [
                MatchTemplateRun(**mtr)
                for mtr in match_template_runs.to_dict("records")
            ]
        return cls(**project_info, acquisition_areas=acquisition_areas, match_template_runs=match_template_runs)
