from pathlib import Path
from typing import List, Union, Optional, Annotated, Any, TypeVar

import pandas as pd
import starfile
from math import isnan

from pydantic import BaseModel
from pydantic import BeforeValidator


def coerce_nan_to_none(x: Any) -> Any:
    if type(x) is str:
        return x
    if isnan(x):
        return None
    return x

T = TypeVar('T')

NoneOrNan = Annotated[Optional[T], BeforeValidator(coerce_nan_to_none)]


class AcquisitionAreaPreProcessing(BaseModel):
    area_name: str
    decolace_acquisition_area_info_path: Union[Path, str] = None
    decolace_grid_info_path: Union[Path, str] = None
    decolace_session_info_path: Union[Path, str] = None
    frames_folder: Union[Path, str] = None
    view_frames_path: NoneOrNan[Union[Path, str]] = None
    view_image_path: NoneOrNan[Union[Path, str]] = None
    cistem_project: NoneOrNan[Union[Path, str]] = None
    unblur_run: bool = False
    ctffind_run: bool = False
    initial_tile_star: NoneOrNan[Union[Path, str]] = None
    refined_tile_star: NoneOrNan[Union[Path, str]] = None
    montage_star: NoneOrNan[Union[Path, str]] = None
    montage_image: NoneOrNan[Union[Path, str]] = None
    experimental_condition: str = "Test"
    notes: str = "Test"

  
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
