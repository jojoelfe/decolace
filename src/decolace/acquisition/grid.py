import glob
import os
import time
from functools import partial
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
from .serialem_helper import connect_sem

import numpy as np


from .acquisition_area import AcquisitionAreaSingle

class GridState(BaseModel):
    acquisition_areas: List[List] = []
    desired_defocus: float = -1.0
    tilt: float = 0.0
    view_file: Optional[str] = None
    view_frames_directory: Optional[str] = None
    stepwise: bool = False

class grid:
    state: GridState = GridState()
    def __init__(self, name, directory, defocus=-1.0, tilt=0.0):
        self.name = name
        self.directory = directory
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.acquisition_areas = []
        self.state.desired_defocus = defocus
        self.state.tilt = tilt

        self.state.view_file = Path(directory, f"{name}_views.mrc").as_posix()
        self.state.view_frames_directory = Path(directory, "viewframes").as_posix()
        Path(self.state.view_frames_directory).mkdir(parents=True, exist_ok=True)
        self.state.view_frames_directory = Path(
            self.state.view_frames_directory
        ).as_posix()

    def save_navigator(self):
        serialem = connect_sem()
        serialem.SaveNavigator(
            Path(self.directory, f"{self.name}_navigator.nav").as_posix()
        )

    def write_to_disk(self, save_areas=False):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{self.name}_{timestr}.npy"
        filename = os.path.join(self.directory, filename)
        if save_areas:
            for aa in self.acquisition_areas:
                aa.write_to_disk()
        np.save(filename, self.state)

    def load_from_disk(self):
        potential_files = glob.glob(os.path.join(self.directory, self.name + "_*.npy"))
        if len(potential_files) < 1:
            raise (FileNotFoundError("Couldn't find saved files"))
        most_recent = sorted(potential_files)[-1]
        self.state = np.load(most_recent, allow_pickle=True).item()
        #self.state = GridState(**self.state)
        self.acquisition_areas = []
        for area in self.state.acquisition_areas:
            self.acquisition_areas.append(AcquisitionAreaSingle(area[0], Path(self.directory) / area[0]))
            self.acquisition_areas[-1].load_from_disk()

    def ensure_view_file_is_open(self):
        serialem = connect_sem()
        if serialem.ReportFileNumber() < 0:
            if Path(self.state.view_file).exists():
                serialem.OpenOldFile(self.state.view_file)
            else:
                serialem.OpenNewFile(self.state.view_file)
            return
        current_file = serialem.ReportCurrentFilename()

        if Path(current_file).as_posix() != self.state.view_file:
            if Path(self.state.view_file).exists():
                serialem.OpenOldFile(self.state.view_file)
            else:
                serialem.OpenNewFile(self.state.view_file)

    def eucentric(self, stage_height_offset=-64.5, do_euc=True):
        serialem = connect_sem()
        print("Do")
        serialem.Copy("A", "K")  # Copy to buffer K
        if do_euc:
            serialem.Eucentricity(1)  # Rough eucentric
        serialem.View()
        serialem.AlignTo("K")
        serialem.ResetImageShift()
        serialem.TiltTo(self.state.tilt)
        serialem.View()
        serialem.Copy("A", "K")  # Copy to buffer K
        serialem.MoveStage(0, 0, stage_height_offset / 3)
        serialem.View()
        serialem.AlignTo("K")
        serialem.ResetImageShift()
        serialem.MoveStage(0, 0, stage_height_offset / 3)
        serialem.View()
        serialem.AlignTo("K")
        serialem.ResetImageShift()
        serialem.MoveStage(0, 0, stage_height_offset / 3)
        serialem.View()
        serialem.AlignTo("K")
        serialem.ResetImageShift()

    def nice_view(self):
        serialem = connect_sem()
        serialem.GoToLowDoseArea("V")
        serialem.ChangeFocus(-250)
        serialem.SetExposure("V", 10)
        serialem.SetDoseFracParams("V", 1, 1, 0)
        serialem.SetFrameTime("V", 1)
        serialem.SetFolderForFrames(
            os.path.abspath(self.state.view_frames_directory)
        )
        serialem.View()
        serialem.ChangeFocus(250)
        serialem.SetExposure("V", 1)
        serialem.SetDoseFracParams("V", 0)
        self.ensure_view_file_is_open()
        serialem.Save()

    def take_map(self):
        serialem = connect_sem()
        serialem.View()
        self.ensure_view_file_is_open()
        serialem.Save()
        serialem.NewMap(0, "decolace_acquisition_map")
        self.save_navigator()
        serialem.TiltTo(0.0)

    def initialize_acquisition_areas(self, navigator_ids):
        pass

    def start_acquisition(self, initial_defocus, progress_callback=None):
        serialem = connect_sem()
        for index, aa in enumerate(self.acquisition_areas):
            if np.sum(aa.state.positions_acquired) == len(
                aa.state.positions_acquired
            ):
                continue
            serialem.SetImageShift(0.0, 0.0)
            if np.sum(aa.state.positions_acquired) == 0:
                progress_callback(
                    grid=self, acquisition_area=aa, report=None, type="start_new_area"
                )
                aa.move_to_position()
            else:
                progress_callback(
                    grid=self, acquisition_area=aa, report=None, type="resume_area"
                )
                aa.move_to_position_if_needed()
            serialem.LongOperation("Da", "2")
            serialem.SetFolderForFrames(
                os.path.join(os.path.abspath(aa.directory), "frames/")
            )

            serialem.GoToLowDoseArea("R")
            initial_beamshift = None
            if index > 0:
                initial_beamshift = self.acquisition_areas[index-1].predict_beamshift(aa.state.acquisition_positions[0])


            serialem.ManageDewarsAndPumps(-1)
            while serialem.AreDewarsFilling():
                time.sleep(60)

            callback = None
            if progress_callback is not None:
                callback = partial(progress_callback, grid=self)
            aa.acquire(initial_defocus=initial_defocus, progress_callback=callback,initial_beamshift=initial_beamshift)
            aa.write_to_disk()
        serialem.SetColumnOrGunValve(0)

    def draw_acquisition_positions_into_napari(self, viewer, map_navids, beam_radius, use_square_beam=False):
        from skimage.transform import AffineTransform
        write_to_disktext = {
                'string': '{order}',
                'size': 10,
                'color': 'white',
                'translation': np.array([0, 0]),
             }
        order = [[] for a in map_navids]
        positions = [[] for a in map_navids]
        corner_positions = []
        for i, aa in enumerate(self.acquisition_areas):
            
            map_index = map_navids.index(aa.state.navigator_map_index)

            # Get the affine matrix converting aa.state.corner_positions_specimen into aa.state.corner_positions_image
            affine = AffineTransform()
            affine.estimate(aa.state.corner_positions_specimen, aa.state.corner_positions_image)


            pos = affine(aa.state.acquisition_positions[:, ::1])
            # Concatenat i to pos along axis 1
            pos = np.concatenate((np.ones((len(pos), 1)) * map_index, pos), axis=1)
            # Find position of aa.navigator_map_index in nav_ids
            
            positions[map_index].extend(pos)
            order[map_index].extend(np.array(range(len(aa.state.acquisition_positions))))
        pos = np.concatenate(positions, axis = 0)
        order = np.concatenate(order, axis = 0)
        
        if use_square_beam:
            symbol='square'
        else:
            symbol='o'
        viewer.add_points(
            pos,
            name="exposures",
            symbol=symbol,
            size=beam_radius * 2 * affine.scale[0],
            face_color="#00000000",
            features={"order":np.array(order)},
            text=write_to_disktext
        )
        return affine
