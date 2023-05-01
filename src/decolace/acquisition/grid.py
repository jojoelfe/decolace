import glob
import os
import time
from functools import partial
from pathlib import Path

import numpy as np

try:
    import serialem
except ModuleNotFoundError:
    print("Couldn't import serialem")

from .acquisition_area import AcquisitionAreaSingle


class grid:
    def __init__(self, name, directory, beam_radius=100, defocus=-0.8, tilt=0.0):
        self.state = {}
        self.name = name
        self.directory = directory
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.state["acquisition_areas"] = []
        self.acquisition_areas = []
        self.state["beam_radius"] = beam_radius
        self.state["desired_defocus"] = defocus
        self.state["tilt"] = tilt

        self.state["view_file"] = Path(directory, f"{name}_views.mrc").as_posix()
        self.state["view_frames_directory"] = Path(directory, "viewframes").as_posix()
        Path(self.state["view_frames_directory"]).mkdir(parents=True, exist_ok=True)
        self.state["view_frames_directory"] = Path(
            self.state["view_frames_directory"]
        ).as_posix()

    def save_navigator(self):
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
        print(f"Loading file {most_recent}")
        self.state = np.load(most_recent, allow_pickle=True).item()
        print("Loading acquisition areas")
        self.acquisition_areas = []
        for area in self.state["acquisition_areas"]:
            self.acquisition_areas.append(AcquisitionAreaSingle(area[0], area[1]))
            self.acquisition_areas[-1].load_from_disk()

    def ensure_view_file_is_open(self):

        if serialem.ReportFileNumber() < 0:
            if Path(self.state["view_file"]).exists():
                serialem.OpenOldFile(self.state["view_file"])
            else:
                serialem.OpenNewFile(self.state["view_file"])
            return
        current_file = serialem.ReportCurrentFilename()

        if Path(current_file).as_posix() != self.state["view_file"]:
            if Path(self.state["view_file"]).exists():
                serialem.OpenOldFile(self.state["view_file"])
            else:
                serialem.OpenNewFile(self.state["view_file"])

    def eucentric(self, stage_height_offset=-33.0, do_euc=True):
        print("Do")
        serialem.Copy("A", "K")  # Copy to buffer K
        if do_euc:
            serialem.Eucentricity(1)  # Rough eucentric
        serialem.View()
        serialem.AlignTo("K")
        serialem.ResetImageShift()
        serialem.TiltTo(self.state["tilt"])
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
        serialem.GoToLowDoseArea("V")
        serialem.ChangeFocus(-200)
        serialem.SetExposure("V", 4)
        serialem.SetDoseFracParams("V", 1, 1, 0)
        serialem.SetFrameTime("V", 1)
        serialem.SetFolderForFrames(
            os.path.abspath(self.state["view_frames_directory"])
        )
        serialem.View()
        serialem.ChangeFocus(200)
        serialem.SetExposure("V", 1)
        serialem.SetDoseFracParams("V", 0)
        self.ensure_view_file_is_open()
        serialem.Save()

    def take_map(self):
        serialem.View()
        self.ensure_view_file_is_open()
        serialem.Save()
        serialem.NewMap(0, "decolace_acquisition_map")
        self.save_navigator()

    def initialize_acquisition_areas(self, navigator_ids):
        pass

    def start_acquisition(self, initial_defocus=24.0, progress_callback=None):
        for aa in self.acquisition_areas:
            if np.sum(aa.state["positions_acquired"]) == len(
                aa.state["positions_acquired"]
            ):
                continue
            serialem.SetImageShift(0.0, 0.0)
            if np.sum(aa.state["positions_acquired"]) == 0:
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

            serialem.ManageDewarsAndPumps(-1)
            while serialem.AreDewarsFilling():
                time.sleep(60)

            callback = None
            if progress_callback is not None:
                callback = partial(progress_callback, grid=self)
            aa.acquire(initial_defocus=initial_defocus, progress_callback=callback)
            aa.write_to_disk()
        serialem.SetColumnOrGunValve(0)
