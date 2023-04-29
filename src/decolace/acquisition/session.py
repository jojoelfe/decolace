import glob
import os
import time
from pathlib import Path

import numpy as np
import serialem

from .grid import grid


class session:
    def __init__(self, name, directory):
        self.state = {}
        self.name = name
        self.directory = directory
        self.grids = []

        self.state["grids"] = []
        self.state["microscope_settings"] = {}
        self.state["active_grid"] = None

    def write_to_disk(self, save_grids=False):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{self.name}_{timestr}.npy"
        filename = os.path.join(self.directory, filename)
        np.save(filename, self.state)
        if save_grids:
            for grid_o in self.grids:
                grid_o.write_to_disk()

    def load_from_disk(self):
        potential_files = glob.glob(os.path.join(self.directory, self.name + "_*.npy"))
        if len(potential_files) < 1:
            raise (FileNotFoundError("Couldn't find saved files"))
        most_recent = sorted(potential_files)[-1]
        print(f"Loading file {most_recent}")
        self.state = np.load(most_recent, allow_pickle=True).item()
        for grid_info in self.state["grids"]:
            self.grids.append(grid(grid_info[0], grid_info[1]))
            self.grids[-1].load_from_disk()

    def add_grid(self, name):
        self.grids.append(grid(name, Path(self.directory, name).as_posix()))
        self.grids[-1].write_to_disk()
        self.state["grids"].append([name, Path(self.directory, name).as_posix()])
        self.state["active_grid"] = len(self.state["grids"]) - 1

    @property
    def active_grid(self):
        return self.grids[self.state["active_grid"]]

    def add_current_setting(self):

        modes = ["V", "R"]
        for mode in modes:
            settings = {}
            serialem.GoToLowDoseArea(mode)
            settings["magnification"] = serialem.ReportMag()
            settings["magnification_index"] = serialem.ReportMagIndex()
            settings["spot_size"] = serialem.ReportSpotSize()
            settings["illuminated_area"] = serialem.ReportIlluminatedArea()
            settings["beam_tilt"] = serialem.ReportBeamTilt()
            settings["objective_stigmator"] = serialem.ReportObjectiveStigmator()
            settings["stage_to_specimen"] = np.array(serialem.StageToSpecimenMatrix(0))
            settings["specimen_to_camera"] = np.array(
                serialem.SpecimenToCameraMatrix(0)
            )
            self.state["IS_to_camera"] = np.array(serialem.ISToCameraMatrix(0))
            self.state["microscope_settings"][mode] = settings
