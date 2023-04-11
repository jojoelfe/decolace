import time
import numpy as np
from .grid import grid
from pathlib import Path
import os
import glob

import sys
sys.path.insert(0, 'C:\Program Files\SerialEM\PythonModules')
import serialem

class session:
    def __init__(self,name,directory):
        self.state = {}
        self.name = name
        self.directory = directory
        self.grids = []

        self.state["grids"] = []
        self.state["microscope_settings"] = {}

    def write_to_disk(self):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{self.name}_{timestr}.npy"
        filename = os.path.join(self.directory, filename)
        np.save(filename, self.state)
        for grid in self.grids:
            grid.write_to_disk()
   
    def load_from_disk(self):
        potential_files = glob.glob(os.path.join(
            self.directory, self.name+"_*.npy"))
        if len(potential_files) < 1:
            raise(FileNotFoundError("Couldn't find saved files"))
        most_recent = sorted(potential_files)[-1]
        print(f"Loading file {most_recent}")
        self.state = np.load(most_recent, allow_pickle=True).item()
        for grid_info in self.state["grids"]:
            self.grids.append(grid(grid_info[0], grid_info[1]))
            self.grids[-1].load_from_disk()
    
    def add_grid(self, name):
        self.grids.append(grid(name, Path(self.directory,name).as_posix()))
        self.state["grids"].append([name, Path(self.directory,name).as_posix()])



    def add_current_setting(self, name, fringe_free=False):

        settings = {}
        settings["magnification"] = serialem.ReportMag()
        settings["magnification_index"] = serialem.ReportMagIndex()
        settings["spot_size"] = serialem.ReportSpotSize()
        settings["illuminated_area"] = serialem.ReportIlluminatedArea()
        settings["beam_tilt"] = serialem.ReportBeamTilt()
        settings["objective_stigmator"] = serialem.ReportObjectiveStigmator()
        settings["fringe_free"] = fringe_free

        if fringe_free:
            settings["fringe_free_nominal_defocus_c2aperture"] = serialem.ReportDefocus()
            settings["fringe_free_stage_z_diff"] = serialem.ReportStageXYZ()[2] - self.eucentric_z

        self.state["microscope_settings"][name] = settings


