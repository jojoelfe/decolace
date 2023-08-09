import glob
import os
import time
from pathlib import Path, PurePath

import numpy as np

try:
    import serialem
except ModuleNotFoundError:
    print("Couldn't import serialem")

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
        #print(f"Loading file {most_recent}")
        self.state = np.load(most_recent, allow_pickle=True).item()
        for grid_info in self.state["grids"]:
            if not Path(grid_info[1]).exists():
                grid_info[1] = Path(self.directory) / Path(grid_info[1]).parts[-1]
            self.grids.append(grid(grid_info[0], grid_info[1]))
            self.grids[-1].load_from_disk()

    def add_grid(self, name, tilt):
        self.grids.append(grid(name, Path(self.directory, name).as_posix()))
        self.grids[-1].state["tilt"] = tilt
        self.grids[-1].write_to_disk()
        self.state["grids"].append([name, Path(self.directory, name).as_posix()])
        self.state["active_grid"] = len(self.state["grids"]) - 1

    @property
    def active_grid(self):
        return self.grids[self.state["active_grid"]]

    def set_active_grid(self, grid_name):
        for i, grid in enumerate(self.grids):
            if grid.name == grid_name:
                self.state["active_grid"] = i
                return
        raise (ValueError("Couldn't find grid with that name"))

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
            settings["IS_to_camera"] = np.array(serialem.ISToCameraMatrix(0))
            self.state["microscope_settings"][mode] = settings
    
    def adjust_beam(self, coverage=0.9):
        from contrasttransferfunction.spectrumhelpers import radial_average
        #Get pixel size
        camera_properties = serialem.CameraProperties()
        pixel_size = camera_properties[4]
        x_dim = camera_properties[0]
        y_dim = camera_properties[1]
        s_dim = min(x_dim,y_dim)
        s_dim_um = s_dim * pixel_size * 0.001
        print(f"Pixel size is {pixel_size} at current magnification. Smallest camera dimension is {s_dim} pixels and {s_dim_um} um.")

        #Set binning to 4 and prevent saving of frames and alignment
        serialem.SetBinning("R",4)
        serialem.SetDoseFracParams("R",0,0,0)


        center_tries = 0
        adjustment = 99
        while adjustment > (1-coverage)*0.05 * s_dim_um:
            beam_shift_before_centering = np.array(serialem.ReportBeamShift())
            serialem.Record()
            serialem.CenterBeamFromImage(0, 1.0)
            beam_shift_after_centering = np.array(serialem.ReportBeamShift())
            adjustment = np.linalg.norm(beam_shift_before_centering-beam_shift_after_centering)
            print(f"Adjusting by {adjustment}")
            center_tries+=1
            if center_tries > 10:
                print(f"Error could not center beam")
                return
        
        #for defocus in range(0,120,10):
        #    serialem.SetDefocus(defocus)

        #    serialem.Record()
        #    serialem.CenterBeamFromImage(0, 1.0)
        #    serialem.Record()
        #    serialem.CenterBeamFromImage(0, 1.0)
        #    serialem.Record()
        #    serialem.Save()
        beam_image = np.asarray(serialem.bufferImage("A"))
        np.save("beam_image_from_sem.npy",beam_image)
        #shortest = min(beam_image.shape)
        #cropped_beam_image = beam_image[]
        #return(radial_average(beam_image))
        

