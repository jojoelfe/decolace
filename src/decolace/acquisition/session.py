import glob
import os
import time
from pathlib import Path, PurePath
from pydantic import BaseModel
from typing import List, Optional
from .serialem_helper import connect_sem
import numpy as np
from rich import print


from .grid import grid

class SessionState(BaseModel):
    grids: List[grid] = []
    microscope_settings: dict = {}
    active_grid: Optional[int] = None
    beam_radius: Optional[float] = None
    fringe_free_focus_vacuum: Optional[float] = None
    min_defocus_for_ffsearch: Optional[float] = 30
    max_defocus_for_ffsearch: Optional[float] = 80
    fringe_free_focus_cross_grating: Optional[float] = None
    dose_rate_e_per_pix_s: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True

class session:
    state: SessionState = SessionState()
    def __init__(self, name, directory):
        self.name = name
        self.directory = directory
        self.grids = []

        

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
        for grid_info in self.state.grids:
            if not Path(grid_info[1]).exists():
                grid_info[1] = Path(self.directory) / Path(grid_info[1]).parts[-1]
            
            try:
                new_grid = grid(grid_info[0], grid_info[1])
                new_grid.load_from_disk()
                self.grids.append(new_grid)
            except FileNotFoundError:
                print(f"Can't load grid {grid_info[0]}")
                

    def add_grid(self, name, tilt):
        self.grids.append(grid(name, Path(self.directory, name).as_posix()))
        self.grids[-1].state["tilt"] = tilt
        self.grids[-1].write_to_disk()
        self.state.grids.append([name, Path(self.directory, name).as_posix()])
        self.state.active_grid = len(self.state["grids"]) - 1

    @property
    def active_grid(self):
        if self.state.active_grid is None:
            return None
        return self.grids[self.state.active_grid]

    def set_active_grid(self, grid_name):
        for i, grid in enumerate(self.grids):
            if grid.name == grid_name:
                self.state.active_grid = i
                return
        raise (ValueError("Couldn't find grid with that name"))

    def add_current_setting(self):
        serialem = connect_sem()
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
            self.microscope_settings[mode] = settings
    
    def prepare_beam_vacuum(self, coverage=0.9):
        serialem = connect_sem()
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
        serialem.SetExposure("R",1)
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
        
        def calculate_fringe_free_score(defocus):
            serialem.SetDefocus(defocus)
            serialem.Record()
            serialem.CenterBeamFromImage(0, 1.0)
            serialem.Record()
            serialem.CenterBeamFromImage(0, 1.0)
            serialem.Record()

            beam_image = np.asarray(serialem.bufferImage("A"))
            # Crop out largest possbile square from the center of slice
            # Get the dimensions of the array
            rows, cols = beam_image.shape

            # Calculate the size of the center square
            size = min(rows, cols)

            # Calculate the starting indices for the slice
            start_row = (rows - size) // 2
            start_col = (cols - size) // 2

            # Slice the array to get the center square
            center = slice[start_row:start_row+size, start_col:start_col+size]
            ra = radial_average(center)
            
            minimal_slope = np.diff(ra).min()
            return minimal_slope

        from scipy.optimize import minimize_scalar

        res = minimize_scalar(calculate_fringe_free_score, bounds=(self.state.min_defocus_for_ffsearch, self.state.max_defocus_for_ffsearch), method='bounded')
        self.state.fringe_free_focus_vacuum = res.x
        serialem.SetDefocus(res.x)
        serialem.Record()
        # Optimize beam size

        beam_diameter = serialem.MeasureBeamDiameter()

        wanted_beam_diameter = coverage * s_dim_um

        for i in range(3):

            current_IA = serialem.ReportIlluminatedArea()
            new_IA = current_IA * wanted_beam_diameter / beam_diameter
            serialem.SetIlluminatedArea(new_IA)
            serialem.Record()
            beam_diameter = serialem.MeasureBeamDiameter()
        serialem.UpdateLowDoseParams("R")
        self.state.beam_radius = beam_diameter / 2
        beam_image = np.asarray(serialem.bufferImage("A"))
        # Get the average value in the center of the image
        center = int(beam_image.shape/2)
        half_width = int(self.state.beam_radius / pixel_size / 2)
        center_value = np.mean(beam_image[center[0]-half_width:center[0]+half_width,center[1]-half_width:center[1]+half_width])
        bin_sqaured = 16 # Should be 16 because bin set to 4
        exposure_time = 1 # Maybe set exposure time to 1 seconds for this methos
        counts_per_electron = serialem.ReportCountScaling # Get from ReportCountScaling
        self.state.dose_rate_e_per_pix_s = center_value / bin_sqaured / exposure_time / counts_per_electron

        print(f"Fringe free defocus over vacuum is {self.state.fringe_free_focus_vacuum} um")
        print(f"Beam radius is {self.state.beam_radius} um")
        print(f"Dose rate is {self.state.dose_rate_e_per_pix_s} e/pix/s")


        

        

