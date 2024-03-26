import glob
import os
import time
from pathlib import Path, PurePath
from pydantic.v1 import BaseModel
from typing import List, Optional
from .serialem_helper import connect_sem
import numpy as np
from rich import print
from enum import Enum

from .grid import grid

class ProblemPolicy(Enum):
    CONTINUE = 'continue'
    PAUSE = 'pause'
    ABORT = 'abort'
    NEXT = 'next'
class SessionState(BaseModel):
    grids: List[List] = []
    microscope_settings: dict = {}
    active_grid: Optional[int] = None
    beam_radius: Optional[float] = None
    fringe_free_focus_vacuum: Optional[float] = None
    min_defocus_for_ffsearch: Optional[float] = -10
    max_defocus_for_ffsearch: Optional[float] = 10
    fringe_free_focus_cross_grating: Optional[float] = None
    dose_rate_e_per_pix_s: Optional[float] = None
    unbinned_pixel_size_A: Optional[float] = None
    cross_euc_Z_height: Optional[float] = None
    cross_ff_Z_height: Optional[float] = None
    euc_to_ff_offset: Optional[float] = None
    problem_policy: ProblemPolicy = 'next'

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
        #self.state = SessionState(**self.state)
        for grid_info in self.state.grids:
            if not Path(grid_info[1]).exists():
                grid_info[1] = Path(self.directory) / Path(grid_info[1]).parts[-1]
            
            try:
                new_grid = grid(grid_info[0], grid_info[1])
                new_grid.load_from_disk()
                self.grids.append(new_grid)
            except FileNotFoundError as e:
                print(f"Can't load grid {grid_info[0]} {e}")
                

    def add_grid(self, name, tilt):
        self.grids.append(grid(name, Path(self.directory, name).as_posix()))
        self.grids[-1].state.tilt = tilt
        self.grids[-1].write_to_disk()
        self.state.grids.append([name, Path(self.directory, name).as_posix()])
        self.state.active_grid = len(self.state.grids) - 1

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
            self.state.microscope_settings[mode] = settings
    

    def prepare_beam_cross_euc(self):
        serialem = connect_sem()
        serialem.Eucentricity(3)
        self.state.cross_euc_Z_height = serialem.ReportStageXYZ()[2]
        serialem.SetBinning("R",4)
        serialem.SetExposure("R",1)
        serialem.SetDoseFracParams("R",0,0,0)

        center_tries = 0
        adjustment = 99
        while adjustment > 0.01 * self.state.beam_radius:
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
            
        print("Please adjust the focus until there are no beam fringes and then run prepare-beam-cross-final")
    
    def prepare_beam_cross_final(self):
        from contrasttransferfunction import CtfFit

        serialem = connect_sem()
        self.state.fringe_free_focus_cross_grating = serialem.ReportDefocus()
        serialem.SetBinning("R",4)
        serialem.SetExposure("R",1)
        serialem.SetDoseFracParams("R",0,0,0)

        center_tries = 0
        adjustment = 99
        while adjustment > 0.01 * self.state.beam_radius:
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

        #serialem.MoveStage(0,0, 10.0)
        wanted_defocus = -1.0
        # Get Focus using record mode wihtout adjusting focus    
        #measured_defocus = serialem.G(-1,-1)
        #defocus_error = wanted_defocus - measured_defocus
        #serialem.MoveStage(0,0,defocus_error)

        serialem.Record()
        serialem.FFT("A")
        powerspectrum = np.asarray(serialem.bufferImage("AF"))
        fit_result = CtfFit.fit_1d(
            powerspectrum,
            pixel_size_angstrom=self.state.unbinned_pixel_size_A * 4,
            voltage_kv=300.0,
            spherical_aberration_mm=2.7,
            amplitude_contrast=0.07)
        measured_defocus = fit_result.ctf.defocus1_angstroms / -10000
        print(measured_defocus)
        defocus_error = wanted_defocus - measured_defocus
        num_tries=0
        while abs(defocus_error) > 0.1 and num_tries < 10:
            num_tries += 1
            serialem.MoveStage(0,0,defocus_error)
            serialem.Record()
            serialem.FFT("A")
            powerspectrum = np.asarray(serialem.bufferImage("AF"))
            fit_result = CtfFit.fit_1d(
                powerspectrum,
                pixel_size_angstrom=self.state.unbinned_pixel_size_A * 4,
                voltage_kv=300.0,
                spherical_aberration_mm=2.7,
                amplitude_contrast=0.07)
            measured_defocus = fit_result.ctf.defocus1_angstroms / -10000
            print(measured_defocus)
            defocus_error = wanted_defocus - measured_defocus
        
        self.state.cross_ff_Z_height = serialem.ReportStageXYZ()[2]
        self.state.euc_to_ff_offset =  self.state.cross_ff_Z_height - self.state.cross_euc_Z_height

        print(f"Measured stage Z offset between eucentric and fring-free position is {self.state.euc_to_ff_offset} um")




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
        self.state.unbinned_pixel_size_A = pixel_size * 10
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
            center = beam_image[start_row:start_row+size, start_col:start_col+size]
            ra = radial_average(center)
            
            minimal_slope = np.diff(ra).min()
            return minimal_slope

        from scipy.optimize import minimize_scalar

        res = minimize_scalar(calculate_fringe_free_score, 
                              bounds=(self.state.min_defocus_for_ffsearch, self.state.max_defocus_for_ffsearch), 
                              method='bounded',
                              options={"maxiter":10,"disp":True})
        res = minimize_scalar(calculate_fringe_free_score, 
                              bounds=(res.x-10.0, res.x+10.0), 
                              method='bounded',
                              options={"maxiter":10,"disp":True})
        self.state.fringe_free_focus_vacuum = res.x
        serialem.SetDefocus(res.x)
        serialem.Record()
        # Optimize beam size

        beam_diameter = serialem.MeasureBeamSize()
        wanted_beam_diameter = coverage * s_dim_um
        print(f"dia {beam_diameter} wanted {wanted_beam_diameter}")

        for i in range(3):

            current_IA = serialem.ReportIlluminatedArea()
            print(f"current_IA {current_IA}")
            new_IA = current_IA * wanted_beam_diameter / beam_diameter[0]
            print(f"new_IA {new_IA}")
            serialem.SetIlluminatedArea(new_IA)
            serialem.UpdateLowDoseParams("R",1)
            serialem.Record()
            beam_diameter = serialem.MeasureBeamSize()
            print(f"dia {beam_diameter} wanted {wanted_beam_diameter}")

        self.state.beam_radius = beam_diameter[0] / 2
        beam_image = np.asarray(serialem.bufferImage("A"))
        # Get the average value in the center of the image
        centery = int(beam_image.shape[0]/2)
        centerx = int(beam_image.shape[1]/2)
        print(beam_image.shape)
        half_width = int((self.state.beam_radius * 1000) / (pixel_size * 8) )
        print(half_width)
        center_image = beam_image[centery-half_width:centery+half_width,centerx-half_width:centerx+half_width]
        print(center_image.shape)
        center_value = np.mean(center_image)
        print(f"Mean {center_value}")
        bin_sqaured = 16 # Should be 16 because bin set to 4
        exposure_time = 1 # Maybe set exposure time to 1 seconds for this methos
        counts_per_electron = serialem.ReportCountScaling()[1] # Get from ReportCountScaling
        self.state.dose_rate_e_per_pix_s = center_value / bin_sqaured / exposure_time / counts_per_electron

        print(f"Fringe free defocus over vacuum is {self.state.fringe_free_focus_vacuum} um")
        print(f"Beam radius is {self.state.beam_radius} um")
        print(f"Dose rate is {self.state.dose_rate_e_per_pix_s} e/pix/s")


        

        

