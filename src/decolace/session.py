import time
import numpy as np


import sys
sys.path.insert(0, 'C:\Program Files\SerialEM\PythonModules')
import serialem

class session:
    def __init__(self,name,directory,load_from_file=False):
        self.state = {}
        self.name = name
        self.directory = directory

        self.state["grids"] = []
        self.state["microscope_settings"] = {}

    def write_to_disk(self):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{self.name}_{timestr}.npy"
        filename = os.path.join(self.directory, filename)
        np.save(filename, self.state)
   
    def load_from_disk(self):
        potential_files = glob.glob(os.path.join(
            self.directory, self.name+"_*.npy"))
        if len(potential_files) < 1:
            raise(FileNotFoundError("Couldn't find saved files"))
        most_recent = sorted(potential_files)[-1]
        print(f"Loading file {most_recent}")
        self.state = np.load(most_recent, allow_pickle=True).item()
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


    def focus_sample_by_stage_movement(self,defocus=-5):
        self.eucentric_z = serialem.ReportStageXYZ()[2]
        serialem.MoveStage(0,0,30)
        repeats = 0
        while repeats < 20:
            repeats += 1 
            serialem.Preview()
            ctf_results = serialem.CtfFind('A',-0.2,-20)
            shift = defocus-ctf_results[0]
            serialem.MoveStage(0,0,shift)