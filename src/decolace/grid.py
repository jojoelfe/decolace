from pathlib import Path
import time
import numpy as np


import sys
sys.path.insert(0, 'C:\Program Files\SerialEM\PythonModules')
import serialem

class grid:
    def __init__(self,name,directory, load_from_file=False, session=None):
        self.state = {}
        self.name = name
        self.directory = directory
        self.session = session
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.state["acquisition_areas"] = []
        if load_from_file:
            self.load_from_disk()

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
    
    def create_acquisition_areas(self, navigator_indices,beam_radius=None):
        for navigattor_index in navigator_indices:
            aa_name = f"acquisition_area{len(self.state['acquisition_areas'])}"
            aa = acquisition_area(name=aa_name,directory=f"./{self.name}/{aa_name}/")
            aa.
    
    def initialize_area_template_matching(self):
        for lamella in self.state["acquisition_areas"]:
            lamella.calculate_acquisition_positions(expansion=1.0)
            lamella.designate_calibration_positions()
            lamella.plot_acquisition_positions()
            lamella.write_to_disk()
    
    def collect_areas_template_matching(self):
        for lamella in self.state["acquisition_areas"]:
            serialem.LongOperation("Da","2")
            serialem.SetFolderForFrames(os.path.join(os.path.abspath(lamella.directory),"frames/"))
            serialem.GoToLowDoseArea('V')
            serialem.SetDefocus(-20)
            lamella.move_to_position()
            serialem.GoToLowDoseArea('R')
            serialem.SetDefocus(0)
            serialem.ManageDewarsAndPumps(-1)
            while serialem.AreDewarsFilling():
                time.sleep(60)
            lamella.acquire_single_tilt_slow(tilt=-15.0)
            lamella.write_to_disk()