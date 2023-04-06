from pathlib import Path
import time
import numpy as np
from . import acquisition_area
import sys
sys.path.insert(0, 'C:\Program Files\SerialEM\PythonModules')
import serialem
import glob

class grid:
    def __init__(self,name,directory, beam_radius = 100, defocus = -0.8, tilt=0.0):
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
        self.state["view_frames_directory"] = Path(directory, f"viewframes").as_posix()
        Path(self.state["view_frames_directory"]).mkdir(parents=True, exist_ok=True)
    
    def save_navigator(self):
        serialem.SaveNavigator(Path(self.directory, f"{self.name}_navigator.nav").as_posix())
        

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
        print(f"Loading acquisition areas")
        self.acquisition_areas = []
        for area in self.state["acquisition_areas"]:
            self.acquisition_areas.append(
                acquisition_area.AcquisitionAreaSingle(area["name"], area["directory"])
                )
            self.acquisition_areas[-1].load_from_disk()
        
    def ensure_view_file_is_open(self):

        current_file = serialem.ReportCurrentFilename()

        if Path(current_file).as_posix() != self.state["view_file"]:
            if Path(self.state["view_file"]).exists():
                serialem.OpenOldFile(self.state["view_file"])
            else:
                serialem.OpenNewFile(self.state["view_file"])
    
    def eucentric(self, stage_height_offset=-33.0):
        serialem.Copy('A','K')# Copy to buffer K
        serialem.Eucentricity(1) # Rough eucentric
        serialem.View()
        serialem.AlignTo('K')
        serialem.ResetImageShift()
        serialem.TiltTo(self.state["tilt"])
        serialem.View()
        serialem.Copy('A','K')# Copy to buffer K
        serialem.MoveStage(0,0,stage_height_offset/3)
        serialem.View()
        serialem.AlignTo('K')
        serialem.ResetImageShift()
        serialem.MoveStage(0,0,stage_height_offset/3)
        serialem.View()
        serialem.AlignTo('K')
        serialem.ResetImageShift()
        serialem.MoveStage(0,0,stage_height_offset/3)
        serialem.View()
        serialem.AlignTo('K')
        serialem.ResetImageShift()
    
    def nice_view(self):
        serialem.GoToLowDoseArea('V')
        serialem.ChangeFocus(-200)
        serialem.SetExposure('V', 4)
        serialem.SetDoseFracParams('V', 1, 1, 0)
        serialem.SetFrameTime('V', 1)
        serialem.SetFolderForFrames(self.state["view_frames_directory"])
        serialem.View()
        serialem.ChangeFocus(200)
        self.ensure_view_file_is_open()
        serialem.Save()
        
    
    def take_map(self):
        serialem.View()
        self.ensure_view_file_is_open()
        serialem.Save()
        serialem.NewMap(0,"decolace_acquisition_map")
        self.save_navigator()
        
    