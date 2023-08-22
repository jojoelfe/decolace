import glob
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import HuberRegressor
from timeit import default_timer as timer
from pydantic import BaseModel
from typing import Optional

from shapely import Polygon, affinity
from contrasttransferfunction import CtfFit


def _hexagonal_cover(polygon, radius):
    """
    Compute hexagonal grid covering the input polygon using spheres of the given radius.

    Args:
        polygon (shapely.geometry.Polygon): Input polygon
        radius (float): Radius of the spheres

    Returns:
        numpy.ndarray: Array of center of the spheres that hexagonally cover the polygon
    """

    # Define a regular hexagon with side length equal to the sphere radius
    hexagon = Polygon(
        [
            (radius * np.cos(angle), radius * np.sin(angle))
            for angle in np.linspace(0, 2 * np.pi, 7)[:-1]
        ]
    )

    # Compute the bounding box of the polygon
    minx, miny, maxx, maxy = polygon.bounds

    # Compute the offset required to center the hexagonal grid within the bounding box
    dx = hexagon.bounds[2] - hexagon.bounds[0]
    dy = hexagon.bounds[3] - hexagon.bounds[1]
    offsetx = (maxx - minx - dx) / 2
    offsety = (maxy - miny - dy) / 2

    # Compute the number of hexagons required in each direction
    nx = int(np.ceil((maxx - minx - dx / 2) / (3 * radius))) + 1
    ny = int(np.ceil((maxy - miny - dy / 2) / (dy * 3 / 2))) + 1

    # Create an empty list to store the center points of the hexagons
    centers = []

    # Loop over each hexagon in the grid and test if it intersects the input polygon
    for j in range(-ny, ny):
        y = miny + offsety + (j * radius * 3 / 2)
        direction = int(((j % 2) - 0.5) * 2)
        for i in range(-nx*direction, nx*direction, direction):
            x = (
                minx
                + offsetx
                + (i * np.sqrt(3) * radius)
                + (j % 2) * 0.5 * np.sqrt(3) * radius
            )
            if polygon.intersects(affinity.translate(hexagon, xoff=x, yoff=y)):
                centers.append((x, y))

    return np.array(centers)

class AcquisitionAreaSingleState(BaseModel):
    desired_defocus: float = -1.0
    cycle_defocus: bool = False
    low_defocus: float = -0.5
    high_defocus: float = -2.0
    defocus_steps: int = 20
    stage_position: Optional[float] = None
    corner_positions_specimen: Optional[np.ndarray] = None
    corner_positions_stage_diff: Optional[np.ndarray] = None
    corner_positions_stage_absolute: Optional[np.ndarray] = None
    corner_positions_image: Optional[np.ndarray] = None
    count_threshold_for_beamshift: float = 1500
    count_threshold_for_ctf: float = 1500

    ctf_cc_threshold: float = 100
    ctf_step_when_unreliable: float = -0.03
    ctf_max_step_initially: float = 2.0
    ctf_max_step: float = 0.5

    acquisition_positions: Optional[np.ndarray] = None
    positions_acquired: Optional[np.ndarray] = None

    beamshift_calibrations_required: int = 25
    beamshift_calibration_measurements: list = []
    beamshift_correction: bool = True

    tilt: float = 0.0

    navigator_map_index: Optional[int]= None
    navigator_center_index: Optional[int] = None

    positions_still_to_fasttrack: int = 0
    fasttrack: bool = False

    class Config:
        arbitrary_types_allowed = True

class AcquisitionAreaSingle:
    state: AcquisitionAreaSingleState = AcquisitionAreaSingleState()

    def __init__(self, name, directory, beam_radius=100, defocus=-1.0, tilt=0.0):
        self.name = name
        self.directory = directory
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.frames_directory = os.path.join(directory, "frames")
        Path(self.frames_directory).mkdir(parents=True, exist_ok=True)
        

    def write_to_disk(self):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{self.name}_{timestr}.npy"
        filename = os.path.join(self.directory, filename)
        np.save(filename, self.state)
        

    def load_from_disk(self):
        potential_files = glob.glob(os.path.join(self.directory, self.name + "_*.npy"))
        if len(potential_files) < 1:
            raise (FileNotFoundError("Couldn't find saved files"))
        most_recent = sorted(potential_files)[-1]
        self.state = np.load(most_recent, allow_pickle=True).item()

     
    def initialize_from_napari(
        self, map_navigator_id: int, center_coordinate, corner_coordinates
    ):
        serialem.LoadOtherMap(map_navigator_id, "A")

        center_item = int(
            serialem.AddImagePosAsNavPoint(
                "A", center_coordinate[0], center_coordinate[1]
            )
        )
        serialem.ChangeItemColor(center_item, 2)
        (index, x, y, z, t) = serialem.ReportOtherItem(center_item)

        self.state.stage_position = np.array([x, y, z])
        self.state.navigator_map_index = map_navigator_id
        self.state.navigator_center_index = center_item

        corner_coordinates_stage = []
        corner_coordinates_stage_diff = []
        corner_coordinates_specimen = []

        for corner in corner_coordinates:
            # Inverted X and Y because they come from numpy
            corner_item = int(serialem.AddImagePosAsNavPoint("A", corner[1], corner[0]))
            serialem.ChangeItemColor(corner_item, 1)

            (ci, cx, cy, cz, ct) = serialem.ReportOtherItem(corner_item)
            corner_coordinates_stage.append((cx, cy))
            dx = cx - x
            dy = cy - y
            corner_coordinates_stage_diff.append((dx, dy))
            serialem.ImageShiftByStageDiff(dx, dy)
            corner_coordinates_specimen.append(serialem.ReportSpecimenShift())
            serialem.SetImageShift(0, 0)
        self.state.corner_positions_specimen = np.array(corner_coordinates_specimen)
        self.state.corner_positions_stage_diff = np.array(
            corner_coordinates_stage_diff
        )
        self.state.corner_positions_stage_absolute = np.array(
            corner_coordinates_stage
        )
        self.state.corner_positions_image = np.array(corner_coordinates)

    def calculate_acquisition_positions_from_napari(self, add_overlap=0.05):

        polygon = Polygon(self.state["corner_positions_specimen"])

        center = _hexagonal_cover(
            polygon, self.state["beam_radius"] * (1 - add_overlap)
        )

        self.state.acquisition_positions = center
        self.state.positions_acquired = np.zeros(
            (self.state.acquisition_positions.shape[0]), dtype="bool"
        )

    def predict_beamshift(self, specimen_coordinates):
        if self.state.beamshift_calibration_measurements is None:
            return None
        data = np.array(self.state.beamshift_calibration_measurements[-100:])
        if len(data) < self.state.beamshift_calibrations_required:
            return None
        reg_x = HuberRegressor().fit(data[:, [0, 1]], data[:, 3])
        reg_y = HuberRegressor().fit(data[:, [0, 1]], data[:, 4])
        return (
            reg_x.predict(np.array(specimen_coordinates).reshape(1, -1)),
            reg_y.predict(np.array(specimen_coordinates).reshape(1, -1)),
        )

    def acquire(
        self,
        established_lock=False,
        initial_defocus=None,
        initial_beamshift=None,
        progress_callback=None,
    ):
        last_bs_correction=0.0
        last_defocus_correction=0.0
        cycle_defocus = False
        if self.state.acquisition_positions is None or self.state.positions_acquired is None:
            raise ValueError("No acquisition positions defined")
        for index in range(len(self.state.acquisition_positions)):
            report = {}
            if self.state.positions_acquired[index]:
                continue
            if index == 0:
                if initial_beamshift is not None and self.state.beamshift_correction:
                    serialem.SetBeamShift(initial_beamshift[0], initial_beamshift[1])
                if initial_defocus is not None:
                    serialem.SetDefocus(initial_defocus)
                
            if index % 10 == 0:
                self.write_to_disk()
            report["position"] = index
            current_speciment_shift = np.array(serialem.ReportSpecimenShift())
            diff = (
                np.array(self.state["acquisition_positions"][index])
                - current_speciment_shift
            )
            serialem.ImageShiftByMicrons(diff[0], diff[1])
            
           
            if self.state.beamshift_correction:
                beam_shift_prediction = self.predict_beamshift(
                    self.state["acquisition_positions"][index]
                )
                if beam_shift_prediction is not None:
                    report["using_beamshift_prediction"] = True
                    serialem.SetBeamShift(
                        beam_shift_prediction[0], beam_shift_prediction[1]
                    )
            if established_lock and index % 5 != 0 and index > 50 and self.state.fasttrack:
                self.state.positions_still_to_fasttrack = 1
                serialem.EarlyReturnNextShot(0)         
            else:
                self.state.positions_still_to_fasttrack = 0
                serialem.ManageDewarsAndPumps(1)

            serialem.Record()
            self.state.positions_acquired[index] = True
            
            if self.state.positions_still_to_fasttrack > 0:
                report["fasttracked"] = True
                report["counts"] = 0
                if progress_callback is not None:
                    progress_callback(report=report, acquisition_area=self)
                continue

            counts = serialem.ReportMeanCounts()
            report["counts"] = counts

            if counts > self.state.count_threshold_for_beamshift and self.state.beamshift_correction:
                report["correcting_beamshift"] = True
                beam_shift_before_centering = np.array(serialem.ReportBeamShift())
                serialem.CenterBeamFromImage(0, 0.4)
                beam_shift_after_centering = np.array(serialem.ReportBeamShift())
                last_bs_correction = np.linalg.norm(
                    beam_shift_before_centering - beam_shift_after_centering
                )
                report["beamshift_correction"] = last_bs_correction
                if last_bs_correction < 0.06:
                    self.state.beamshift_calibration_measurements.append(
                        [
                            self.state.acquisition_positions[index][0],
                            self.state.acquisition_positions[index][1],
                            serialem.ReportDefocus(),
                            beam_shift_after_centering[0],
                            beam_shift_after_centering[1],
                        ]
                    )

            if counts < self.state.count_threshold_for_ctf:
                if progress_callback is not None:
                    progress_callback(report=report, acquisition_area=self)
                continue
            serialem.FFT("A")
            powerspectrum = np.asarray(serialem.bufferImage("AF"))
            fit_result = CtfFit.fit_1d(
                powerspectrum,
                pixel_size_angstrom=4.24,
                voltage_kv=300.0,
                spherical_aberration_mm=2.7,
                amplitude_contrast=0.07)
            measured_defocus = fit_result.ctf.defocus1_angstroms / -10000
            
            report["measured_defocus"] = measured_defocus
            report["ctf_cc"] = fit_result.cross_correlation
            report["ctf_res"] = 0.0
            if (
                fit_result.cross_correlation < self.state.ctf_cc_threshold
                #ctf_results[4] < self.state["ctf_quality_threshold"]
                #or ctf_results[5] > self.state["ctf_res_threshold"]
            ):
                if progress_callback is not None:
                    progress_callback(report=report, acquisition_area=self)
                serialem.ChangeFocus(self.state.ctf_step_when_unreliable)

            # Set desired defocus
            if self.state.cycle_defocus:
                fraction_of_gradient = (np.cos((index % self.state.defocus_steps)/self.state.defocus_steps * 2* np.pi) + 1) /2
                self.state.desired_defocus = self.state.low_defocus + fraction_of_gradient * (self.state.high_defocus-self.state.low_defocus)
            offset = self.state.desired_defocus - measured_defocus
            
            if abs(offset) > self.state.ctf_max_step and established_lock:
                offset = self.state.ctf_max_step * np.sign(offset)
            else:
                if abs(offset) > self.state.ctf_max_step_initially:
                    offset = self.state.ctf_max_step_initially * np.sign(offset)
            if abs(offset) < 0.1 and not established_lock and last_bs_correction < 0.06:
                established_lock = True
            if abs(offset) > 0.001:
                serialem.ChangeFocus(offset)
            report["adjusted_defocus"] = True
            report["defocus_adjusted_by"] = offset

            
            if progress_callback is not None:
                progress_callback(report=report, acquisition_area=self)
           
    def move_to_position(self):
        if self.state.navigator_center_index is not None:
            serialem.RealignToOtherItem(
                int(self.state.navigator_center_index), 0, 0, 0.05, 4, 1
            )
        else:
            if self.state.navigator_map_index is not None:
                serialem.RealignToOtherItem(
                    int(self.state.navigator_map_index), 0, 0, 0.05, 4, 1
                )
            else:
                raise ValueError("No navigator position set")

    def move_to_position_if_needed(self):
        if self.state.navigator_center_index is not None:
            wanted_stage_position = serialem.ReportOtherItem(
                int(self.state.navigator_center_index)
            )
        else:
            wanted_stage_position = serialem.ReportOtherItem(
                int(self.state.navigator_map_index)
            )
        stage_position = serialem.ReportStageXYZ()
        wanted_stage_position = np.array(
            [wanted_stage_position[1], wanted_stage_position[2]]
        )
        stage_position = np.array([stage_position[0], stage_position[1]])
        if np.linalg.norm(wanted_stage_position - stage_position) > 0.001:
            self.move_to_position()
