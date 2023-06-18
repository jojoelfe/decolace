import glob
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import HuberRegressor
from timeit import default_timer as timer
try:
    import serialem
except ModuleNotFoundError:
    print("Couldn't import serialem")
from shapely import Polygon, affinity


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
        print(direction)
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


class AcquisitionAreaSingle:
    def __init__(self, name, directory, beam_radius=100, defocus=-1.0, tilt=0.0):
        self.state = {}
        self.name = name
        self.directory = directory
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.frames_directory = os.path.join(directory, "frames")
        Path(self.frames_directory).mkdir(parents=True, exist_ok=True)
        self.state["beam_radius"] = beam_radius
        self.state["desired_defocus"] = defocus
        self.state["stage_position"] = None
        self.state["corner_positions_specimen"] = None
        self.state["corner_positions_stage_diff"] = None
        self.state["corner_positions_stage_absolute"] = None
        self.state["corner_positions_image"] = None
        self.state["view_stage_to_specimen"] = None
        self.state["view_specimen_to_camera"] = None
        self.state["record_speciment_to_camera"] = None
        self.state["record_IS_to_camera"] = None

        self.state["count_threshold_for_beamshift"] = 2000
        self.state["count_threshold_for_ctf"] = 2000

        self.state["ctf_quality_threshold"] = 0.10
        self.state["ctf_res_threshold"] = 10.0

        # Array (n,2) of speciment coordinates to be acquired
        self.state["acquisition_positions"] = None

        self.state["ctf_step_when_unreliable"] = -0.03
        self.state["max_ctf_step"] = 0.5

        self.state["beamshift_calibration"] = {}
        self.state["beamshift_calibration"]["model"] = None
        self.state["beamshift_calibration"]["measurements"] = []

        # array (bool,n) indicating of position acquired
        self.state["positions_acquired"] = []

        # {directory: defocus_measures, absolute_defocus, score, resolution}
        self.state["defocus_calibration"] = {}
        self.state["defocus_calibration"]["model"] = None
        self.state["defocus_calibration"]["measurements"] = []
        self.state['use_focus_prediction'] = False
        self.state["tilt"] = tilt

        self.state["navigator_map_index"] = None
        self.state["navigator_center_index"] = None

        self.state["positions_still_to_fasttrack"] = 0
        self.state["currently_fasttracking"] = 0
        self.state["maximum_number_of_fasttracks"] = 10

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

    def plot_acquisition_positions(self):
        top_left = self.state["corner_positions_specimen"][0]
        top_right = self.state["corner_positions_specimen"][1]
        bottom_right = self.state["corner_positions_specimen"][2]
        bottom_left = self.state["corner_positions_specimen"][3]

        msquare = np.array([top_left, top_right, bottom_right, bottom_left])

        line = plt.Polygon(msquare, closed=True, fill=None, edgecolor="r")
        plt.axes()
        plt.gca().add_patch(line)
        for i, ap in enumerate(self.state["acquisition_positions"]):

            color = "b"
            circle = plt.Circle(
                ap, radius=self.state["beam_radius"], fill=None, edgecolor=color
            )
            plt.gca().add_patch(circle)
        plt.axis("scaled")
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.show()

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
        self.state["stage_position"] = np.array([x, y, z])
        self.state["navigator_map_index"] = map_navigator_id
        self.state["navigator_center_index"] = center_item

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
        self.state["corner_positions_specimen"] = np.array(corner_coordinates_specimen)
        self.state["corner_positions_stage_diff"] = np.array(
            corner_coordinates_stage_diff
        )
        self.state["corner_positions_stage_absolute"] = np.array(
            corner_coordinates_stage
        )
        self.state["corner_positions_image"] = np.array(corner_coordinates)

    def calculate_acquisition_positions_from_napari(self, add_overlap=0.05):

        self.state["acquisition_positions"] = []
        self.state["positions_acquired"] = []

        polygon = Polygon(self.state["corner_positions_specimen"])

        center = _hexagonal_cover(
            polygon, self.state["beam_radius"] * (1 - add_overlap)
        )

        self.state["acquisition_positions"] = center
        self.state["positions_acquired"] = np.zeros(
            (self.state["acquisition_positions"].shape[0]), dtype="bool"
        )

    def predict_focus(self, specimen_coordinates):
        data = np.array(self.state["defocus_calibration"]["measurements"])
        if len(data) < 5:
            return None
        reg = HuberRegressor().fit(data[:, [0, 1]], data[:, 2])
        return reg.predict(np.array(specimen_coordinates).reshape(1, -1))

    def predict_beamshift(self, specimen_coordinates):
        data = np.array(self.state["beamshift_calibration"]["measurements"][-100:])
        if len(data) < 25:
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
        serialem.ChangeFocus(-1.0)
        for index in range(len(self.state["acquisition_positions"])):
            start = timer()
            report = {}
            if self.state["positions_acquired"][index]:
                continue
            if index == 0:
                if initial_beamshift is not None:
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
            focus_prediction = None #self.predict_focus(
                #self.state["acquisition_positions"][index]
            #)
            if focus_prediction is not None and self.state['use_focus_prediction']:
                report["using_focus_prediction"] = True
                serialem.SetDefocus(focus_prediction)
            beam_shift_prediction = self.predict_beamshift(
                self.state["acquisition_positions"][index]
            )
            if beam_shift_prediction is not None:
                report["using_beamshift_prediction"] = True
                serialem.SetBeamShift(
                    beam_shift_prediction[0], beam_shift_prediction[1]
                )
            predictions = timer()
            if established_lock and index % 5 != 0 and index > 50:
                self.state["positions_still_to_fasttrack"] = 1
                serialem.EarlyReturnNextShot(0)         
            else:
                self.state["positions_still_to_fasttrack"] = 0
                serialem.ManageDewarsAndPumps(1)

            
            #if self.state["positions_still_to_fasttrack"] > 0:
                #serialem.EarlyReturnNextShot(0)
            serialem.Record()
            self.state["positions_acquired"][index] = True
            #if self.state["positions_still_to_fasttrack"] > 0:
            #    self.state["positions_still_to_fasttrack"] -= 1
            #    report["fasttracked"] = True
            #    report["counts"] = 0
            #    if progress_callback is not None:
            #        progress_callback(report=report, acquisition_area=self)
            #    continue
            record = timer()
            print(f"{predictions-start} {record-predictions}")
            if self.state["positions_still_to_fasttrack"] > 0:
                report["fasttracked"] = True
                report["counts"] = 0
                #serialem.ChangeFocus(-0.01)
                if progress_callback is not None:
                    progress_callback(report=report, acquisition_area=self)
                continue

            counts = serialem.ReportMeanCounts()
            report["counts"] = counts

            if counts > self.state["count_threshold_for_beamshift"]:
                report["correcting_beamshift"] = True
                beam_shift_before_centering = np.array(serialem.ReportBeamShift())
                serialem.CenterBeamFromImage(0, 0.4)
                beam_shift_after_centering = np.array(serialem.ReportBeamShift())
                correction = np.linalg.norm(
                    beam_shift_before_centering - beam_shift_after_centering
                )
                report["beamshift_correction"] = correction
                if correction < 0.06:
                    self.state["beamshift_calibration"]["measurements"].append(
                        [
                            self.state["acquisition_positions"][index][0],
                            self.state["acquisition_positions"][index][1],
                            serialem.ReportDefocus(),
                            beam_shift_after_centering[0],
                            beam_shift_after_centering[1],
                        ]
                    )

            if counts < self.state["count_threshold_for_ctf"]:
                if progress_callback is not None:
                    progress_callback(report=report, acquisition_area=self)
                continue
            serialem.FFT("A")
            #serialem.Save()
            ctf_results = serialem.CtfFind("A", -0.1, -12)
            #ctf_results = []
            #serialem.CropCenterToSize("A",700,700)
            
            #plot_results = serialem.Ctfplotter("A", -0.1,-7,1,0,15.0,1)
            #print(plot_results)
            if len(ctf_results) < 6:
                if progress_callback is not None:
                    progress_callback(report=report, acquisition_area=self)
                continue
            report["measured_defocus"] = ctf_results[0]
            report["ctf_cc"] = ctf_results[4]
            report["ctf_res"] = ctf_results[5]
            if (
                ctf_results[4] < self.state["ctf_quality_threshold"]
                or ctf_results[5] > self.state["ctf_res_threshold"]
            ):
                if progress_callback is not None:
                    progress_callback(report=report, acquisition_area=self)
                serialem.ChangeFocus(-0.05)
                continue

            offset = self.state["desired_defocus"] - ctf_results[0]
            if offset < 0.5:
                self.state["defocus_calibration"]["measurements"].append(
                    [
                        self.state["acquisition_positions"][index][0],
                        self.state["acquisition_positions"][index][1],
                        serialem.ReportDefocus() + offset,
                    ]
                )
                self.state['use_focus_prediction'] = False
            else :
                self.state['use_focus_prediction'] = False
            if abs(offset) > self.state["max_ctf_step"] and established_lock:
                offset = self.state["max_ctf_step"] * np.sign(offset)
            else:
                if abs(offset) > 2.0:
                    offset = 2.0 * np.sign(offset)
            if abs(offset) < 0.1 and not established_lock and correction < 0.06:
                established_lock = True
            if abs(offset) > 0.001:
                serialem.ChangeFocus(offset)
            report["adjusted_defocus"] = True
            report["defocus_adjusted_by"] = offset

            #if self.state['use_focus_prediction'] == True and correction < 0.01 and len(self.state["defocus_calibration"]["measurements"]) > 20:
            #    if self.state["currently_fasttracking"] < self.state["maximum_number_of_fasttracks"]:
            #        self.state["currently_fasttracking"] += 1
            #    self.state["positions_still_to_fasttrack"] = self.state["currently_fasttracking"]
            #else :
            #    self.state["currently_fasttracking"] = 0
        
            if progress_callback is not None:
                progress_callback(report=report, acquisition_area=self)
            others = timer()
            print(f"{predictions-start} {record-predictions} {others-record}")

    def move_to_position(self):
        if self.state["navigator_center_index"] is not None:
            serialem.RealignToOtherItem(
                int(self.state["navigator_center_index"]), 0, 0, 0.05, 4, 1
            )
        else:
            serialem.RealignToOtherItem(
                int(self.state["navigator_map_index"]), 0, 0, 0.05, 4, 1
            )

    def move_to_position_if_needed(self):
        if self.state["navigator_center_index"] is not None:
            wanted_stage_position = serialem.ReportOtherItem(
                int(self.state["navigator_center_index"])
            )
        else:
            wanted_stage_position = serialem.ReportOtherItem(
                int(self.state["navigator_map_index"])
            )
        stage_position = serialem.ReportStageXYZ()
        wanted_stage_position = np.array(
            [wanted_stage_position[1], wanted_stage_position[2]]
        )
        stage_position = np.array([stage_position[0], stage_position[1]])
        if np.linalg.norm(wanted_stage_position - stage_position) > 0.001:
            self.move_to_position()
