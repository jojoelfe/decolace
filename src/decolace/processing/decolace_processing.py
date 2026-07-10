import contextlib
import json
import multiprocessing
import sqlite3
from functools import partial
from pathlib import Path
from typing import Optional

import mrcfile
import numpy as np
import pandas as pd
import starfile
from rich.progress import track
from scipy import optimize
from scipy.ndimage import binary_erosion, mean
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from skimage import filters, transform
from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked
from skimage.transform import resize


def read_data_from_cistem(database_filename: Path) -> pd.DataFrame:
    with contextlib.closing(sqlite3.connect(database_filename)) as con:

        df1 = pd.read_sql_query(
            "SELECT IMAGE_ASSET_ID,"
            " MOVIE_ASSET_ID,"
            " IMAGE_ASSETS.FILENAME,"
            " MOVIE_ASSETS.FILENAME as movie_filename,"
            " CTF_ESTIMATION_ID,"
            " ALIGNMENT_ID,"
            " IMAGE_ASSETS.PIXEL_SIZE as image_pixel_size,"
            " MOVIE_ASSETS.PIXEL_SIZE as movie_pixel_size,"
            " IMAGE_ASSETS.X_SIZE,"
            " IMAGE_ASSETS.Y_SIZE"
            " FROM IMAGE_ASSETS"
            " INNER JOIN MOVIE_ASSETS ON MOVIE_ASSETS.MOVIE_ASSET_ID == IMAGE_ASSETS.PARENT_MOVIE_ID",
            con,
        )
        df2 = pd.read_sql_query(
            "SELECT CTF_ESTIMATION_ID,DEFOCUS1,DEFOCUS2,DEFOCUS_ANGLE,OUTPUT_DIAGNOSTIC_FILE,SCORE, DETECTED_RING_RESOLUTION FROM ESTIMATED_CTF_PARAMETERS",
            con,
        )
        selected_micrographs = pd.merge(df1, df2, on="CTF_ESTIMATION_ID")
        df3 = pd.read_sql_query(
            "SELECT MOVIE_ASSET_ID,IMAGE_SHIFT_X, IMAGE_SHIFT_Y, CONTENT_JSON, ACQUISITION_TIME FROM MOVIE_ASSETS_METADATA",
            con,
        )
        selected_micrographs = pd.merge(selected_micrographs, df3, on="MOVIE_ASSET_ID")
        df4 = pd.read_sql_query(
            "SELECT ALIGNMENT_ID, ORIGINAL_X_SIZE, ORIGINAL_Y_SIZE, CROP_CENTER_X, CROP_CENTER_Y FROM MOVIE_ALIGNMENT_LIST",
            con,
        )
        selected_micrographs = pd.merge(selected_micrographs, df4, on="ALIGNMENT_ID")
    return selected_micrographs


def read_matches_data(database_filename: Path, tm_job_id: int) -> pd.DataFrame:
    with contextlib.closing(sqlite3.connect(database_filename)) as con:
        matches_data = pd.read_sql_query(
            f"SELECT IMAGE_ASSETS.FILENAME AS IMAGE_FILENAME,"
            f" PROJECTION_RESULT_OUTPUT_FILE, SCALED_MIP_OUTPUT_FILE "
            f"FROM TEMPLATE_MATCH_LIST INNER JOIN IMAGE_ASSETS ON TEMPLATE_MATCH_LIST.IMAGE_ASSET_ID = IMAGE_ASSETS.IMAGE_ASSET_ID "
            f"WHERE TEMPLATE_MATCH_JOB_ID={tm_job_id}",
            con,
        )
    return matches_data


def read_decolace_data(decolace_filename: Path) -> dict:

    return np.load(decolace_filename, allow_pickle=True).item()

def prune_tiles(tile_metadata, max_mean_density: Optional[float] = None) -> tuple[pd.DataFrame, str]:
    if max_mean_density is None:
        return tile_metadata, ""
    good_tiles = []
    initial_len = len(tile_metadata)
    for i, tile in tile_metadata.iterrows():
        with mrcfile.open(tile["tile_filename"]) as mrc:
            mean_density = mrc.header["dmean"]
        if mean_density < max_mean_density:
            good_tiles.append(i)
    tile_metadata = tile_metadata.loc[good_tiles]
    message = f"Pruned {initial_len-len(good_tiles)} tiles with a mean density higher than {max_mean_density}.\n"
    return tile_metadata.copy(), message

def prune_bad_shifts(shifts: pd.DataFrame, inital_area_cutoff_as_fraction_of_median:float = 0.2, cc_cutoff_as_fraction_of_media:float=0.5):

    init_len = len(shifts)
    shifts = shifts[shifts["initial_area"] > inital_area_cutoff_as_fraction_of_median * shifts["initial_area"].median()]
    message = f"Pruned {init_len-len(shifts)} shifts with an initial area smaller than {inital_area_cutoff_as_fraction_of_median} times the median initial area of {shifts['initial_area'].median()}\n"

    init_len = len(shifts)
    shifts = shifts[shifts["max_cc"] > cc_cutoff_as_fraction_of_media * shifts["max_cc"].median()]
    message += f"Pruned {init_len-len(shifts)} shifts with a max cc smaller than {cc_cutoff_as_fraction_of_media} times the median max cc of {shifts['max_cc'].median()}\n"

    return shifts, message

def create_tile_metadata(
    cistem_data: pd.DataFrame, decolace_data: dict, output_filename: Path
):
    try:
        IS_to_camera = decolace_data["microscope_settings"]["R"]["IS_to_camera"].reshape(2, 2)
    except KeyError:
        IS_to_camera = decolace_data['record_IS_to_camera'].reshape(2, 2)

    result_tiles = pd.DataFrame(
        {
            "tile_filename": pd.Series(dtype="object"),
            "tile_movie_filename": pd.Series(dtype="object"),
            "tile_pixel_size": pd.Series(dtype="float"),
            "tile_x_size": pd.Series(dtype="int"),
            "tile_y_size": pd.Series(dtype="int"),
            "tile_orig_x_size": pd.Series(dtype="int"),
            "tile_orig_y_size": pd.Series(dtype="int"),
            "tile_x_crop_center": pd.Series(dtype="int"),
            "tile_y_crop_center": pd.Series(dtype="int"),
            "tile_mask_filename": pd.Series(dtype="object"),
            "tile_plotted_result_filename": pd.Series(dtype="object"),
            "tile_image_shift_x": pd.Series(dtype="float"),
            "tile_image_shift_y": pd.Series(dtype="float"),
            "tile_image_shift_pixel_x": pd.Series(dtype="int"),
            "tile_image_shift_pixel_y": pd.Series(dtype="int"),
            "tile_microscope_focus": pd.Series(dtype="float"),
            "tile_defocus": pd.Series(dtype="float"),
            "tile_acquisition_time": pd.Series(dtype="int"),
        }
    )

    for i, item in cistem_data.iterrows():

        # Initially convert image shift to unbinned pixels
        image_shift = np.array([item["IMAGE_SHIFT_X"], item["IMAGE_SHIFT_Y"]])
        shift_in_camera_pixels = np.dot(IS_to_camera, image_shift)
        shift_in_image_pixel = shift_in_camera_pixels / (
            item["image_pixel_size"] / (item["movie_pixel_size"])
        )
        crop_x = (item["ORIGINAL_X_SIZE"] - item["X_SIZE"]) / 2
        crop_y = (item["ORIGINAL_Y_SIZE"] - item["Y_SIZE"]) / 2
        crop_center_x = item["CROP_CENTER_X"]
        crop_center_y = item["CROP_CENTER_Y"]
        cistem_data.loc[i, "image_shift_pixel_x"] = (
            shift_in_image_pixel[0] + crop_x + crop_center_x
        )
        cistem_data.loc[i, "image_shift_pixel_y"] = (
            -shift_in_image_pixel[1] + crop_y + crop_center_y
        )

    # Now iterate again, to setup the required data
    for i, item in cistem_data.iterrows():
        new_entry = {}
        new_entry["tile_filename"] = item["FILENAME"]
        new_entry["tile_movie_filename"] = item["movie_filename"]
        new_entry["tile_mask_filename"] = item["FILENAME"][:-4] + "_mask.mrc"
        new_entry["tile_pixel_size"] = item["image_pixel_size"]
        new_entry["tile_x_size"] = item["X_SIZE"]
        new_entry["tile_y_size"] = item["Y_SIZE"]
        new_entry["tile_orig_x_size"] = item["ORIGINAL_X_SIZE"]
        new_entry["tile_orig_y_size"] = item["ORIGINAL_Y_SIZE"]
        new_entry["tile_x_crop_center"] = item["CROP_CENTER_X"]
        new_entry["tile_y_crop_center"] = item["CROP_CENTER_Y"]
        new_entry["tile_image_shift_x"] = item["IMAGE_SHIFT_X"]
        new_entry["tile_image_shift_y"] = item["IMAGE_SHIFT_Y"]
        new_entry["tile_image_shift_pixel_x"] = item["image_shift_pixel_x"]
        new_entry["tile_image_shift_pixel_y"] = item["image_shift_pixel_y"]
        new_entry["tile_microscope_focus"] = json.loads(item["CONTENT_JSON"])["Defocus"]
        new_entry["tile_defocus"] = (item["DEFOCUS1"] + item["DEFOCUS2"]) / 2
        new_entry["tile_acquisition_time"] = item["ACQUISITION_TIME"]
        matches_dir = Path(item["FILENAME"]).parent.parent / "TemplateMatching"
        matches_filenames = list(
            matches_dir.glob(Path(item["FILENAME"][:-6]).name + "*plotted_result*.mrc")
        )
        if len(matches_filenames) > 0:
            new_entry["tile_plotted_result_filename"] = str(matches_filenames[0])
        else:
            new_entry["tile_plotted_result_filename"] = "None"

        result_tiles.loc[len(result_tiles.index)] = new_entry

    results = {"tiles": result_tiles}
    starfile.write(results, output_filename, overwrite=True)

    return results


def create_montage_metadata(
    tile_data: pd.DataFrame,
    output_path_metadata: Path,
    binning: float,
    output_path_montage: Path,
):
    # Create the montage metadata
    montage_metadata = pd.DataFrame(
        {
            "montage_filename": pd.Series(dtype="string"),
            "matches_montage_filename": pd.Series(dtype="string"),
            "montage_pixel_size": pd.Series(dtype="float"),
            "montage_binning": pd.Series(dtype="float"),
            "montage_x_size": pd.Series(dtype="int"),
            "montage_y_size": pd.Series(dtype="int"),
        }
    )
    unbinned_size_x = (
        tile_data["tile_image_shift_pixel_x"].max()
        + tile_data["tile_x_size"].max()
        - tile_data["tile_image_shift_pixel_x"].min() + binning*5
    )
    unbinned_size_y = (
        tile_data["tile_image_shift_pixel_y"].max()
        + tile_data["tile_y_size"].max()
        - tile_data["tile_image_shift_pixel_y"].min() +  binning*5
    )

    x_offset = tile_data["tile_image_shift_pixel_x"].min()
    y_offset = tile_data["tile_image_shift_pixel_y"].min()

    binned_size_x = unbinned_size_x // binning
    binned_size_y = unbinned_size_y // binning
    pixel_size = tile_data["tile_pixel_size"].values[0] * binning

    montage_metadata.loc[0, "montage_filename"] = str(output_path_montage)
    montage_metadata.loc[0, "matches_montage_filename"] = "montage_matches.mrc"
    montage_metadata.loc[0, "montage_pixel_size"] = pixel_size
    montage_metadata.loc[0, "montage_binning"] = binning
    montage_metadata.loc[0, "montage_x_size"] = binned_size_x
    montage_metadata.loc[0, "montage_y_size"] = binned_size_y

    for i, item in tile_data.iterrows():
        tile_data.loc[i, "tile_x_offset"] = item["tile_image_shift_pixel_x"] - x_offset
        tile_data.loc[i, "tile_y_offset"] = item["tile_image_shift_pixel_y"] - y_offset
        tile_data.loc[i, "tile_x_offset_binned"] = (
            item["tile_image_shift_pixel_x"] - x_offset
        ) // binning
        tile_data.loc[i, "tile_y_offset_binned"] = (
            item["tile_image_shift_pixel_y"] - y_offset
        ) // binning

    results = {"montage": montage_metadata, "tiles": tile_data}
    starfile.write(results, output_path_metadata, overwrite=True)
    return results

def calculate_diagonal_radius(box_size: int = 512) -> int:
    return int(np.around(np.sqrt(2 * (box_size / 2 - 0.5) ** 2)))

def distance_from_center_array(shape) -> np.ndarray:
    x, y = np.ogrid[0:shape[0], 0:shape[1]]
    r = np.hypot(x - (shape[0] - 1) / 2, y - (shape[1] - 1) / 2)
    return r

def radial_average(spectrum: np.ndarray) -> np.ndarray:
    distance_from_center = distance_from_center_array(spectrum.shape)
    bins = np.around(distance_from_center).astype(np.int32)
    return mean(spectrum, labels=bins, index=np.arange(1, calculate_diagonal_radius(spectrum.shape[0])+1))

def create_montage(montage_metadata: dict, output_path_montage: Path, erode_mask: int = 0, correct_dark_ring: bool = True, dark_ring_start: float = 0.9, dark_ring_windowlength: int = 50):
    import time
    # Create the montage
    prev = time.perf_counter()
    montage = np.zeros(
        (
            int(montage_metadata["montage"]["montage_y_size"].values[0]),
            int(montage_metadata["montage"]["montage_x_size"].values[0]),
        ),
        dtype=np.float32,
    )
    mask_montage = np.zeros(
        (
            int(montage_metadata["montage"]["montage_y_size"].values[0]),
            int(montage_metadata["montage"]["montage_x_size"].values[0]),
        ),
        dtype=np.float32,
    )
    binning = montage_metadata["montage"]["montage_binning"].values[0]
    #print(f"Initialisation took {time.perf_counter() - prev} seconds")
    prev = time.perf_counter()
    # Sort montage_metadate by tile_acquisition_time
    montage_metadata["tiles"] = montage_metadata["tiles"].sort_values(
        by="tile_acquisition_time", ascending=True
    )
    for item in track(
        montage_metadata["tiles"].iterrows(),
        total=len(montage_metadata["tiles"].index),
        description="Creating montage",
    ):
        prev = time.perf_counter()
        item = item[1]
        tile = mrcfile.open(item["tile_filename"]).data.copy()
        tile = tile[0]
        # tile -= np.min(tile)
        tile *= 1000.0
        mask = mrcfile.open(item["tile_mask_filename"]).data.copy()
        mask = mask[0]
        mask.dtype = np.uint8
        mask_float = mask / 255.0
        if erode_mask > 0:
            mask_float = mask_float > 0.5
            mask_float = binary_erosion(mask_float, iterations=erode_mask)
            mask_float = mask_float.astype(np.float32)
            mask_float *= 1.0
        tile_binned_dimensions = (
            int(tile.shape[0] / binning),
            int(tile.shape[1] / binning),
        )
        #print(f"Opening took {time.perf_counter() - prev} seconds")
        prev = time.perf_counter()
        #print(tile.shape)
        if correct_dark_ring and np.abs(tile.shape[0] / tile.shape[1] - 1) < 0.1:
            ra = radial_average(tile)
            shape = tile.shape
            x, y = np.ogrid[0:shape[0], 0:shape[1]]
            size = np.min(shape)
            r = np.hypot(x - (size - 1) / 2, y - (size - 1) / 2)
            correction_image = np.ones_like(r)

            smoothed_curve = savgol_filter(ra[int((shape[0]//2)*0.9):(shape[0]//2)], dark_ring_windowlength, 3)
            smoothed_curve = smoothed_curve / smoothed_curve[0]
            values_to_correct = r[np.where(np.logical_and(r>=int((shape[0]//2)*dark_ring_start), r < shape[0//2]))]
            oldvalues = radial_average(r)[int((shape[0]//2)*0.9):(shape[0]//2)]
            correction_image[np.where(np.logical_and(r>=int((shape[0]//2)*dark_ring_start), r < shape[0//2]))] = np.interp(values_to_correct,oldvalues, smoothed_curve)

            #correction_image[r>=int((tile.shape[0]//2)*dark_ring_start)].flatten() = interp1d(correction_image[r>=int((tile.shape[0]//2)*dark_ring_start)].flatten(), 
            tile = tile / correction_image

        tile = resize(tile, tile_binned_dimensions, anti_aliasing=True)
        mask_float = resize(mask_float, tile_binned_dimensions, anti_aliasing=False)
        mask_float = mask_float > 0.5
        mask_float = mask_float.astype(np.float32)
        mask_float *= 1.0
        #print(f"Resizing took {time.perf_counter() - prev} seconds")
        prev = time.perf_counter()
        insertion_slice = (
            slice(
                int(item["tile_y_offset_binned"]),
                int(item["tile_y_offset_binned"]) + tile_binned_dimensions[0],
            ),
            slice(
                int(item["tile_x_offset_binned"]),
                int(item["tile_x_offset_binned"]) + tile_binned_dimensions[1],
            ),
        )
        tile *= mask_float

        

        existing_mask = 1.0 - mask_montage[insertion_slice]
        tile *= existing_mask
        mask_float *= existing_mask
        print(np.sum(mask_float))
        mask_montage[insertion_slice] += mask_float

        # CHeck if the column tile_intensity_correction exists
        if "tile_intensity_correction" in item:
            tile *= item["tile_intensity_correction"]

        montage[insertion_slice] += tile
        #print(f"Insertion took {time.perf_counter() - prev} seconds")
        
    median_value = np.median(montage[np.nonzero(mask_montage)])
    montage[mask_montage == 0] = median_value
    with mrcfile.new(output_path_montage, overwrite=True) as mrc:
        mrc.set_data(montage)
        mrc.voxel_size = montage_metadata["montage"]["montage_pixel_size"].values[0]
    with mrcfile.new(
        output_path_montage.parent / "montage_mask.mrc", overwrite=True
    ) as mrc:
        mrc.set_data(mask_montage)
        mrc.voxel_size = montage_metadata["montage"]["montage_pixel_size"].values[0]


def adjust_metadata_for_matches(montage_data: dict, match_data: pd.DataFrame, image: str = "PROJECTION_RESULT_OUTPUT_FILE"):

    # Create new column called key in tile data that coontaints the tile filename without
    # the part after the last undersocre
    montage_data["tiles"]["key"] = montage_data["tiles"]["tile_filename"].str.rsplit(
        "_", n=1, expand=True
    )[0]
    # Do the same for the match data
    match_data["key"] = match_data["IMAGE_FILENAME"].str.rsplit("_", n=1, expand=True)[0]

    # Merge the two dataframes on the key column
    merged = pd.merge(montage_data["tiles"], match_data, on="key")
    merged["tile_filename"] = merged[image]
    montage_data["tiles"] = merged
    return montage_data


def find_tile_pairs(tile_data: pd.DataFrame, distance_threshold_A: float):

    points = np.array(
        [
            (
                a["tile_image_shift_pixel_x"] * a["tile_pixel_size"],
                a["tile_image_shift_pixel_y"] * a["tile_pixel_size"],
            )
            for i, a in tile_data.iterrows()
        ]
    )
    kd_tree = cKDTree(points)

    pairs = kd_tree.query_pairs(r=distance_threshold_A)
    row_pairs = [(tile_data.iloc[a[0]], tile_data.iloc[a[1]]) for a in pairs]
    return row_pairs


def calculate_shifts(row_pairs: list, num_proc: int = 1, erode_mask: int = 0, filter_cutoff_frequency_ratio: float = 0.02, filter_order: float = 4.0, mask_size_cutoff: int = 100, overlap_ratio: float = 0.1):
    pool = multiprocessing.Pool(processes=num_proc)

    # map the worker function to the input data using the pool
    results = pool.imap_unordered(
        partial(determine_shift_by_cc2, erode_mask=erode_mask, filter_cutoff_frequency_ratio=filter_cutoff_frequency_ratio, filter_order=filter_order,mask_size_cutoff=mask_size_cutoff, overlap_ratio=overlap_ratio), row_pairs
    )
    shifts = []
    # use the rich.progress module to track the progress of the results
    for result in track(
        results,
        total=len(row_pairs),
        description="Calculating shifts from pairwise cross-correlations",
    ):
        shifts.append(result)
        # process the result here

    shifts = pd.DataFrame([a for a in shifts if a is not None])
    return shifts


def determine_shift_by_cc(
    doubled,
    erode_mask: float = 0,
    filter_cutoff_frequency_ratio: float = 0.02,
    filter_order=4.0,
    mask_size_cutoff: int = 100,
    overlap_ratio: float = 0.1,
    debug: bool = False,
    debug_object = {}
):
    import time
    # Create the montage
    prev = time.perf_counter()
    # Given the infow of two images, calculate the refined relative shifts by crosscorrelation return
    im1, im2 = doubled

    with mrcfile.open(im1["tile_filename"]) as mrc:
        reference = np.copy(mrc.data[0])
        reference = filters.butterworth(
            reference,
            cutoff_frequency_ratio=filter_cutoff_frequency_ratio,
            order=filter_order,
            high_pass=False,
        )
        if debug:
            debug_object["reference"] = reference.copy()
    with mrcfile.open(im2["tile_filename"]) as mrc:
        moving = np.copy(mrc.data[0])
        moving = filters.butterworth(
            moving,
            cutoff_frequency_ratio=filter_cutoff_frequency_ratio,
            order=filter_order,
            high_pass=False,
        )
        if debug:
            debug_object["moving"] = moving.copy()
    #print(f"Loading images took {time.perf_counter() - prev} seconds")
    prev = time.perf_counter()
    diff = (
        im2["tile_image_shift_pixel_x"] - im1["tile_image_shift_pixel_x"],
        im2["tile_image_shift_pixel_y"] - im1["tile_image_shift_pixel_y"],
    )
    tform = transform.SimilarityTransform(translation=(diff[0], diff[1])).inverse
    moving = transform.warp(moving, tform, output_shape=reference.shape)
    if debug:
        debug_object["moving_moved"] = moving.copy()
    #print(f"Transforming images took {time.perf_counter() - prev} seconds")
    prev = time.perf_counter()
    with mrcfile.open(im1["tile_mask_filename"]) as mrc:
        reference_mask = np.copy(mrc.data[0])
        reference_mask.dtype = np.uint8
        reference_mask = reference_mask / 255.0
    with mrcfile.open(im2["tile_mask_filename"]) as mrc:
        moving_mask = np.copy(mrc.data[0])
        moving_mask.dtype = np.uint8
        moving_mask = moving_mask / 255.0
    #print(f"Opening masks took {time.perf_counter() - prev} seconds")
    prev = time.perf_counter()
    if erode_mask > 0:
        reference_mask = reference_mask > 0.5
        moving_mask = moving_mask > 0.5
        reference_mask = binary_erosion(reference_mask, iterations=erode_mask)
        moving_mask = binary_erosion(moving_mask, iterations=erode_mask)
    #print(f"Eroding masks took {time.perf_counter() - prev} seconds")
    prev = time.perf_counter()
    moving_mask = transform.warp(moving_mask, tform, output_shape=reference_mask.shape)
    mask = np.minimum(reference_mask, moving_mask) > 0.9
    if debug:
        debug_object["mask"] = mask.copy()
    if np.sum(mask) < mask_size_cutoff:
        return None
    reference *= mask
    moving *= mask
    #print(f"precross took {time.perf_counter() - prev} seconds")
    prev = time.perf_counter()
    xcorr = cross_correlate_masked(
        moving,
        reference,
        mask,
        mask,
        axes=tuple(range(moving.ndim)),
        mode="full",
        overlap_ratio=overlap_ratio,
    )
    if debug:
        debug_object["xcorr"] = xcorr.copy()
    #print(f"Cross took {time.perf_counter() - prev} seconds")
    prev = time.perf_counter()
    # Generalize to the average of multiple equal maxima
    maxima = np.stack(np.nonzero(xcorr == xcorr.max()), axis=1)
    center = np.mean(maxima, axis=0)
    shift = center - np.array(reference.shape) + 1
    shift = -shift

    with np.errstate(all="raise"):
        try:
            ratio = np.sum(reference) / np.sum(moving)
        except FloatingPointError:
            ratio = 1
    #print(f"Final took {time.perf_counter() - prev} seconds")
    prev = time.perf_counter()
    return {
        "shift_x": diff[0] + shift[1],
        "shift_y": diff[1] + shift[0],
        "initial_area": np.sum(mask),
        "max_cc": xcorr.max(),
        "add_shift": np.linalg.norm(shift),
        "int_ratio": ratio,
        "image_1": im1["tile_filename"],
        "image_2": im2["tile_filename"],
    }

def determine_shift_by_cc2(
    doubled,
    erode_mask: float = 0,
    filter_cutoff_frequency_ratio: float = 0.02,
    filter_order=4.0,
    mask_size_cutoff: int = 100,
    overlap_ratio: float = 0.1,
    debug: bool = False,
    debug_object = {}
):
    # Given the infow of two images, calculate the refined relative shifts by crosscorrelation return
    im1, im2 = doubled

    # Open the masks
    with mrcfile.open(im1["tile_mask_filename"]) as mrc:
        reference_mask = np.copy(mrc.data[0])
        reference_mask.dtype = np.uint8
        reference_mask = reference_mask / 255.0
    with mrcfile.open(im2["tile_mask_filename"]) as mrc:
        moving_mask = np.copy(mrc.data[0])
        moving_mask.dtype = np.uint8
        moving_mask = moving_mask / 255.0

    if erode_mask > 0:
        reference_mask = reference_mask > 0.5
        moving_mask = moving_mask > 0.5
        reference_mask = binary_erosion(reference_mask, iterations=erode_mask)
        moving_mask = binary_erosion(moving_mask, iterations=erode_mask)

    # Transform mask2
    diff = (
        int(im2["tile_image_shift_pixel_x"] - im1["tile_image_shift_pixel_x"]),
        int(im2["tile_image_shift_pixel_y"] - im1["tile_image_shift_pixel_y"]),
    ) 
    tform = transform.SimilarityTransform(translation=(diff[0], diff[1])).inverse
    moving_mask = transform.warp(moving_mask, tform, output_shape=reference_mask.shape)
    mask = np.minimum(reference_mask, moving_mask) > 0.9
    if np.sum(mask) < mask_size_cutoff:
        return None
    # Get bounding box of the mask
    bbox = np.array([np.min(np.nonzero(mask)[0]), np.max(np.nonzero(mask)[0]), np.min(np.nonzero(mask)[1]), np.max(np.nonzero(mask)[1])])
    # Calculate bounding box for moving
    bbox_moving = np.array([bbox[0] - diff[1], bbox[1] - diff[1], bbox[2] - diff[0], bbox[3] - diff[0]]) 
    mask = mask[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    

    with mrcfile.open(im1["tile_filename"]) as mrc:
        reference = np.copy(mrc.data[0])
        # Cut out the bounding box
        reference = reference[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        reference = filters.butterworth(
            reference,
            cutoff_frequency_ratio=filter_cutoff_frequency_ratio,
            order=filter_order,
            high_pass=False,
        )
        if debug:
            debug_object["reference"] = reference.copy()
    with mrcfile.open(im2["tile_filename"]) as mrc:
        moving = np.copy(mrc.data[0])
        # Cut out the bounding box
        moving = moving[bbox_moving[0]:bbox_moving[1], bbox_moving[2]:bbox_moving[3]]
        moving = filters.butterworth(
            moving,
            cutoff_frequency_ratio=filter_cutoff_frequency_ratio,
            order=filter_order,
            high_pass=False,
        )
        if debug:
            debug_object["moving"] = moving.copy()
    
   


    
    reference *= mask
    moving *= mask
   
    xcorr = cross_correlate_masked(
        moving,
        reference,
        mask,
        mask,
        axes=tuple(range(moving.ndim)),
        mode="full",
        overlap_ratio=overlap_ratio,
    )
    if debug:
        debug_object["xcorr"] = xcorr.copy()
    #print(f"Cross took {time.perf_counter() - prev} seconds")
    # Generalize to the average of multiple equal maxima
    maxima = np.stack(np.nonzero(xcorr == xcorr.max()), axis=1)
    center = np.mean(maxima, axis=0)
    shift = center - np.array(reference.shape) + 1
    shift = -shift

    with np.errstate(all="raise"):
        try:
            ratio = np.sum(reference) / np.sum(moving)
        except FloatingPointError:
            ratio = 1
    return {
        "shift_x": diff[0] + shift[1],
        "shift_y": diff[1] + shift[0],
        "initial_area": np.sum(mask),
        "max_cc": xcorr.max(),
        "add_shift": np.linalg.norm(shift),
        "int_ratio": ratio,
        "image_1": im1["tile_filename"],
        "image_2": im2["tile_filename"],
    }


def _position_residuals(is_pixel, index_image_1, index_image_2, shifts):
    distance = (is_pixel[index_image_2] - is_pixel[index_image_1]) - shifts
    return distance


def calculate_refined_image_shifts(tile_data, shifts):
    tile_data["filename_index"] = tile_data["tile_filename"]
    tile_data.set_index("filename_index", inplace=True)
    initial_pixel_coordinates = np.array(
        [
            (a["tile_image_shift_pixel_x"], a["tile_image_shift_pixel_y"])
            for i, a in tile_data.iterrows()
        ]
    )

    indices_image1 = np.array(
        [
            np.repeat(
                [
                    tile_data.index.get_loc(shift["image_1"])
                    for i, shift in shifts.iterrows()
                ],
                2,
            ),
            np.tile([0, 1], len(shifts)),
        ]
    )
    indices_image1 = np.ravel_multi_index(
        indices_image1, initial_pixel_coordinates.shape
    )
    indices_image2 = np.array(
        [
            np.repeat(
                [
                    tile_data.index.get_loc(shift["image_2"])
                    for i, shift in shifts.iterrows()
                ],
                2,
            ),
            np.tile([0, 1], len(shifts)),
        ]
    )
    indices_image2 = np.ravel_multi_index(
        indices_image2, initial_pixel_coordinates.shape
    )
    shifts_array = shifts[["shift_x", "shift_y"]].to_numpy().flatten()
    initial_pixel_coordinates = initial_pixel_coordinates.flatten()

    res = optimize.least_squares(
        _position_residuals,
        x0=initial_pixel_coordinates,
        bounds=(initial_pixel_coordinates - 5000, initial_pixel_coordinates + 5000),
        args=(indices_image1, indices_image2, shifts_array),
    )
    new_positions = res.x.reshape(int(len(res.x) / 2), 2)
    tile_data["tile_image_shift_pixel_x_original"] = tile_data[
        "tile_image_shift_pixel_x"
    ]
    tile_data["tile_image_shift_pixel_y_original"] = tile_data[
        "tile_image_shift_pixel_y"
    ]
    tile_data["tile_image_shift_pixel_x"] = [a[0] for a in new_positions]
    tile_data["tile_image_shift_pixel_y"] = [a[1] for a in new_positions]


def _intensity_residuals(
    intensity_correction, shifts, indices_image_1, indices_image_2
):
    distance = 1.0 - (
        np.array(shifts["int_ratio"]) * intensity_correction[indices_image_1]
    ) / (intensity_correction[indices_image_2])
    return distance


def calculate_refined_intensity(tile_data, shifts):
    intensity_correction = np.array([1.0 for i in range(len(tile_data))])
    min_cor = np.array([0.2 for i in range(len(tile_data))])
    max_cor = np.array([5.0 for i in range(len(tile_data))])
    indices_image_1 = np.array(
        [tile_data.index.get_loc(shift["image_1"]) for i, shift in shifts.iterrows()]
    )
    indices_image_2 = np.array(
        [tile_data.index.get_loc(shift["image_2"]) for i, shift in shifts.iterrows()]
    )

    res = optimize.least_squares(
        _intensity_residuals,
        x0=intensity_correction,
        bounds=(min_cor, max_cor),
        args=(shifts, indices_image_1, indices_image_2),
    )
    tile_data["tile_intensity_correction"] = res.x


def assemble_matches(montage_info, refine_info):
    refine_info["cisTEMOriginalImageFilename"] = refine_info[
        "cisTEMOriginalImageFilename"
    ].str.replace("'", "")
    info = montage_info["tiles"].merge(
        refine_info,
        how="inner",
        left_on="tile_filename",
        right_on="cisTEMOriginalImageFilename",
    )
    # print(info.loc[0])
    info["tile_x"] = info["cisTEMOriginalXPosition"]
    info["tile_y"] = info["cisTEMOriginalYPosition"]
    info["cisTEMOriginalXPosition"] = (
        info["tile_x"] + info["tile_x_offset"] * info["cisTEMPixelSize"]
    )

    info["cisTEMOriginalYPosition"] = (
        info["tile_y"] + info["tile_y_offset"] * info["cisTEMPixelSize"]
    )

    info["cisTEMOriginalImageFilename"] = montage_info["montage"][
        "montage_filename"
    ].loc[0]
    info["cisTEMPixelSize"] = montage_info["montage"]["montage_pixel_size"].loc[0]

    info["mask_value"] = 0.0
    info["display"] = False
    # Check the value of the mask at each match
    for mask_filename in info["tile_mask_filename"].unique():
        # Open mask and convert to 0-1 floats
        if mask_filename is None:
            continue
        with mrcfile.open(mask_filename) as mask:
            mask_data = np.copy(mask.data[0])
            mask_data.dtype = np.uint8
            mask_float = mask_data / 255.0
        #
        peak_indices = info.loc[
            info["tile_mask_filename"] == mask_filename
        ].index.tolist()
        for i in peak_indices:
            try:
                info.loc[i, "mask_value"] = mask_float[
                    int(info.loc[i, "tile_y"] / info.loc[i, "tile_pixel_size"]),
                    int(info.loc[i, "tile_x"] / info.loc[i, "tile_pixel_size"]),
                ]
            except IndexError:
                info.loc[i, "mask_value"] = 0
            if info.loc[i, "mask_value"] > 0.7:
                info.loc[i, "display"] = True
    # Correct for defocus and nominal focus
    # median_defocus = info["tile_defocus"].median()
    median_microscope_focus = info["tile_microscope_focus"].median()
    # defocus_correction = info["tile_defocus"] - median_defocus
    microscope_focus_correction = (
        info["tile_microscope_focus"] - median_microscope_focus
    )
    # info["defocus"] += defocus_correction
    info["cisTEMDefocus1"] = (
        microscope_focus_correction * 10000 + info["cisTEMDefocus1"]
    )
    info["cisTEMDefocus2"] = (
        microscope_focus_correction * 10000 + info["cisTEMDefocus2"]
    )
    # Write the new starfile
    return info

def create_tile_borders_overlay(montage_metadata: dict, output_path_overlay: Path, project_path: Path,erode_mask: int = 0):
    import matplotlib.pyplot as plt
    import numpy
    import sqlite3

    scale_factor = 500

    w = montage_metadata["montage"]["montage_x_size"].values[0] / scale_factor
    h = montage_metadata["montage"]["montage_y_size"].values[0] / scale_factor

    fig = plt.figure(frameon=False)
    fig.set_size_inches(w,h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    
    
    db = sqlite3.connect(project_path)
    thickness_info = pd.read_sql_query("SELECT IMAGE_ASSETS.FILENAME, IMAGE_ASSETS.IMAGE_ASSET_ID, ESTIMATED_CTF_PARAMETERS.SAMPLE_THICKNESS, ESTIMATED_CTF_PARAMETERS.DETECTED_RING_RESOLUTION FROM ESTIMATED_CTF_PARAMETERS INNER JOIN IMAGE_ASSETS ON ESTIMATED_CTF_PARAMETERS.IMAGE_ASSET_ID = IMAGE_ASSETS.IMAGE_ASSET_ID", db)
    info = montage_metadata['tiles'].merge(thickness_info, left_on='tile_filename', right_on='FILENAME')

    x_centers = info['tile_x_offset_binned'] + info['tile_x_size'] / (2 * montage_metadata['montage']['montage_binning'][0])
    y_centers = info['tile_y_offset_binned'] + info['tile_y_size'] / (2 * montage_metadata['montage']['montage_binning'][0])
    
    
    ax.scatter(x_centers, y_centers, s=3000,facecolors='none', edgecolors='r')
    # Write text into the plot
    for i, txt in enumerate(info['SAMPLE_THICKNESS']):
        ax.text(x_centers[i], y_centers[i], f"{info['IMAGE_ASSET_ID'][i]}: {txt:.2f}", color='r', fontsize=3)
    ax.set_xlim(0, w * scale_factor)
    ax.set_ylim(0, h * scale_factor)
    fig.add_axes(ax)
    fig.savefig(output_path_overlay, dpi=scale_factor)
    plt.close()