import pandas as pd
import numpy as np
import sqlite3
import contextlib    
from pathlib import Path
import json
import starfile
import mrcfile
from skimage.transform import resize
from rich.progress import track

def read_data_from_cistem(database_filename: Path) -> pd.DataFrame:
    with contextlib.closing(sqlite3.connect(database_filename)) as con:

        df1 = pd.read_sql_query(f"SELECT IMAGE_ASSET_ID,MOVIE_ASSET_ID,IMAGE_ASSETS.FILENAME, MOVIE_ASSETS.FILENAME as movie_filename, CTF_ESTIMATION_ID , ALIGNMENT_ID, IMAGE_ASSETS.PIXEL_SIZE as image_pixel_size, MOVIE_ASSETS.PIXEL_SIZE as movie_pixel_size, IMAGE_ASSETS.X_SIZE, IMAGE_ASSETS.Y_SIZE  FROM IMAGE_ASSETS INNER JOIN MOVIE_ASSETS ON MOVIE_ASSETS.MOVIE_ASSET_ID == IMAGE_ASSETS.PARENT_MOVIE_ID", con)
        df2 = pd.read_sql_query("SELECT CTF_ESTIMATION_ID,DEFOCUS1,DEFOCUS2,DEFOCUS_ANGLE,OUTPUT_DIAGNOSTIC_FILE,SCORE, DETECTED_RING_RESOLUTION FROM ESTIMATED_CTF_PARAMETERS",con)
        selected_micrographs = pd.merge(df1,df2,on="CTF_ESTIMATION_ID")
        df3 = pd.read_sql_query(f"SELECT MOVIE_ASSET_ID,IMAGE_SHIFT_X, IMAGE_SHIFT_Y, CONTENT_JSON FROM MOVIE_ASSETS_METADATA",con)
        selected_micrographs = pd.merge(selected_micrographs,df3,on="MOVIE_ASSET_ID")
        df4 = pd.read_sql_query(f"SELECT ALIGNMENT_ID, ORIGINAL_X_SIZE, ORIGINAL_Y_SIZE, CROP_CENTER_X, CROP_CENTER_Y FROM MOVIE_ALIGNMENT_LIST",con)
        selected_micrographs = pd.merge(selected_micrographs,df4,on="ALIGNMENT_ID")
    return(selected_micrographs)

def read_matches_data(database_filename: Path, tm_job_id: int) -> pd.DataFrame:
    with contextlib.closing(sqlite3.connect(database_filename)) as con:
        matches_data = pd.read_sql_query(f"SELECT IMAGE_ASSETS.FILENAME AS IMAGE_FILENAME, PROJECTION_RESULT_OUTPUT_FILE FROM TEMPLATE_MATCH_LIST INNER JOIN IMAGE_ASSETS ON TEMPLATE_MATCH_LIST.IMAGE_ASSET_ID = IMAGE_ASSETS.IMAGE_ASSET_ID WHERE TEMPLATE_MATCH_JOB_ID={tm_job_id}", con)
    return matches_data

def read_decolace_data(decolace_filename: Path) -> dict:
    
    return np.load(decolace_filename, allow_pickle=True).item()


def create_tile_metadata(cistem_data: pd.DataFrame, decolace_data: dict, output_filename: Path):
    IS_to_camera = decolace_data['record_IS_to_camera'].reshape(2,2)

    result_tiles = pd.DataFrame({
        'tile_filename': pd.Series(dtype='object'),
        'tile_movie_filename': pd.Series(dtype='object'),
        
        'tile_pixel_size': pd.Series(dtype='float'),
        'tile_x_size': pd.Series(dtype='int'),
        'tile_y_size': pd.Series(dtype='int'),
        'tile_orig_x_size': pd.Series(dtype='int'),
        'tile_orig_y_size': pd.Series(dtype='int'),
        'tile_x_crop_center': pd.Series(dtype='int'),
        'tile_y_crop_center': pd.Series(dtype='int'),
        'tile_mask_filename': pd.Series(dtype='object'),
        'tile_plotted_result_filename': pd.Series(dtype='object'),
        'tile_image_shift_x': pd.Series(dtype='float'),
        'tile_image_shift_y': pd.Series(dtype='float'),
        'tile_image_shift_pixel_x': pd.Series(dtype='int'),
        'tile_image_shift_pixel_y': pd.Series(dtype='int'),
        'tile_microscope_focus': pd.Series(dtype='float'),
        'tile_defocus': pd.Series(dtype='float'),
    })


    for i, item in cistem_data.iterrows():

        # Initially convert image shift to unbinned pixels
        image_shift = np.array([item["IMAGE_SHIFT_X"],item["IMAGE_SHIFT_Y"]])
        shift_in_camera_pixels = np.dot(IS_to_camera,image_shift)
        shift_in_image_pixel = (shift_in_camera_pixels / (item["image_pixel_size"]/(item["movie_pixel_size"])))
        crop_x = (item["ORIGINAL_X_SIZE"] - item["X_SIZE"]) / 2
        crop_y = (item["ORIGINAL_Y_SIZE"] - item["Y_SIZE"]) / 2
        crop_center_x = item["CROP_CENTER_X"] 
        crop_center_y = item["CROP_CENTER_Y"] 
        cistem_data.loc[i,"image_shift_pixel_x"] = shift_in_image_pixel[0] + crop_x + crop_center_x
        cistem_data.loc[i,"image_shift_pixel_y"] = - shift_in_image_pixel[1] + crop_y + crop_center_y
        
    # Now iterate again, to setup the required data
    for i, item in cistem_data.iterrows():
        new_entry = {}
        new_entry['tile_filename'] = item['FILENAME']
        new_entry['tile_movie_filename'] = item['movie_filename']
        new_entry['tile_mask_filename'] = item['FILENAME'][:-4]+"_mask.mrc"
        new_entry['tile_pixel_size'] = item['image_pixel_size']
        new_entry['tile_x_size'] = item['X_SIZE']
        new_entry['tile_y_size'] = item['Y_SIZE']
        new_entry['tile_orig_x_size'] = item['ORIGINAL_X_SIZE']
        new_entry['tile_orig_y_size'] = item['ORIGINAL_Y_SIZE']
        new_entry['tile_x_crop_center'] = item['CROP_CENTER_X']
        new_entry['tile_y_crop_center'] = item['CROP_CENTER_Y']
        new_entry['tile_image_shift_x'] = item['IMAGE_SHIFT_X']
        new_entry['tile_image_shift_y'] = item['IMAGE_SHIFT_Y']
        new_entry['tile_image_shift_pixel_x'] = item['image_shift_pixel_x']
        new_entry['tile_image_shift_pixel_y'] = item['image_shift_pixel_y']
        new_entry['tile_microscope_focus'] = json.loads(item['CONTENT_JSON'])["Defocus"]
        new_entry['tile_defocus'] = (item['DEFOCUS1'] + item['DEFOCUS2'])/2
        matches_dir = Path(item['FILENAME']).parent.parent / 'TemplateMatching' 
        matches_filenames = list(matches_dir.glob(Path(item['FILENAME'][:-6]).name +"*plotted_result*.mrc"))
        if len(matches_filenames) > 0:
            new_entry['tile_plotted_result_filename'] = str(matches_filenames[0])
        else:
            new_entry['tile_plotted_result_filename'] = "None"
       
        
        result_tiles.loc[len(result_tiles.index)] = new_entry

    results = {
        "tiles": result_tiles
    }
    starfile.write(results, output_filename,overwrite=True)

    return(results)

def create_montage_metadata(tile_data: pd.DataFrame, output_path_metadata: Path, binning: int):
    # Create the montage metadata
    montage_metadata =  pd.DataFrame({
        'montage_filename': pd.Series(dtype='string'),
        'matches_montage_filename': pd.Series(dtype='string'),
        'montage_pixel_size': pd.Series(dtype='float'),
        'montage_binning': pd.Series(dtype='float'),
        'montage_x_size': pd.Series(dtype='int'),
        'montage_y_size': pd.Series(dtype='int'),
    })
    unbinned_size_x = tile_data["tile_image_shift_pixel_x"].max() + tile_data["tile_x_size"].max() - tile_data["tile_image_shift_pixel_x"].min()
    unbinned_size_y = tile_data["tile_image_shift_pixel_y"].max() + tile_data["tile_y_size"].max() - tile_data["tile_image_shift_pixel_y"].min()
   

    x_offset = tile_data["tile_image_shift_pixel_x"].min()
    y_offset = tile_data["tile_image_shift_pixel_y"].min()

    binned_size_x = unbinned_size_x // binning
    binned_size_y = unbinned_size_y // binning
    pixel_size = tile_data["tile_pixel_size"].values[0] * binning
    x_offset_binned = x_offset // binning
    y_offset_binned = y_offset // binning

    montage_metadata.loc[0,"montage_filename"] = "montage.mrc"
    montage_metadata.loc[0,"matches_montage_filename"] = "montage_matches.mrc"
    montage_metadata.loc[0,"montage_pixel_size"] = pixel_size
    montage_metadata.loc[0,"montage_binning"] = binning
    montage_metadata.loc[0,"montage_x_size"] = binned_size_x
    montage_metadata.loc[0,"montage_y_size"] = binned_size_y

    for i, item in tile_data.iterrows():
        tile_data.loc[i, "tile_x_offset"] = item["tile_image_shift_pixel_x"] - x_offset
        tile_data.loc[i, "tile_y_offset"] = item["tile_image_shift_pixel_y"] - y_offset
        tile_data.loc[i, "tile_x_offset_binned"] = (item["tile_image_shift_pixel_x"] - x_offset) // binning
        tile_data.loc[i, "tile_y_offset_binned"] = (item["tile_image_shift_pixel_y"] - y_offset) // binning

    results = {
        "montage": montage_metadata,
        "tiles": tile_data
    }
    starfile.write(results, output_path_metadata,overwrite=True)
    return(results)

def create_montage(montage_metadata: dict, output_path_montage: Path):
    # Create the montage
    montage = np.zeros((int(montage_metadata["montage"]["montage_y_size"].values[0]), int(montage_metadata["montage"]["montage_x_size"].values[0])), dtype=np.float32)
    mask_montage = np.zeros((int(montage_metadata["montage"]["montage_y_size"].values[0]), int(montage_metadata["montage"]["montage_x_size"].values[0])), dtype=np.float32)
    binning = montage_metadata["montage"]["montage_binning"].values[0]
    for item in track(montage_metadata["tiles"].iterrows(), total=len(montage_metadata["tiles"].index), description="Creating montage"):
        item = item[1]
        tile = mrcfile.open(item["tile_filename"]).data.copy()
        tile = tile[0]
        #tile -= np.min(tile)
        tile *= 1000.0
        mask = mrcfile.open(item["tile_mask_filename"]).data.copy()
        mask = mask[0]
        mask.dtype = np.uint8
        mask_float  = mask / 255.0
        tile_binned_dimensions = (int(tile.shape[0] / binning), int(tile.shape[1] / binning))
        tile =  resize(tile,tile_binned_dimensions,anti_aliasing=True)
        mask_float = resize(mask_float,tile_binned_dimensions,anti_aliasing=True)

        insertion_slice = (
            slice(int(item["tile_y_offset_binned"]),
                int(item["tile_y_offset_binned"])+tile_binned_dimensions[0]),
            slice(int(item["tile_x_offset_binned"]),
                int(item["tile_x_offset_binned"])+tile_binned_dimensions[1])
        )
        tile *= mask_float
        existing_mask = 1.0 - mask_montage[insertion_slice]
        tile *= existing_mask
        mask_float *= existing_mask
        mask_montage[insertion_slice] += mask_float
        montage[insertion_slice] += tile

    with mrcfile.new(output_path_montage, overwrite=True) as mrc:
         mrc.set_data(montage)
         mrc.voxel_size = montage_metadata["montage"]["montage_pixel_size"].values[0]

def adjust_metadata_for_matches(montage_data: dict, match_data: pd.DataFrame):

    # Create new column called key in tile data that coontaints the tile filename without
    # the part after the last undersocre
    montage_data['tiles']['key'] = montage_data['tiles']['tile_filename'].str.rsplit('_', 1, expand=True)[0]
    # Do the same for the match data
    match_data['key'] = match_data['IMAGE_FILENAME'].str.rsplit('_', 1, expand=True)[0]

    # Merge the two dataframes on the key column
    merged = pd.merge(montage_data['tiles'], match_data, on='key')
    merged['tile_filename'] = merged['PROJECTION_RESULT_OUTPUT_FILE']
    montage_data['tiles'] = merged
    return montage_data