import pandas as pd
import numpy as np
import sqlite3
import contextlib

from pathlib import Path

def read_data_from_cistem(database_filename: Path) -> pd.DataFrame:
    with contextlib.closing(sqlite3.connect(database_filename)) as con:

        df1 = pd.read_sql_query(f"SELECT IMAGE_ASSET_ID,MOVIE_ASSET_ID,IMAGE_ASSETS.FILENAME, MOVIE_ASSETS.FILENAME as movie_filename, CTF_ESTIMATION_ID , ALIGNMENT_ID, IMAGE_ASSETS.PIXEL_SIZE as image_pixel_size, MOVIE_ASSETS.PIXEL_SIZE as movie_pixel_size, IMAGE_ASSETS.X_SIZE, IMAGE_ASSETS.Y_SIZE, IMAGE_ASSETS.ORIGINAL_X_SIZE, IMAGE_ASSETS.ORIGINAL_Y_SIZE, IMAGE_ASSETS.CROP_CENTER_X, IMAGE_ASSETS.CROP_CENTER_Y  FROM IMAGE_ASSETS INNER JOIN MOVIE_ASSETS ON MOVIE_ASSETS.MOVIE_ASSET_ID == IMAGE_ASSETS.PARENT_MOVIE_ID", con)
        df2 = pd.read_sql_query("SELECT CTF_ESTIMATION_ID,DEFOCUS1,DEFOCUS2,DEFOCUS_ANGLE,OUTPUT_DIAGNOSTIC_FILE,SCORE, DETECTED_RING_RESOLUTION FROM ESTIMATED_CTF_PARAMETERS",con)
        selected_micrographs = pd.merge(df1,df2,on="CTF_ESTIMATION_ID")
        df3 = pd.read_sql_query(f"SELECT MOVIE_ASSET_ID,IMAGE_SHIFT_X, IMAGE_SHIFT_Y, CONTENT_JSON FROM MOVIE_ASSETS_METADATA",con)
        selected_micrographs = pd.merge(selected_micrographs,df3,on="MOVIE_ASSET_ID")
        
    return(selected_micrographs)


def create_tile_metadata(cistem_data: pd.DataFrame, decolace_data: dict, output_filename: Path):
    

    result_tiles = pd.DataFrame({
        'tile_filename': pd.Series(dtype='object'),
        'tile_movie_filename': pd.Series(dtype='object'),
        'tile_x_offset': pd.Series(dtype='int'),
        'tile_y_offset': pd.Series(dtype='int'),
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
        cistem_data.loc[i,"image_shift_pixel_y"] = - shift_in_image_pixel[1] + crop_y - crop_center_y
        
    
    # Get the range of the shifts taking binning into account
    max_image_x = np.max(expand_data.loc[:,"X_SIZE"]/binning)
    max_image_y = np.max(expand_data.loc[:,"Y_SIZE"]/binning)
    min_shift_x = np.min(expand_data.loc[:,"image_shift_pixel_x"]/binning) - 2 * binning
    min_shift_y = np.min(expand_data.loc[:,"image_shift_pixel_y"]/binning) - 2 * binning
    max_shift_x = np.max(expand_data.loc[:,"image_shift_pixel_x"]/binning) + max_image_x + 2 * binning
    max_shift_y = np.max(expand_data.loc[:,"image_shift_pixel_y"]/binning) + max_image_y + 2 * binning

    # Decide on dimensions of montage
    montage_y_size = int(max_shift_y-min_shift_y)
    montage_x_size = int(max_shift_x-min_shift_x)
    montage_pixel_size = expand_data['image_pixel_size'].iloc[0] * binning

    result_montage.loc[0] =  {
        'montage_filename': output_base + "_montage.tif",
        'matches_montage_filename': output_base+"_matches_montage.tif",
        'montage_pixel_size': montage_pixel_size,
        'montage_binning': binning,
        'montage_x_size': montage_x_size,
        'montage_y_size': montage_y_size,
    }
    
    # No iterate again, to setup the required data
    for i, item in expand_data.iterrows():
        new_entry = {}
        #print(item['FILENAME'])
        new_entry['tile_filename'] = item['FILENAME'][:-6]+image_suffix
        new_entry['tile_movie_filename'] = item['movie_filename']
        new_entry['tile_mask_filename'] = item['FILENAME'][:-6]+mask_suffix
        new_entry['tile_pixel_size'] = item['image_pixel_size']
        new_entry['tile_x_size'] = item['X_SIZE']
        new_entry['tile_y_size'] = item['Y_SIZE']
        new_entry['tile_orig_x_size'] = item['ORIGINAL_X_SIZE']
        new_entry['tile_orig_y_size'] = item['ORIGINAL_Y_SIZE']
        new_entry['tile_x_crop_center'] = item['CROP_CENTER_X']
        new_entry['tile_y_crop_center'] = item['CROP_CENTER_Y']
        new_entry['tile_image_shift_x'] = item['IMAGE_SHIFT_X']
        new_entry['tile_image_shift_y'] = item['IMAGE_SHIFT_Y']
        new_entry['tile_microscope_focus'] = json.loads(item['CONTENT_JSON'])["Defocus"]
        new_entry['tile_defocus'] = (item['DEFOCUS1'] + item['DEFOCUS2'])/2
        matches_dir = Path(item['FILENAME']).parent.parent / 'TemplateMatching' 
        #print(Path(item['FILENAME'][:-6]).name +"*plotted_result*.mrc")
        matches_filenames = list(matches_dir.glob(Path(item['FILENAME'][:-6]).name +"*plotted_result*.mrc"))
        #print(matches_filenames)
        if len(matches_filenames) > 0:
            new_entry['tile_plotted_result_filename'] = str(matches_filenames[0])
        else:
            new_entry['tile_plotted_result_filename'] = "None"
        # Calulate the offset
        insert_point_x = item["image_shift_pixel_x"] 
        insert_point_y = item["image_shift_pixel_y"]
        insert_point_x /= binning
        insert_point_y /= binning
        insert_point_x -= min_shift_x
        insert_point_y -= min_shift_y

        new_entry['tile_x_offset'] = insert_point_x * montage_pixel_size
        new_entry['tile_y_offset'] = insert_point_y * montage_pixel_size
        
        result_tiles.loc[len(result_tiles.index)] = new_entry

    results = {
        "montage": result_montage,
        "tiles": result_tiles
    }
    print(result_montage.shape)
    starfile.write(results, output_base + ".star",overwrite=True)

    return(results)