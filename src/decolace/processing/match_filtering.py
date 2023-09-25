import typer
from pathlib import Path

def get_distance_to_edge(orig_image_filename,refined_matches,binning_boxsize):
    from scipy.ndimage import distance_transform_edt
    import mrcfile
    import numpy as np
    image_filename = orig_image_filename.strip("'")
    
    # Compute distance to mask edge
    mask_filename = image_filename.replace(".mrc", "_mask.mrc")
    if not Path(mask_filename).exists():
        typer.echo(f"No mask for {image_filename}")
        return
    with mrcfile.open(mask_filename.strip("'")) as mask:
        mask_data = np.copy(mask.data[0])
        mask_data.dtype = np.uint8
        mask_binary = mask_data > 128
        distance_transform = distance_transform_edt(mask_binary)
    peak_indices = refined_matches.loc[
        refined_matches["cisTEMOriginalImageFilename"] == orig_image_filename
    ].index.tolist()
    for i in peak_indices:
        pixel_position_x = int(refined_matches.loc[i, "cisTEMOriginalXPosition"] / refined_matches.loc[i, "cisTEMPixelSize"])
        pixel_position_y = int(refined_matches.loc[i, "cisTEMOriginalYPosition"] / refined_matches.loc[i, "cisTEMPixelSize"])
        try:
            refined_matches.loc[i, "LACEBeamEdgeDistance"] = distance_transform[
                pixel_position_y,
                pixel_position_x,
            ]
        except IndexError:
            refined_matches.loc[i, "LACEBeamEdgeDistance"] = 0
    # Compute variance after binning
    with mrcfile.open(image_filename) as image:
        micrograph = image.data
        if micrograph.ndim == 3:
            micrograph = micrograph[0]
    for i in peak_indices:
        pixel_position_x = int(refined_matches.loc[i, "cisTEMOriginalXPosition"] / refined_matches.loc[i, "cisTEMPixelSize"])
        pixel_position_y = int(refined_matches.loc[i, "cisTEMOriginalYPosition"] / refined_matches.loc[i, "cisTEMPixelSize"])
        if pixel_position_x < binning_boxsize // 2 or pixel_position_y < binning_boxsize // 2:
            refined_matches.loc[i, "LACEVarianceAfterBinning"] = -1
            continue
        if pixel_position_x > micrograph.shape[1] - binning_boxsize // 2 or pixel_position_y > micrograph.shape[0] - binning_boxsize // 2:
            refined_matches.loc[i, "LACEVarianceAfterBinning"] = -1
            continue
        binning_box = micrograph[
            pixel_position_y - binning_boxsize // 2 : pixel_position_y + binning_boxsize // 2,
            pixel_position_x - binning_boxsize // 2 : pixel_position_x + binning_boxsize // 2,
        ].copy()
        binning_box -= np.mean(binning_box)
        binning_box = binning_box / np.sqrt(np.var(binning_box))
        # Bin by 4
        binning_box = binning_box.reshape(
            binning_box.shape[0] // 4,
            4,
            binning_box.shape[1] // 4,
            4,
        ).mean(axis=(1, 3))
        refined_matches.loc[i, "LACEVarianceAfterBinning"] = np.var(binning_box)
