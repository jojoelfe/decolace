from pathlib import Path
import mrcfile
from skimage import filters
from rich import print
import numpy as np

def highpassfilter_montage(
        montage_path: Path,
        output_path: Path,
        filter_cutoff_frequency_ratio: float = 0.001,
        filter_order: float = 4.0, 
        ):
    with mrcfile.open(montage_path) as mrc:
        montage = mrc.data
        if montage.ndim == 3:
            montage = montage[0]
    
    filtered_montage = filters.butterworth(
                montage,
                cutoff_frequency_ratio=filter_cutoff_frequency_ratio,
                order=filter_order,
                high_pass=True,
            )
    filtered_montage -= np.median(filtered_montage)
    filtered_montage /= np.quantile(filtered_montage, 0.95)
    filtered_montage += 0.5
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(filtered_montage)

    print(f"Wrote filtered montage to {output_path}")
    return