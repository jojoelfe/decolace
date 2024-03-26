from pathlib import Path
import mrcfile
from skimage import filters
from rich import print
import numpy as np

def subtract_linear_background_model(
        montage_path: Path,
        output_path: Path,
        n_knots: int = 5,
        knots: str = "uniform",
):
    from sklearn.linear_model import HuberRegressor
    import mrcfile
    import numpy as np
    from sklearn.preprocessing import SplineTransformer, StandardScaler
    from sklearn.pipeline import Pipeline
    from scipy.stats import describe
    import matplotlib.pyplot as plt
    with mrcfile.open(montage_path) as mrc:
        data = mrc.data
    with mrcfile.open(Path(montage_path).parent / "montage_mask.mrc") as mrc:
        mask = mrc.data
    # Create list of X,Y values wher mask > 0.99 using numpy
    # Create binned by 4 version of data and mask
    data_trimmed = data[:data.shape[0] - data.shape[0] % 8, :data.shape[1] - data.shape[1] % 8]
    mask_trimmed = mask[:mask.shape[0] - mask.shape[0] % 8, :mask.shape[1] - mask.shape[1] % 8]
    data_binned = data_trimmed.reshape(data_trimmed.shape[0] // 8, 8, data_trimmed.shape[1] // 8, 8).mean(axis=(1, 3))
    mask_binned = mask_trimmed.reshape(mask_trimmed.shape[0] // 8, 8, mask_trimmed.shape[1] // 8, 8).mean(axis=(1, 3))

    X = np.where(mask_binned > 0.99)
    Y = data_binned[X]

    model = Pipeline([('splitne', SplineTransformer(n_knots=n_knots,degree=3,knots=knots)),
                    ('scaler', StandardScaler()),
                    ('linear', HuberRegressor(alpha=0.0, epsilon=1.35))])

    #huber = HuberRegressor(alpha=0.0, epsilon=1.35)
    model.fit(np.array(X).T, Y)
   
    X = np.where(mask > -0.99)
    background = model.predict((np.array(X)/8).T).reshape(data.shape)   

    filtered_montage = data - background
    filtered_montage -= np.mean(filtered_montage[np.where(mask > 0.99)])
    filtered_montage /= 4 * np.std(filtered_montage[np.where(mask > 0.99)])
    #filtered_montage *= 0.4
    filtered_montage += 0.5
    plt.hist(filtered_montage[np.where(mask > 0.99)], bins=100)
    plt.savefig(output_path.parent / "histogram.png")
    plt.close()
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data((filtered_montage).astype(np.float32))



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