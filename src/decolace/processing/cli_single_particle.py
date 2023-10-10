import typer
import logging
from rich.logging import RichHandler
from rich import print
from pathlib import Path


app = typer.Typer()

@app.command()
def split_particles_into_optical_groups(
    ctx: typer.Context,
    tmpackage_star_file: Path,
    parameters_star_file: Path,
    split_by_session: bool = True,
    beam_image_shift_threshold: float = 1.0
):
    import starfile
    import pandas as pd
    import sqlite3
    from sklearn.cluster import KMeans

    particle_info = starfile.read(tmpackage_star_file)

    aa_fd = pd.DataFrame([aa.model_dump() for aa in ctx.obj.acquisition_areas])
    

    aa_fd["image_folder"] = aa_fd["cistem_project"].map(lambda x: str(Path(x).parent / "Assets" / "Images" ))
    aa_fd["session"] = aa_fd["frames_folder"].map(lambda x: str(Path(x).parent.parent.parent.name))
    particle_info["image_folder"] = particle_info["cisTEMOriginalImageFilename"].str.strip("'").map(lambda x: str(Path(x).parent))
    
    particle_info["cisTEMOriginalImageFilename"] = particle_info["cisTEMOriginalImageFilename"].str.strip("'")
    particle_info = particle_info.join(aa_fd.set_index("image_folder"), on="image_folder", rsuffix="_aa")
    for i, cistem_project in enumerate(particle_info["cistem_project"].unique()):
        db = sqlite3.connect(cistem_project)
        microgaph_info = pd.read_sql_query("SELECT IMAGE_ASSETS.FILENAME, IMAGE_SHIFT_X, IMAGE_SHIFT_Y FROM IMAGE_ASSETS INNER JOIN MOVIE_ASSETS ON IMAGE_ASSETS.PARENT_MOVIE_ID=MOVIE_ASSETS.MOVIE_ASSET_ID INNER JOIN MOVIE_ASSETS_METADATA ON MOVIE_ASSETS.MOVIE_ASSET_ID=MOVIE_ASSETS_METADATA.MOVIE_ASSET_ID", db)
        if i == 0:
            particle_info = particle_info.merge(microgaph_info, right_on="FILENAME", left_on="cisTEMOriginalImageFilename",suffixes=(None,None),how="left")
            particle_info.set_index("cisTEMOriginalImageFilename", inplace=True)
        else:
            particle_info.update(microgaph_info.set_index("FILENAME"))
  
    particle_info["image_shift_label"] = particle_info.apply(lambda x: f"{x['session']}_{x['IMAGE_SHIFT_X'] // 1}_{x['IMAGE_SHIFT_Y'] // 1}", axis=1)
    particle_info.reset_index(inplace=True)
    print(particle_info["image_shift_label"].value_counts())
    
        

