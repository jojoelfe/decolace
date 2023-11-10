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
    beam_image_shift_threshold: float = 1.0,
    optical_group_output: Path = "/dev/null"
):
    import starfile
    import pandas as pd
    import sqlite3
    import numpy as np
    from pycistem.programs import refine_ctf, estimate_beamtilt

    particle_info = starfile.read(tmpackage_star_file)
    print(len(particle_info))

    aa_fd = pd.DataFrame([aa.model_dump() for aa in ctx.obj.acquisition_areas])
    

    aa_fd["image_folder"] = aa_fd["cistem_project"].map(lambda x: str(Path(x).parent / "Assets" / "Images" ))
    aa_fd["session"] = "all"
    if split_by_session:
        aa_fd["session"] = aa_fd["frames_folder"].map(lambda x: str(Path(x).parent.parent.parent.name))
    
    particle_info["image_folder"] = particle_info["cisTEMOriginalImageFilename"].str.strip("'").map(lambda x: str(Path(x).parent))

    
    particle_info["cisTEMOriginalImageFilename"] = particle_info["cisTEMOriginalImageFilename"].str.strip("'")
    particle_info = particle_info.join(aa_fd.set_index("image_folder"), on="image_folder", rsuffix="_aa")
    for i, cistem_project in enumerate(particle_info["cistem_project"].unique()):
        db = sqlite3.connect(cistem_project)
        microgaph_info = pd.read_sql_query("SELECT MOVIE_ALIGNMENT_LIST.OUTPUT_FILE, IMAGE_SHIFT_X, IMAGE_SHIFT_Y FROM MOVIE_ALIGNMENT_LIST INNER JOIN MOVIE_ASSETS_METADATA ON MOVIE_ALIGNMENT_LIST.MOVIE_ASSET_ID=MOVIE_ASSETS_METADATA.MOVIE_ASSET_ID", db)
        microgaph_info = microgaph_info.drop_duplicates(subset="OUTPUT_FILE")
        if i == 0:
            particle_info = particle_info.merge(microgaph_info, right_on="OUTPUT_FILE", left_on="cisTEMOriginalImageFilename",suffixes=(None,None),how="left")
            particle_info.set_index("cisTEMOriginalImageFilename", inplace=True)
        else:
            particle_info.update(microgaph_info.set_index("OUTPUT_FILE"))
  
    particle_info["image_shift_label"] = particle_info.apply(lambda x: f"{x['session']}_{x['IMAGE_SHIFT_X'] // beam_image_shift_threshold}_{x['IMAGE_SHIFT_Y'] // beam_image_shift_threshold}", axis=1)
    particle_info.reset_index(inplace=True)
    paramter_file_params = starfile.read(parameters_star_file)

    updated_params = []
    results = {}
    #print(sum(particle_info["image_shift_label"].value_counts()>1000))
    #return
    for subgroup in particle_info["image_shift_label"].value_counts().index:
        if particle_info["image_shift_label"].value_counts()[subgroup] < 1000:
            continue

    
    
        parameter_file_to_write = paramter_file_params[particle_info["image_shift_label"] == subgroup].copy()

        parameters_star_file_out = parameters_star_file.parent / f"{parameters_star_file.stem}_{subgroup}.star"
        print(f"Writing {parameters_star_file_out}")
        starfile.write(parameter_file_to_write, parameters_star_file_out, overwrite=True)
        refine_ctf_parameters = refine_ctf.RefineCtfParameters(
            input_particle_images="/scratch/erice/elferich/processing_thp1_ribosomes_unbinned/Assets/ParticleStacks/particle_stack_2.mrc",
            input_star_filename=str(parameters_star_file_out),
            input_reconstruction="/groups/elferich/pycistem/examples/volume_9_1_masked.mrc",
            ouput_phase_difference_image=str(parameters_star_file.parent / f"{parameters_star_file.stem}_{subgroup}_phase_diff.mrc"),
            molecular_mass_kDa=3000.0,
            outer_mask_radius=200.0,
            beamtilt_refinement=True,
        )

        refine_ctf.run(refine_ctf_parameters,num_threads=10)

        pbs = []
        for i in range(501):
            start_position = i*(290880//500)
            end_position = (i+1)*(290880//500)
            if end_position > 290880:
                end_position = 290880
            pbs.append(estimate_beamtilt.EstimateBeamtiltParameters(
                input_phase_difference_image =refine_ctf_parameters.ouput_phase_difference_image,
                pixel_size = 1.0,
                voltage_kV = 300.0,
                spherical_aberration_mm = 2.7,
                first_position_to_search = start_position,
                last_position_to_search = end_position
            ))

        result = estimate_beamtilt.run(pbs,num_procs=num_cores,cmd_prefix=cmd_prefix)

        best_result = result[result["score"] == result["score"].min()].iloc[0]
        results[subgroup] = best_result
        np.save(parameters_star_file.parent / f"{parameters_star_file.stem}_results.npy",[results], allow_pickle=True)
        print(best_result)
        parameter_file_to_write["cisTEMBeamTiltX"] = best_result["beam_tilt_x"] * 1000.0
        parameter_file_to_write["cisTEMBeamTiltY"] = best_result["beam_tilt_y"] * 1000.0
        parameter_file_to_write["cisTEMImageShiftX"] = best_result["particle_shift_x"]
        parameter_file_to_write["cisTEMImageShiftY"] = best_result["particle_shift_y"]

        updated_params.append(parameter_file_to_write)
    
    updated_params = pd.concat(updated_params)
    parameters_star_file_out = parameters_star_file.parent / f"{parameters_star_file.stem}_corrected.star"
    starfile.write(updated_params, parameters_star_file_out, overwrite=True)

    

@app.command()
def estimate_beamtilt_in_optical_groups(
    ctx: typer.Context,
    tmpackage_star_file: Path,
    parameters_star_file: Path,
    cmd_prefix: str = typer.Option("", help="Prefix of run command"),
    split_by_session: bool = True,
    beam_image_shift_threshold: float = 1.0,
    num_cores: int = typer.Option(10, help="Number of cores to use"),

):        
    pass

@app.command()
def fit_beamtilt_model_optical_groups():
    pass

@app.command()
def apply_beamtilt_model_optical_groups():
    pass
