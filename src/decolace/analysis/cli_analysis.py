from pathlib import Path
from typing import List
from types import SimpleNamespace
import typer
import glob
from rich import print
from rich.logging import RichHandler
from typer.core import TyperGroup

from decolace.processing.project_managment import ProcessingProject

from decolace.processing.cli_project_managment import app as project_managment_app
from decolace.processing.cli_preprocessing import app as preprocessing_app
from decolace.processing.cli_montaging import app as montaging_app

class OrderCommands(TyperGroup):
  def list_commands(self, ctx: typer.Context):
    """Return list of commands in the order appear."""
    return list(self.commands)    # get commands using self.commands

app = typer.Typer(
    cls=OrderCommands,
)

@app.command()
def visualize():
    """Visualize an acquisition_area"""
    pass

@app.command()
def edit_conditions(
    ctx: typer.Context,
):
    """Edit conditions of an acquisition_area"""
    import pandas as pd
    import pandasgui

    data = pd.DataFrame(
                [aa.dict() for aa in ctx.obj.acquisition_areas]
            )
    pandasgui.show(data)
    print(data["experimental_condition"])

@app.command()
def cluster_by_distance_and_orientation(
    ctx: typer.Context,
    distance_threshold: float = typer.Option(1000, help="Distance threshold for clustering"),
    orientation_threshold: float = typer.Option(20, help="Orientation threshold for clustering in degrees"),
    orientation_axis: str = typer.Option("1,0,0", help="Axis for orientation clustering"),
):
    import starfile
    import numpy as np
    from scipy.spatial import KDTree
    import eulerangles

    input_dir = Path(ctx.obj.project.project_path) / "Matches"
    for aa in ctx.obj.acquisition_areas:
        star_path = input_dir / f"{aa.area_name}_{ctx.obj.match_template_job.run_name}_{ctx.obj.match_template_job.run_id}_filtered.star"
        if not star_path.exists():
            continue
        print(aa.area_name)
        data = starfile.read(star_path)

        coordinates = np.stack((data["cisTEMOriginalXPosition"].to_numpy(), data["cisTEMOriginalYPosition"].to_numpy(), (data["cisTEMDefocus1"].to_numpy()+ data["cisTEMDefocus2"].to_numpy()) / 2 ),axis=1)
        euler_angles = np.stack((data["cisTEMAnglePhi"].to_numpy(), data["cisTEMAngleTheta"].to_numpy(), data["cisTEMAnglePsi"].to_numpy()),axis=1)
        # Get roration matrices
        rotation_matrices = eulerangles.invert_rotation_matrices(eulerangles.euler2matrix(euler_angles, axes='zyz',
                             intrinsic=True,
                             right_handed_rotation=True))
        
        tree = KDTree(coordinates)
        pairs = tree.query_pairs(r=distance_threshold, output_type='ndarray')


        data['CLUSTER_SCORE'] = 0
        print(f"Distance {distance_threshold} has {len(pairs)} pairs")
        # Get all pairs that ar aligned correctly
        vectors = np.dot(rotation_matrices, np.array([0,0,1]))
        #vectors1 = np.dot(rotation_matrices[pairs[:,0]], np.array([1,0,0]))
        #vectors2 = np.dot(rotation_matrices[pairs[:,1]], np.array([1,0,0]))
        dotproducts = np.sum(vectors[pairs[:,0]] * vectors[pairs[:,1]], axis=1) 
        angles = np.degrees(np.arccos(dotproducts)) # No normalization needed as long as rotated vector is a unit vector
        potential_clusters = pairs[angles < orientation_threshold]
        print(f"There are {len(potential_clusters)} pairs with aligned orientation")
        potential_clusters_midpoints = (coordinates[potential_clusters[:,0]] + coordinates[potential_clusters[:,1]]) / 2
        potential_clusters_normal = vectors[potential_clusters[:,0]] + vectors[potential_clusters[:,1]]
        potential_clusters_normal /= np.linalg.norm(potential_clusters_normal, axis=1)[:,None]
        potential_additional_candidates = tree.query_ball_point(potential_clusters_midpoints,r=distance_threshold*2)
        
        valid = 0
        for i,potential_cluster in enumerate(potential_additional_candidates):
            potential_cluster = np.array(potential_cluster)
            dotproducts = np.sum(potential_clusters_normal[i] * vectors[potential_cluster], axis=1)
            angles = np.degrees(np.arccos(dotproducts))
            potential_cluster = potential_cluster[angles < orientation_threshold]
            
            distance_to_plane = np.abs(np.sum((coordinates[potential_cluster]-potential_clusters_midpoints[i]) * potential_clusters_normal[i], axis=1))
            potential_cluster = potential_cluster[distance_to_plane < 400]
            if len(potential_cluster) > 3:
                
                data.iloc[potential_cluster, data.columns.get_loc('CLUSTER_SCORE')] += len(potential_cluster)
        print(data['CLUSTER_SCORE'].describe())
        return
        output_path = input_dir / f"{aa.area_name}_{ctx.obj.match_template_job.run_name}_{ctx.obj.match_template_job.run_id}_filtered_numclusters.star"
        starfile.write(data, output_path, overwrite=True)

@app.command()
def plot_distribution_of_pairwise_orientation(
    ctx: typer.Context,
):
    import starfile
    import numpy as np
    from scipy.spatial import KDTree
    from healpy import pix2ang

    from decolace.analysis.orientations import calculate_quaternion_rotation
    input_dir = Path(ctx.obj.project.project_path) / "Matches"
    distance_shells = [
        (0, 300),
        (300, 500),
        (500, 700),
        (700, 900),
        (900, 1100),
        (1100, 1300),
    ]
    rotations = []
    for i, distance in enumerate(distance_shells):
        rotations.append([])
    
    
    for aa in ctx.obj.acquisition_areas:

        star_path = input_dir / f"{aa.area_name}_{ctx.obj.match_template_job.run_name}_{ctx.obj.match_template_job.run_id}_filtered.star"
        if not star_path.exists():
            continue
        print(aa.area_name)
        data = starfile.read(star_path)
        coordinates = np.stack((data["cisTEMOriginalXPosition"].to_numpy(), data["cisTEMOriginalYPosition"].to_numpy()),axis=1)
        euler_angles = np.stack((data["cisTEMAnglePhi"].to_numpy(), data["cisTEMAngleTheta"].to_numpy(), data["cisTEMAnglePsi"].to_numpy()),axis=1)
        
        tree = KDTree(coordinates)
        pairs = tree.query_pairs(r=distance_shells[-1][1])
            
        print(f"Distance {distance} has {len(pairs)} pairs")
        for pair in pairs:
            distance = np.linalg.norm(coordinates[pair[0]] - coordinates[pair[1]])
            shell_i = None
            shell = None
            for i,distancep in enumerate(distance_shells):
                if distancep[0] < distance < distancep[1]:
                    shell_i = i
                    shell = distancep
                    break
            if shell_i is None or shell is None:
                print(f"what {distance}")
                continue
            rotation = calculate_quaternion_rotation(euler_angles[pair[0]], euler_angles[pair[1]])
            axis = rotation.axis
            if axis[2] < 0:
                axis = -axis
            # If angle between x axis and axivec is smaller than 10deg
            #if np.arccos(np.dot(axis, np.array([1,0,0]))) < np.deg2rad(20) or np.arccos(np.dot(-axis, np.array([1,0,0]))) < np.deg2rad(20):
            #    rotations[shell_i].append(axis)
    
    for i, axis_vecs in enumerate(rotations):
        heightscale=0.3
        widthscale=0.25
        nside = 2**3
        angular_sampling = np.sqrt(3 / np.pi) * 60 / nside
        theta, phi = pix2ang(nside, np.arange(12 * nside ** 2))
        phi = np.pi - phi
        hp = np.column_stack((np.sin(theta) * np.cos(phi),
                                np.sin(theta) * np.sin(phi),
                                np.cos(theta)))
        kdtree = KDTree(hp)
        #st = np.sin(np.deg2rad(df[star.Relion.ANGLETILT]))
        #ct = np.cos(np.deg2rad(df[star.Relion.ANGLETILT]))
        #sp = np.sin(np.deg2rad(df[star.Relion.ANGLEROT]))
        #cp = np.cos(np.deg2rad(df[star.Relion.ANGLEROT]))
        #ptcls = np.column_stack((st * cp, st * sp, ct)) I need to create ptcls
        ptcls = np.array(axis_vecs)
        _, idx = kdtree.query(ptcls)
        cnts = np.bincount(idx, minlength=theta.size)
        frac = cnts / np.max(cnts).astype(np.float64)
        mu = np.mean(frac)
        sigma = np.std(frac)
        print("%.0f (%.2f%%) +/- %.1f (%.2f%%) particles per bin" %
            (np.mean(cnts), mu, np.std(cnts), sigma))
        color_scale = (frac - mu) / sigma
        color_scale[color_scale > 5] = 5
        color_scale[color_scale < -1] = -1
        color_scale /= 6
        color_scale += 1 / 6.
        imin = np.argmin(cnts)
        imax = np.argmax(cnts)
        print("Min %d particles (%.2f%%); color %f,0,%f" %
            (cnts[imin], frac[imin] * 100, color_scale[imin], 1 - color_scale[imin]))
        print("Max %d particles (%.2f%%); color %f,0,%f" %
            (cnts[imax], frac[imax] * 100, color_scale[imax], 1 - color_scale[imax]))
        r = 250
        rp = np.reshape(r + r * frac * heightscale, (-1, 1))
        base1 = hp * r
        base2 = hp * rp
        base1 = base1[:, [0, 1, 2]] + np.array([r]*3)
        base2 = base2[:, [0, 1, 2]] + np.array([r]*3)
        height = np.squeeze(np.abs(rp - r))
        idx = np.where(height >= 0.01)[0]
        width = widthscale * np.pi * r * angular_sampling / 360
        bild = np.hstack((base1, base2, np.ones((base1.shape[0], 1)) * width))
        fmt_color = ".color %f 0 %f\n"
        fmt_cyl = ".cylinder %f %f %f %f %f %f %f\n"
        with open(f"/tmp/testc{distance_shells[i][0]}_{distance_shells[i][1]}.bild", "w") as f:
            for i in idx:
                f.write(fmt_color % (color_scale[i], 1 - color_scale[i]))
                f.write(fmt_cyl % tuple(bild[i]))
    
            




@app.callback()
def main(
    ctx: typer.Context,
    project: Path = typer.Option(None, help="Path to wanted project file",rich_help_panel="Expert Options"),
    acquisition_area_ids: List[int] = typer.Option(None, help="List of acquisition areas ids to process"),
    acquisition_area_names: List[str] = typer.Option(None, help="List of acquisition areas names to process"),
    match_template_job_id: int = typer.Option(None, help="ID of template match job"),
    select_condition: str = typer.Option(None, help="Condition to select acquisition areas"),
):
    """DeCoLACE analysis functions"""
    from decolace.processing.project_managment import process_experimental_conditions
    if project is None:
        potential_projects = glob.glob("*.decolace")
        if len(potential_projects) == 0:
            typer.echo("No project file found")
            raise typer.Exit()
        project = Path(potential_projects[0])
    project_object = ProcessingProject.read(project)
    aas_to_process = project_object.acquisition_areas
    if len(acquisition_area_names) > 0:
        aas_to_process = [aa for aa in project_object.acquisition_areas if aa.area_name in acquisition_area_names]
    if len(acquisition_area_ids) > 0:
        aas_to_process = [aa for i, aa in enumerate(project_object.acquisition_areas) if i in acquisition_area_ids]
    if select_condition is not None:
        conditions = process_experimental_conditions(aas_to_process)
        key, value = select_condition.split("=")
        aas_to_process = [aa for i, aa in enumerate(aas_to_process) if key in conditions and i in conditions[key] and conditions[key][i] == value]
    if match_template_job_id is not None:
        array_position = [mtr.run_id  for mtr in project_object.match_template_runs].index(match_template_job_id)
        mtr = project_object.match_template_runs[array_position]
    else:
        mtr = None
    ctx.obj = SimpleNamespace(project = project_object, acquisition_areas = aas_to_process, match_template_job = mtr)
