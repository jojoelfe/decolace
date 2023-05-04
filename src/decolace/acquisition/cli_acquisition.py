import glob
import os
from enum import Enum
from pathlib import Path

import typer
from rich.panel import Panel
from rich.progress import Progress
import numpy as np
import shapely
from .acquisition_area import AcquisitionAreaSingle
from .session import session

try:
    import serialem
except ModuleNotFoundError:
    print("Couldn't import serialem")

app = typer.Typer()


def load_session(name, directory):
    if directory is None:
        directory = Path(os.getcwd()).as_posix()
    else:
        directory = Path(directory).as_posix()
    if name is None:
        potential_files = glob.glob(os.path.join(directory, "*.npy"))
        if len(potential_files) < 1:
            raise (FileNotFoundError("Couldn't find saved files"))
        most_recent = sorted(potential_files)[-1]
        name = os.path.basename(most_recent).split("_")[0]
    session_o = session(name, directory)
    session_o.load_from_disk()
    return session_o


def load_area(name, directory):
    if directory is None:
        directory = Path(os.getcwd()).as_posix()
    else:
        directory = Path(directory).as_posix()
    if name is None:
        potential_files = glob.glob(os.path.join(directory, "*.npy"))
        if len(potential_files) < 1:
            raise (FileNotFoundError("Couldn't find saved files"))
        most_recent = sorted(potential_files)[-1]
        name = os.path.basename(most_recent).split("_")[0]
    area_o = AcquisitionAreaSingle(name, directory)
    area_o.load_from_disk()
    return area_o


@app.command()
def new_session(
    name: str = typer.Argument(..., help="Name of the session"),
    directory: str = typer.Option(None, help="Directory to save session in"),
):
    if directory is None:
        directory = os.getcwd()
    else:
        directory = Path(directory).as_posix()
    session_o = session(name, directory)
    session_o.write_to_disk()
    typer.echo(f"Created new session {name} in {directory}")


@app.command()
def save_microscope_settings(
    name: str = typer.Option(..., help="Name of the session"),
    directory: str = typer.Option(..., help="Directory to save session in"),
):

    session_o = load_session(name, directory)
    session_o.add_current_setting()
    session_o.write_to_disk()
    typer.echo(f"Saved microscope settings for session {session_o.name}")

@app.command()
def set_beam_radius(
    beam_radius: float = typer.Argument(..., help="Beam radius in um"),
    name: str = typer.Option(..., help="Name of the session"),
    directory: str = typer.Option(..., help="Directory to save session in"),
):
    session_o = load_session(name, directory)
    session_o.state["beam_radius"] = beam_radius
    session_o.write_to_disk()
    typer.echo(f"Set beam radius to {beam_radius} for session {session_o.name}")

@app.command()
def new_grid(
    name: str = typer.Argument(..., help="Name of the grid"),
    tilt: float = typer.Argument(..., help="Tilt to apply to the the grid"),
    session_name: str = typer.Option(None, help="Name of the session"),
    directory: str = typer.Option(None, help="Directory the session is saved in"),
):
    session_o = load_session(session_name, directory)
    session_o.add_grid(name, tilt=tilt)
    session_o.write_to_disk()
    typer.echo(f"Created new grid {name} for session {session_o.name}")


@app.command()
def euc_and_nice_view(
    session_name: str = typer.Option(None, help="Name of the session"),
    directory: str = typer.Option(None, help="Directory to save session in"),
):
    session_o = load_session(session_name, directory)
    session_o.active_grid.eucentric()
    session_o.active_grid.nice_view()
    session_o.active_grid.write_to_disk()


@app.command()
def status(
    session_name: str = typer.Option(None, help="Name of the session"),
    directory: str = typer.Option(None, help="Directory to save session in"),
):
    session_o = load_session(session_name, directory)
    typer.echo(f"Session: {session_o.name} contains {len(session_o.grids)} grids")
    typer.echo(f"Active grid: {session_o.active_grid.name}")
    for grid in session_o.grids:
        typer.echo(
            f"Grid: {grid.name} contains {len(grid.acquisition_areas)} acquisition areas"
        )


@app.command()
def new_map(
    session_name: str = typer.Option(..., help="Name of the session"),
    directory: str = typer.Option(..., help="Directory to save session in"),
):
    session_o = load_session(session_name, directory)
    session_o.active_grid.take_map()
    session_o.write_to_disk()
    typer.echo(f"Created new map for grid {session_o.active_grid.name}")


@app.command()
def add_acquisition_area():
    typer.echo("Added acquisition area")


class DeCoData(str, Enum):
    session = "session"
    grid = "grid"
    area = "area"


@app.command()
def show_exposures(
    type: DeCoData = typer.Argument(DeCoData.session),
    name: str = typer.Argument(None, help="Name of the session"),
    directory: str = typer.Argument(None, help="Directory to save session in"),
):
    import napari

    if type == DeCoData.area:
        area_o = load_area(name, directory)
        order = np.array(range(len(area_o.state["acquisition_positions"])))
        write_to_disktext = {
            'string': '{order}',
            'size': 10,
            'color': 'white',
            'translation': np.array([0, 0]),
         }
        viewer = napari.view_points(
            area_o.state["acquisition_positions"][:, ::-1],
            name="exposures",
            size=area_o.state["beam_radius"] * 2,
            face_color="#00000000",
            features={"order":order},
            text=write_to_disktext
        )
        viewer.add_shapes(
            area_o.state["corner_positions_specimen"][:, ::-1],
            name="area",
            face_color="#00000000",
            edge_color="red",
            edge_width=0.1,
        )
        napari.run()
        typer.Exit()

@app.command()
def set_active_grid(
    name: str = typer.Argument(..., help="Name of the grid"),
    session_name: str = typer.Option(None, help="Name of the session"),
    directory: str = typer.Option(None, help="Directory the session is saved in"),
):
    session_o = load_session(session_name, directory)
    session_o.set_active_grid(name)
    session_o.write_to_disk()
    typer.echo(f"Set active grid to {name} for session {session_o.name}")


@app.command()
def setup_areas(
    session_name: str = typer.Option(None, help="Name of the session"),
    directory: str = typer.Option(None, help="Directory to save session in"),
):
    session_o = load_session(session_name, directory)
    
    num_items = serialem.ReportNumTableItems()
    maps = []
    map_coordinates = []
    map_navids = []
    for i in range(1,int(num_items)+1):
        nav_item_info = serialem.ReportOtherItem(i)
        nav_id = int(nav_item_info[0])
        nav_note = serialem.GetVariable("navNote")
        
        if nav_note == "decolace_acquisition_map":
            serialem.LoadOtherMap(i,"A")
            image = np.asarray(serialem.bufferImage("A")).copy()
            maps.append(image)
            map_navids.append(nav_id)
            map_coordinates.append(nav_item_info[1:3])
    import napari
    from napari.layers import Shapes
    from magicgui import magicgui
    viewer = napari.view_image(np.array(maps))
    
    @magicgui(shapes={'label': 'Setup areas'})
    def my_widget(shapes: Shapes):
        points = []
        areas = shapes.data
        for area in areas:
            map_id = area[0,0]
            if np.sum(area[:,0] - map_id) != 0:
                raise("Error: Map ID is not the same for all points in the polygon")
            name = f"area{map_id}"
            polygon = shapely.geometry.Polygon(area[:,1:3])
            aa = AcquisitionAreaSingle(name,Path(session_o.active_grid.directory,name).as_posix(),beam_radius=session_o.state["beam_radius"],tilt=session_o.active_grid.state["tilt"])   
            aa.initialize_from_napari(map_navids[int(map_id)], [polygon.centroid.y, polygon.centroid.x], area[:,1:3])
            aa.calculate_acquisition_positions_from_napari()
            aa.write_to_disk()
            session_o.active_grid.state["acquisition_areas"].append([aa.name,aa.directory])
        session_o.active_grid.write_to_disk()
        session_o.active_grid.save_navigator()
             
        
        
    viewer.window.add_dock_widget(my_widget)
    napari.run()
    print("Done")
    typer.Exit()



@app.command()
def acquire(
    session_name: str = typer.Option(None, help="Name of the session"),
    directory: str = typer.Option(None, help="Directory to save session in"),
    stepwise: bool = typer.Option(False, help="Acquire stepwise"),
):
    session_o = load_session(session_name, directory)

    total_shots = sum(
        [
            len(aa.state["acquisition_positions"])
            for aa in session_o.active_grid.acquisition_areas
        ]
    )

    progress = Progress(auto_refresh=False)
    grid_task = progress.add_task("grid", total=total_shots)
    aa_task = progress.add_task("area")

    progress.console.print(
        Panel(
            "[bold blue]Starting acquisition of grid {session_o.active_grid.name}.",
            padding=1,
        )
    )

    def progress_callback(grid, report, acquisition_area, type=None):
        if type == "start_new_area":
            progress.log(
                f"Starting acquisition of area {acquisition_area.name} in grid {grid.name}"
            )
            progress.reset(
                aa_task,
                total=len(acquisition_area.state["acquisition_positions"]),
                description=f"{acquisition_area.name}",
            )
            progress.start_task(aa_task)
        elif type == "resume_area":
            progress.log(
                f"Resuming acquisition of area {acquisition_area.name} in grid {grid.name}"
            )
        if report is not None:
            progress.update(aa_task, advance=1)
            progress.update(grid_task, advance=1)
            log_string = f"{report['position']}/{len(acquisition_area.state['acquisition_positions'])} "
            if "using_focus_prediction" in report:
                log_string += "FP :check_mark: "
            else:
                log_string += "FP :x: "
            if "using_beamshift_prediction" in report:
                log_string += "BSP :check_mark: "
            else:
                log_string += "BSP :x: "
            if "fasttracked" in report:
                log_string += "FT :check_mark: "
            else:
                log_string += "FT :x: "

            log_string += f"Counts {report['counts']} "

            if "beamshift_correction" in report:
                log_string += f"BSC: {report['beamshift_correction']:.4f}um "

            if "measured_defocus" in report:
                log_string += f"MF: {report['measured_defocus']:.2f}um {report['ctf_cc']:.2f}CC {report['ctf_cc']:.2f}A"

            if "defocus_adjusted_by" in report:
                log_string += f"DA: {report['defocus_adjusted_by']:.2f}um"

            progress.log(log_string)
            if stepwise:
                cont = typer.confirm("Continue?")
                if not cont:
                    save = typer.confirm("Save?")
                    if save:
                        acquisition_area.write_to_disk()
                    print("Aborting!")
                    raise typer.Abort()

    session_o.active_grid.acquire(progress_callback=progress_callback)
