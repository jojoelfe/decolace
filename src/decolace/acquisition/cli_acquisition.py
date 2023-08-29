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
import signal
from types import SimpleNamespace
from .serialem_helper import set_sem_ip, set_sem_port, connect_sem


class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self, *args):
    print("Recieved SIGTERM")
    self.kill_now = True



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
def prepare_beam_vacuum(
    name: str = typer.Option(None, help="Name of the session"),
    directory: str = typer.Option(None , help="Directory to save session in"),
):
    session_o = load_session(name, directory)
    session_o.prepare_beam_vacuum()
    session_o.write_to_disk()
    typer.echo(f"Prepared beam vacuum for session {session_o.name}")

@app.command()
def prepare_beam_cross(
    name: str = typer.Option(None, help="Name of the session"),
    directory: str = typer.Option(None , help="Directory to save session in"),
):
    session_o = load_session(name, directory)
    session_o.prepare_beam_cross()
    session_o.write_to_disk()
    typer.echo(f"Prepared beam vacuum for session {session_o.name}")

@app.command()
def print_session_state(
    name: str = typer.Option(None, help="Name of the session"),
    directory: str = typer.Option(None , help="Directory to save session in"),
): 
    session_o = load_session(name, directory)
    print(session_o.state)

@app.command()
def set_session_state(
    name: str = typer.Option(None, help="Name of the session"),
    directory: str = typer.Option(None , help="Directory to save session in"),
    key: str = typer.Argument(...),
    value: float = typer.Argument(...),
): 
    session_o = load_session(name, directory)
    session_o.state.__dict__.update({key:value})
    session_o.write_to_disk()
    print(session_o.state)

@app.command()
def save_microscope_settings(
    name: str = typer.Option(None, help="Name of the session"),
    directory: str = typer.Option(None , help="Directory to save session in"),
):

    session_o = load_session(name, directory)
    session_o.add_current_setting()
    session_o.write_to_disk()
    typer.echo(f"Saved microscope settings for session {session_o.name}")

@app.command()
def print_session_state(
    name: str = typer.Option(None, help="Name of the session"),
    directory: str = typer.Option(None , help="Directory to save session in"),
):
    session_o = load_session(name, directory)
    typer.echo(session_o.state)

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
def nice_view(
    session_name: str = typer.Option(None, help="Name of the session"),
    directory: str = typer.Option(None, help="Directory to save session in"),
):
    session_o = load_session(session_name, directory)
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
    session_name: str = typer.Option(None, help="Name of the session"),
    directory: str = typer.Option(None, help="Directory to save session in"),
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
    show_camera: bool = typer.Option(False, help="SHow the camera")
):
    import napari

    if type == DeCoData.grid:
        session_o = load_session(name, directory)
        grid_o = session_o.active_grid
        write_to_disktext = {
                'string': '{order}',
                'size': 10,
                'color': 'white',
                'translation': np.array([0, 0]),
             }
        order = []
        positions = []
        corner_positions = []
        for i, aa in enumerate(grid_o.acquisition_areas):
            order.append(np.array(range(len(aa.state.acquisition_positions))))
            
            pos = aa.state.acquisition_positions[:, ::-1]
            # Concatenat i to pos along axis 1
            pos = np.concatenate((np.ones((len(pos), 1)) * i, pos), axis=1)
            positions.append(pos)
        
        pos = np.concatenate(positions, axis = 0)
        order = np.concatenate(order, axis = 0)
        
        
        viewer = napari.view_points(
            pos,
            name="exposures",
            size=session_o.state.beam_radius * 2,
            face_color="#00000000",
            features={"order":np.array(order)},
            text=write_to_disktext
        )
        #viewer.add_shapes(
        #    area_o.state["corner_positions_specimen"][:, ::-1],
        #    name="area",
        #    face_color="#00000000",
        #    edge_color="red",
        #    edge_width=0.1,
        #)
        if show_camera:
            viewer.add_points(
                pos,
                name="camera",
                size=0.61,
                face_color="#00000000",
                edge_color="#00ff00",
                symbol='square',
                edge_width=0.02
            )
        #viewer.add_shapes(
        #    np.array(corner_positions),
        #    name="area",
        #    face_color="#00000000",
        #)

    if type == DeCoData.area:
        area_o = load_area(name, directory)
        order = np.array(range(len(area_o.state.acquisition_positions)))
        write_to_disktext = {
            'string': '{order}',
            'size': 10,
            'color': 'white',
            'translation': np.array([0, 0]),
         }
        viewer = napari.view_points(
            area_o.state.acquisition_positions[:, ::-1],
            name="exposures",
            size=area_o.state.beam_radius * 2,
            face_color="#00000000",
            features={"order":order},
            text=write_to_disktext
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
def skip_area(
    name: str = typer.Argument(..., help="Name of the grid"),
    session_name: str = typer.Option(None, help="Name of the session"),
    directory: str = typer.Option(None, help="Directory the session is saved in"),
):
    session_o = load_session(session_name, directory)
    for aa in session_o.active_grid.acquisition_areas:
        if aa.name == name:
            aa.state.positions_acquired = np.zeros(
                (aa.state.acquisition_positions.shape[0]), dtype="bool"
            )
            aa.state.positions_acquired[:] = True
            typer.echo(f"Skipping {aa.name}")
            aa.write_to_disk()



@app.command()
def setup_lamellae(
    session_name: str = typer.Option(None, help="Name of the session"),
    directory: str = typer.Option(None, help="Directory to save session in"),
):
    session_o = load_session(session_name, directory)

    import napari
    from magicgui import magicgui, magic_factory
    from multiprocessing import Pool
    viewer = napari.Viewer()

    def get_nice_view(session_o):
        serialem = connect_sem()
        session_o.active_grid.nice_view()

    @magic_factory(
            call_button='Get Nice View',

    )
    def nice_view_button() -> napari.types.LayerDataTuple:
        pool = Pool(processes=1)
        image = pool.map(get_nice_view,[session_o])[0]
        existing_data = viewer.layers
        

@app.command()
def setup_areas(
    session_name: str = typer.Option(None, help="Name of the session"),
    directory: str = typer.Option(None, help="Directory to save session in"),
):
    serialem = connect_sem()
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
            name = f"area{len(session_o.active_grid.state.acquisition_areas)+2}"
            polygon = shapely.geometry.Polygon(area[:,1:3])
            aa = AcquisitionAreaSingle(name,Path(session_o.active_grid.directory,name).as_posix(),tilt=session_o.active_grid.state.tilt)   
            aa.initialize_from_napari(map_navids[int(map_id)], [polygon.centroid.y, polygon.centroid.x], area[:,1:3])
            aa.calculate_acquisition_positions_from_napari(beam_radius=session_o.state.beam_radius)
            aa.write_to_disk()
            session_o.active_grid.state.acquisition_areas.append([aa.name,aa.directory])
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
    defocus_offset: float = typer.Option(6.0)
):
    killer = GracefulKiller()
    session_o = load_session(session_name, directory)
    session_o.active_grid.state.stepwise = stepwise
    total_shots = sum(
        [
            len(aa.state.acquisition_positions)
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
                total=len(acquisition_area.state.acquisition_positions),
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
            log_string = f"{report['position']}/{len(acquisition_area.state.acquisition_positions)} "
            if "using_focus_prediction" in report:
                log_string += "FP :green_heart: "
            else:
                log_string += "FP :x: "
            if "using_beamshift_prediction" in report:
                log_string += "BSP :green_heart: "
            else:
                log_string += "BSP :x: "
            if "fasttracked" in report:
                log_string += "FT :green_heart: "
            else:
                log_string += "FT :x: "

            log_string += f"Counts {report['counts']:.1f} "

            if "beamshift_correction" in report:
                log_string += f"BSC: {report['beamshift_correction']:.4f}um "

            if "measured_defocus" in report:
                log_string += f"MF: {report['measured_defocus']:.2f}um {report['ctf_cc']:.3f}CC {report['ctf_res']:.2f}A "

            if "defocus_adjusted_by" in report:
                log_string += f"DA: {report['defocus_adjusted_by']:.3f}um"

            progress.log(log_string)
            if killer.kill_now:
                grid.state.stepwise = True
            if grid.state.stepwise:
                cont = typer.confirm("Continue?")
                if not cont:
                    save = typer.confirm("Save?")
                    if save:
                        acquisition_area.write_to_disk()
                    continous = typer.confirm("Continous:")
                    if continous:
                        grid.state.stepwise = False
                    else:
                        print("Aborting!")
                        raise typer.Abort()

    session_o.active_grid.start_acquisition(initial_defocus=session_o.state.fringe_free_focus_cross_grating-defocus_offset,progress_callback=progress_callback)

@app.callback()
def main(
    ctx: typer.Context,
    serialem_port: int = typer.Option(None, help="Serial port for SEM"),
    serialem_ip: str = typer.Option(None, help="IP address for SEM"),
):
    """DeCoLACE acquisition commands"""
    
    if serialem_port is not None:
        set_sem_port(serialem_port)
    if serialem_ip is not None:
        set_sem_ip(serialem_ip)
    ctx.obj = SimpleNamespace()

