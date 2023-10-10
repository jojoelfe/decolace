import glob
import os
from enum import Enum
from pathlib import Path

import typer
from rich.panel import Panel
from rich.progress import Progress
import numpy as np
import shapely
import signal

class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self, *args):
    print("Recieved SIGTERM")
    self.kill_now = True



app = typer.Typer()


def get_image_fn(dummy):
    import serialem as sem
    sem.ConnectToSEM(48888,"192.168.122.149")
    image = np.asarray(sem.bufferImage("P")).copy()
    return image


@app.command()
def get_image():
    import napari
    from napari.layers import Image
    from magicgui import magicgui
    from multiprocessing import Pool

    @magicgui(call_button="Get image")
    def my_widget() -> Image:
        pool = Pool(processes = 1)
        image = pool.map(get_image_fn,[0])[0]
        return Image(image)
    
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(my_widget)
    napari.run()
    print("Done")

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
            name = f"area{len(session_o.active_grid.state['acquisition_areas'])+2}"
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


