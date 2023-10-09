import typer
from pathlib import Path
from typing import List, Optional

from decolace.processing.project_managment import AcquisitionAreaPreProcessing

app = typer.Typer()

@app.command()
def create_project(
    name: str = typer.Argument(..., help="Name of project"),
    directory: Optional[Path] = typer.Option(
        None, help="Directory to create project in, current directory by default"
    ),
):
    """
    Create a new DeCoLACE processing project
    """
    from decolace.processing.project_managment import ProcessingProject
    if directory is None:
        directory = Path.cwd()
    project = ProcessingProject(project_name = name, project_path= directory)
    project.write()


@app.command()
def add_acquisition_area(
    ctx: typer.Context,
    area_name: str = typer.Argument(..., help="Name of the area"),
    session_name: str = typer.Argument(..., help="Name of session"),
    grid_name: str = typer.Argument(..., help="Name of grid"),
    area_directory: Path = typer.Argument(..., help="Path to area directory"),
):
    """
    Add a single acquisition area to a DeCoLACE processing project
    """
    from decolace.acquisition.acquisition_area import AcquisitionAreaSingle
    import numpy as np

    aa = AcquisitionAreaSingle(area_name, area_directory.as_posix())
    aa.load_from_disk()
    if np.sum(aa.state['positions_acquired']) == 0:
                print(f"{aa.name}: No Data")
    aa_pre = AcquisitionAreaPreProcessing(
                area_name = f"{session_name}_{grid_name}_{aa.name}",
                decolace_acquisition_area_info_path = sorted(Path(aa.directory).glob(f"{aa.name}*.npy"))[-1],
                decolace_grid_info_path = None,
                decolace_session_info_path = None,
                frames_folder = aa.frames_directory,
            )
    ctx.obj.project.acquisition_areas.append(aa_pre)
    ctx.obj.project.write()
    typer.echo(f"Added acquisition area {aa.name} ")


@app.command()
def add_session(
    ctx: typer.Context,
    session_name: str = typer.Argument(..., help="Name of session"),
    session_directory: Path = typer.Argument(..., help="Path to session directory"),
    ignore_grids: List[str] = typer.Option(["cross"], help="Grids to ignore"),
):
    """
    Add a DeCoLACE acquisition session to a DeCoLACE processing project
    """
    from decolace.acquisition.session import session
    import numpy as np
    
    my_session = session(session_name, session_directory.as_posix())
    session_path = sorted(Path(my_session.directory).glob(f"{my_session.name}*.npy"))[-1]
    my_session.load_from_disk()
    num_aa = 0
    num_grids = 0
    for grid in my_session.grids:
        if grid.name in ignore_grids:
            continue
        grid_path = sorted(Path(grid.directory).glob(f"{grid.name}*.npy"))[-1]
        num_grids += 1
        for aa in grid.acquisition_areas:
            if np.sum(aa.state['positions_acquired']) == 0:
                print(f"{grid.name} - {aa.name}: No Data")
                continue
            aa_pre = AcquisitionAreaPreProcessing(
                area_name = f"{my_session.name}_{grid.name}_{aa.name}",
                decolace_acquisition_area_info_path = sorted(Path(aa.directory).glob(f"{aa.name}*.npy"))[-1],
                decolace_grid_info_path = grid_path,
                decolace_session_info_path = session_path,
                frames_folder = aa.frames_directory,
            )
            ctx.obj.project.acquisition_areas.append(aa_pre)
            num_aa += 1
    typer.echo(f"Added {num_aa} acquisition areas from {num_grids} grids to {ctx.obj.project.project_name}")
    ctx.obj.project.write()




@app.command()
def status(
    ctx: typer.Context,
    project_main: Path = typer.Option(None, help="Path to wanted project file")
):
    from pycistem.database import get_num_movies, get_num_images, get_num_already_processed_images
    from rich.table import Table
    from rich import print
    from glob import glob

   
    #process_experimental_conditions(ctx.obj.acquisition_areas)

    table = Table(title="Preprocessing Status")
    table.add_column("AA id")
    table.add_column("AA name")
    table.add_column("cisTEM")
    table.add_column("Unblur")
    table.add_column("Ctffind")
    table.add_column("Montage")

    for aa in ctx.obj.acquisition_areas:
        table.add_row(
            aa.area_name,
            f"✓ {get_num_movies(aa.cistem_project)}" if aa.cistem_project is not None else ":x:",
            f"✓ {get_num_images(aa.cistem_project)}" if aa.unblur_run else ":x:",
            "✓" if aa.ctffind_run else ":x:",
            "✓" if aa.montage_image is not None else ":x:",
        )
    print(table)

    match_template_table = Table(title="Match Template Status")
    match_template_table.add_column("AA")
    
    for mtm in ctx.obj.project.match_template_runs:
        match_template_table.add_column(mtm.run_name)
        
    for aa in ctx.obj.acquisition_areas:
        mtm_status = []
        for mtm in ctx.obj.project.match_template_runs:        
            mtm_status.append(f"{get_num_already_processed_images(aa.cistem_project, mtm.run_id)}/{get_num_images(aa.cistem_project)}")
        
        match_template_table.add_row(
            aa.area_name, *mtm_status)
    
    print(match_template_table)