import typer
import logging
from rich.logging import RichHandler
from pathlib import Path
from typing import Optional


app = typer.Typer()

@app.command()
def generate_cistem_projects(
    ctx: typer.Context,
    pixel_size: float = typer.Option(...,help="Movie pixelsize"),
    exposure_dose: float = typer.Option(...,help="Dose in e/A/frame")
):
    """
    Generate cisTEM projects for each acquisition area
    """
    from decolace.processing.create_cistem_projects_for_session import create_project_for_area 

    for aa in ctx.obj.acquisition_areas:
        if aa.cistem_project is not None:
            continue
        typer.echo(f"Creating cistem project for {aa.area_name}")
        cistem_project_path = create_project_for_area(aa.area_name, ctx.obj.project.project_path.absolute() / "cistem_projects", aa.frames_folder, pixel_size=pixel_size, exposure_dose=exposure_dose, bin_to=ctx.obj.project.processing_pixel_size)
        aa.cistem_project = cistem_project_path
        typer.echo(f"Saved as {cistem_project_path}")
    ctx.obj.project.write()
   
@app.command()
def run_unblur(
    ctx: typer.Context,
    num_cores: int = typer.Option(10, help="Number of cores to use"),
    cmd_prefix: str = typer.Option("", help="Prefix of run command"),
    cmd_suffix: str = typer.Option("", help="Suffix of run command"),
    run_additional: Optional[str] = typer.Option(None, help="Run additional unblur runs"),
    custom_binning_factor: Optional[float] = typer.Option(None, help="Custom binning factor for unblur")
):
    """
    Run unblur for each acquisition area
    """
    from pycistem.programs import unblur
    import pycistem
    pycistem.set_cistem_path(ctx.obj.cistem_path)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[
            RichHandler(),
            #logging.FileHandler(current_output_directory / "log.log")
        ]
    )
    
    for aa in ctx.obj.acquisition_areas:
        if aa.unblur_run and run_additional is None:
            continue
        typer.echo(f"Running unblur for {aa.area_name}")
        pars = unblur.parameters_from_database(aa.cistem_project,decolace=True)
        if run_additional is not None:
            for par in pars:
                par.output_filename = par.output_filename.replace("_auto_" , f"_auto_{run_additional}_")
                par.amplitude_spectrum_filename = par.amplitude_spectrum_filename.replace("_auto.mrc" , f"_auto_{run_additional}.mrc")
                par.small_sum_image_filename = par.small_sum_image_filename.replace("_auto.mrc" , f"_auto_{run_additional}.mrc")
        if custom_binning_factor is not None:
            for par in pars:
                par.output_binning_factor = custom_binning_factor
        

        res = unblur.run(pars,num_procs=num_cores,cmd_prefix=cmd_prefix,cmd_suffix=cmd_suffix,save_output=True,save_output_path="/tmp/output")

        unblur.write_results_to_database(aa.cistem_project,pars,res,change_image_assets=run_additional is None)
        if run_additional is None:
            aa.unblur_run = True
            ctx.obj.project.write()


@app.command()
def run_ctffind(
    ctx: typer.Context,
    cmd_prefix: str = typer.Option("", help="Prefix of run command"),
    cmd_suffix: str = typer.Option("", help="Suffix of run command"),
    num_cores: int = typer.Option(10, help="Number of cores to use"),
):
    """
    Run ctffind for each acquisition area
    """
    from pycistem.programs import ctffind

  
    for aa in ctx.obj.acquisition_areas:
        if aa.ctffind_run:
            continue
        typer.echo(f"Running ctffind for {aa.area_name}")
        pars, image_info = ctffind.parameters_from_database(aa.cistem_project,decolace=True)

        res = ctffind.run(pars,num_procs=num_cores,cmd_prefix=cmd_prefix,cmd_suffix=cmd_suffix)

        ctffind.write_results_to_database(aa.cistem_project,pars,res,image_info)
        aa.ctffind_run = True
        ctx.obj.project.write()
 
@app.command()
def update_database(
    project_main: Path = typer.Option(None, help="Path to wanted project file")
):
    from pycistem.core import Project

    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)
    
    for aa in project.acquisition_areas:
        pro = Project()
        pro.OpenProjectFromFile(Path(aa.cistem_project).as_posix())
        pro.database.CheckandUpdateSchema()
        pro.Close(True,True)

@app.command()
def redo_projects(
    project_main: Path = typer.Option(None, help="Path to wanted project file")
):
   
    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)

    for aa in project.acquisition_areas:
        aa.cistem_project = None
    
    project.write()

@app.command()
def redo_unblur(
    project_main: Path = typer.Option(None, help="Path to wanted project file")
):
   
    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)

    for aa in project.acquisition_areas:
        aa.unblur_run = False
    
    project.write()

@app.command()
def redo_ctffind(
    project_main: Path = typer.Option(None, help="Path to wanted project file")
):
   
    if project_main is None:
        project_path = Path(glob.glob("*.decolace")[0])
    project = ProcessingProject.read(project_path)

    for aa in project.acquisition_areas:
        aa.ctffind_run = False
    
    project.write()
