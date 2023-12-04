from pathlib import Path
from typing import List
from types import SimpleNamespace
import typer
import glob
from rich import print
from rich.logging import RichHandler
from typer.core import TyperGroup

from decolace.processing.project_managment import ProcessingProject, DLContext, DLGlobals

from decolace.processing.cli_project_managment import app as project_managment_app
from decolace.processing.cli_preprocessing import app as preprocessing_app
from decolace.processing.cli_montaging import app as montaging_app
from decolace.processing.cli_match_template import app as match_template_app
from decolace.processing.cli_single_particle import app as single_particle_app
from decolace.processing.cli_edittags import app as edittags_app
from decolace.processing.cli_visualization import app as visualization_app

class OrderCommands(TyperGroup):
  def list_commands(self, ctx: typer.Context):
    """Return list of commands in the order appear."""
    return list(self.commands)    # get commands using self.commands

app = typer.Typer(
    cls=OrderCommands,
)

for command in project_managment_app.registered_commands:
    command.rich_help_panel="Project Managment Commands"
app.registered_commands += project_managment_app.registered_commands
for command in preprocessing_app.registered_commands:
    command.rich_help_panel="CisTEM Preprocessing Commands"
app.registered_commands += preprocessing_app.registered_commands
for command in montaging_app.registered_commands:
    command.rich_help_panel="DeCoLACE Montaging Commands"
app.registered_commands += montaging_app.registered_commands
for command in match_template_app.registered_commands:
    command.rich_help_panel="Template Matching Commands"
app.registered_commands += match_template_app.registered_commands
for command in single_particle_app.registered_commands:
    command.rich_help_panel="Single Particle Commands"
app.registered_commands += single_particle_app.registered_commands
for command in edittags_app.registered_commands:
    command.rich_help_panel="Edit Tags Commands"
app.registered_commands += edittags_app.registered_commands
for command in visualization_app.registered_commands:
    command.rich_help_panel="Visualization Commands"
app.registered_commands += visualization_app.registered_commands


@app.callback()
def main(
    ctx: DLContext,
    project: Path = typer.Option(None, help="Path to wanted project file",rich_help_panel="Expert Options"),
    acquisition_area_name: List[str] = typer.Option(None, help="List of acquisition areas names to process"),
    select_condition: str = typer.Option(None, help="Condition to select acquisition areas"),
    match_template_job_id: int = typer.Option(None, help="ID of template match job"),
    cistem_path: str = typer.Option("/groups/cryoadmin/software/CISTEM/je_dev/", help="Path to cistem binaries"),
):
    """DeCoLACE processing pipeline"""
    from decolace.processing.project_managment import process_experimental_conditions
    if project is None:
        potential_projects = glob.glob("*.decolace")
        if len(potential_projects) == 0:
            project_object = None
            ctx.obj = SimpleNamespace(project = None, acquisition_areas = [], match_template_job = None, cistem_path = cistem_path)
            return
        project = Path(potential_projects[0])
    project_object = ProcessingProject.read(project)
    aas_to_process = project_object.acquisition_areas
    if len(acquisition_area_name) > 0:
        aas_to_process = [aa for aa in project_object.acquisition_areas if aa.area_name in acquisition_area_name]
    if select_condition is not None:
        conditions = process_experimental_conditions(aas_to_process)
        key, value = select_condition.split("=")
        aas_to_process = [aa for i, aa in enumerate(aas_to_process) if key in conditions and i in conditions[key] and conditions[key][i] == value]
    if match_template_job_id is not None:
        array_position = [mtr.run_id  for mtr in project_object.match_template_runs].index(match_template_job_id)
        mtr = project_object.match_template_runs[array_position]
    else:
        mtr = None
    ctx.obj = DLGlobals(project = project_object, acquisition_areas = aas_to_process, match_template_job = mtr, cistem_path = cistem_path)
