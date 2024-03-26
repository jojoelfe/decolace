import typer
from pathlib import Path
from typing import List, Optional
from decolace.processing.project_managment import DLContext

app = typer.Typer()

@app.command()
def render_matches_via_blender(
    ctx: DLContext,
    filter_set_name: str = typer.Argument(..., help="Name of filter set"),
):
    """
    Render matches via blender
    """
    from decolace.processing.match_visualization import render_aa

    for aa in ctx.obj.project.acquisition_areas:
        filename = ctx.obj.project.project_path / "visualization" / "matches" / f"{aa.area_name}_{ctx.obj.match_template_job.run_id}_tm_package_filtered_{filter_set_name}.png"
        filename.parent.mkdir(parents=True, exist_ok=True)
        render_aa(ctx.obj.project, aa, ctx.obj.match_template_job, filter_set_name, filename)
