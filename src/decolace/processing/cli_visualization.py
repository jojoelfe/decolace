import typer
from pathlib import Path
from typing import List, Optional
from decolace.processing.project_managment import DLContext

app = typer.Typer()

@app.command()
def render_matches_via_blender(
    ctx: DLContext,
    filter_set_name: str = typer.Argument(..., help="Name of filter set"),
    blender_template: Path = typer.Option(..., help="Path to blender template file"),
    tag: str = typer.Option("", help="Tag for the rendered images"),
):
    """
    Render matches via blender
    """
    from decolace.processing.match_visualization import render_aa

    for aa in ctx.obj.acquisition_areas:
        filename = ctx.obj.project.project_path / "visualization" / "matches" / f"{aa.area_name}_{ctx.obj.match_template_job.run_id}_tm_package_filtered_{filter_set_name}_{tag}.png"
        filename.parent.mkdir(parents=True, exist_ok=True)
        render_aa(ctx.obj.project, aa, ctx.obj.match_template_job, filter_set_name, filename, blender_template)
