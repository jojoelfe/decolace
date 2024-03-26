import typer
app = typer.Typer()


@app.command()
def edit_experimental(
    ctx: typer.Context,
):
    try:
        from fastapi import FastAPI
        from starlette.staticfiles import StaticFiles        
        import uvicorn
        import os 
        from pathlib import Path
        from decolace.processing.project_managment import AcquisitionAreaPreProcessing, process_experimental_conditions
        from pydantic import BaseModel
        class AcquisitionAreaExperimentalConditions(BaseModel):
            area_name: str
            experimental_conditions: dict
        current_directory = Path(__file__).parent.absolute()
        api = FastAPI()
        @api.get("/aa")
        async def get_aa() -> list[AcquisitionAreaExperimentalConditions]:
            experiment_consitions_dicts = process_experimental_conditions(ctx.obj.acquisition_areas)
            return [AcquisitionAreaExperimentalConditions(area_name=aa.area_name, experimental_conditions=experiment_consitions_dicts[i]) for i, aa in enumerate(ctx.obj.acquisition_areas)]
        api.mount("/", StaticFiles(directory=current_directory/'web_components/acquisition_area_spreadsheet/',html=True), name="static")

        
        # serve the ctx.aa object using FastAPI
        
        
        
        
        
        uvicorn.run(api, host="127.0.0.1", port=8000, log_level="info")
    except ModuleNotFoundError as e:
        print(e)
        typer.echo("FastAPI not installed. Please install using 'pip install fastapi'.")
    
    