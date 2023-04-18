import typer

app = typer.Typer()


@app.command()
def new_sesion(name: str = typer.Argument(..., help="Name of the session")):
    typer.echo(f"Created new session {name}")


@app.command()
def new_grid(name: str = typer.Argument(..., help="Name of the grid")):
    typer.echo(f"Created new grid {name}")


@app.command()
def add_acquisition_area():
    typer.echo("Added acquisition area")


@app.command()
def image_lamella():
    typer.echo("Image lamella")


@app.command()
def acquire():
    typer.echo("Acquire")
