from __future__ import annotations

from pathlib import Path

import typer

from deep_earth_degasation.config import load_config

app = typer.Typer(help="Deep Earth Degasation MVP utilities")


@app.command()
def validate_config(config_path: Path) -> None:
    """Validate a YAML configuration file."""
    config = load_config(config_path)
    typer.echo(config.model_dump_json(indent=2))


@app.command()
def status() -> None:
    """Print package status."""
    typer.echo("deep-earth-degasation MVP skeleton is installed.")


if __name__ == "__main__":
    app()
