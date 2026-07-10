"""
Main entrypoint for all ZenML pipeline runs.

Usage:
    python run.py run --workflow matrix_factorization --pipeline training_pipeline --config workflows/matrix_factorization/configs/aws/training_pipeline.yaml --stack aws_stack
    python run.py run --workflow matrix_factorization --pipeline training_pipeline --no-cache
    python run.py list-workflows
    python run.py list-pipelines --workflow matrix_factorization
"""

from __future__ import annotations

import importlib
from pathlib import Path

import typer
from zenml.client import Client
from zenml.exceptions import EntityExistsError

app = typer.Typer(
    name="aips-recs",
    help="AIPS Recommendations — ZenML MLOps pipeline runner",
    add_completion=False,
)

WORKFLOWS_DIR = Path(__file__).parent / "workflows"


def _discover_workflows() -> list[str]:
    """Return all workflow names found under workflows/ (dirs with __init__.py)."""
    if not WORKFLOWS_DIR.exists():
        return []
    return sorted(
        d.name
        for d in WORKFLOWS_DIR.iterdir()
        if d.is_dir() and (d / "__init__.py").exists() and not d.name.startswith("_")
    )


def _discover_pipelines(workflow: str) -> list[str]:
    """Return pipeline short names available for a workflow by scanning its pipelines/ dir."""
    pipelines_dir = WORKFLOWS_DIR / workflow / "pipelines"
    if not pipelines_dir.exists():
        return []
    return sorted(
        d.stem
        for d in pipelines_dir.iterdir()
        if not d.is_dir() and not d.name.startswith("_") and d.name.endswith(".py")
    )


def _set_stack(stack_name: str | None) -> None:
    """Activate the given ZenML stack if provided."""
    if stack_name:
        client = Client()
        client.activate_stack(stack_name)
        typer.echo(f"Active stack set to: {stack_name}")


def _set_project(project_name: str) -> None:
    """Set the active ZenML project to the given name."""
    if project_name:
        client = Client()
        try:
            client.create_project(project_name, "Auto-created by run.py")  # Create if not exists
            client.set_active_project(project_name)
        except EntityExistsError:
            client.set_active_project(project_name)
        typer.echo(f"Active project set to: {project_name}")


def _dispatch(workflow: str, pipeline: str, run_options: dict) -> None:
    """Dynamically import and execute the pipeline function for the given workflow."""
    module_name = pipeline
    module_path = f"workflows.{workflow}.pipelines.{module_name}"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        typer.echo(f"Pipeline module not found: {module_path}\n  {exc}", err=True)
        raise typer.Exit(code=1) from exc
    pipeline_fn = getattr(module, module_name, None)
    if pipeline_fn is None:
        typer.echo(
            f"Pipeline function '{module_name}' not found in {module_path}",
            err=True,
        )
        raise typer.Exit(code=1)
    pipeline_fn.with_options(**run_options)()


@app.command()
def run(
    workflow: str = typer.Option(
        ...,
        "--workflow",
        "-w",
        help="Workflow to run (directory name under workflows/).",
    ),
    pipeline: str = typer.Option(
        ...,
        "--pipeline",
        "-p",
        help="Pipeline to run (e.g. training, serving, monitoring).",
    ),
    config: Path | None = typer.Option(  # noqa: B008
        None,
        "--config",
        "-c",
        help=(
            "Path to ZenML pipeline run config YAML. "
            "Defaults to workflows/<workflow>/configs/local/<pipeline>.yaml."
        ),
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    stack: str | None = typer.Option(
        None,
        "--stack",
        "-s",
        help="ZenML stack to activate before running. Overrides current active stack.",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable step-level caching for this run.",
    ),
) -> None:
    """Run a ZenML pipeline for a given workflow."""
    workflows = _discover_workflows()
    if workflow not in workflows:
        typer.echo(
            f"Unknown workflow '{workflow}'. Available: {', '.join(workflows) or 'none'}",
            err=True,
        )
        raise typer.Exit(code=1)

    pipelines = _discover_pipelines(workflow)
    if pipeline not in pipelines:
        typer.echo(
            f"Unknown pipeline '{pipeline}' for workflow '{workflow}'. "
            f"Available: {', '.join(pipelines) or 'none'}",
            err=True,
        )
        raise typer.Exit(code=1)

    if config is None:
        config = WORKFLOWS_DIR / workflow / "configs" / "local" / f"{pipeline}.yaml"
        if not config.exists():
            typer.echo(
                f"No --config provided and default not found: {config}",
                err=True,
            )
            raise typer.Exit(code=1)

    # Set the active ZenML stack if provided
    _set_stack(stack)

    # (PRO Only) Set the active ZenML project to the workflow name
    # _set_project(workflow)

    run_options = {
        "config_path": str(config),
        "enable_cache": not no_cache,
    }

    typer.echo(f"Running '{workflow}/{pipeline}' with config '{config}'")
    _dispatch(workflow, pipeline, run_options)


@app.command("list-workflows")
def list_workflows() -> None:
    """List all available workflows under workflows/."""
    workflows = _discover_workflows()
    if not workflows:
        typer.echo("No workflows found under workflows/")
        return
    typer.echo("Available workflows:")
    for wf in workflows:
        typer.echo(f"  - {wf}")


@app.command("list-pipelines")
def list_pipelines(
    workflow: str = typer.Option(
        ...,
        "--workflow",
        "-w",
        help="Workflow to list pipelines for.",
    ),
) -> None:
    """List all available pipelines for a workflow."""
    workflows = _discover_workflows()
    if workflow not in workflows:
        typer.echo(
            f"Unknown workflow '{workflow}'. Available: {', '.join(workflows) or 'none'}",
            err=True,
        )
        raise typer.Exit(code=1)
    pipelines = _discover_pipelines(workflow)
    if not pipelines:
        typer.echo(f"No pipelines found for workflow '{workflow}'")
        return
    typer.echo(f"Pipelines for '{workflow}':")
    for p in pipelines:
        typer.echo(f"  - {p}")


if __name__ == "__main__":
    app()
