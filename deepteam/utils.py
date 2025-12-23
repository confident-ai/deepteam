import time
import inspect
import os
from contextlib import contextmanager
from typing import Optional, List, Callable
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from deepteam.test_case import RTTurn

PROGRESS_ENABLED = os.getenv("DEEPTEAM_SHOW_PROGRESS", "true").lower() in (
    "true",
    "1",
    "yes",
)


def validate_model_callback_signature(
    model_callback: Callable,
    async_mode: bool,
):
    if async_mode and not inspect.iscoroutinefunction(model_callback):
        raise ValueError(
            "`model_callback` must be an async callback function. `async_mode` has been set to True."
        )
    if not async_mode and inspect.iscoroutinefunction(model_callback):
        raise ValueError(
            "`model_callback` must be a sync callback function. `async_mode` has been set to False."
        )


def format_turns(turns: List[RTTurn]):
    if not turns:
        raise ValueError("There are no 'turns' to format.")

    formatted_turns = "Full Conversation To Evaluate: \n"
    for turn in turns:
        formatted_turns += f"Role: {turn.role} \n"
        formatted_turns += f"Content: {turn.content} \n\n"
    formatted_turns += "End of conversation. \n"

    return formatted_turns


@contextmanager
def _null_progress():
    """No-op context manager for when progress is disabled."""
    yield None


def create_progress(enabled: Optional[bool] = None) -> Progress:
    if enabled is False or (enabled is None and not PROGRESS_ENABLED):
        return _null_progress()

    return Progress(
        SpinnerColumn(),
        TextColumn("[bold bright_white]{task.description}", justify="left"),
        BarColumn(bar_width=None),
        TaskProgressColumn(
            text_format="[cyan]{task.completed}[/]/[bright_white]{task.total}"
        ),
        TimeElapsedColumn(),
        expand=True,
        transient=False,
    )


def add_pbar(
    progress: Optional[Progress],
    description: str,
    total: Optional[int] = None,
    enabled: Optional[bool] = None,
) -> Optional[int]:
    if progress is None or not hasattr(progress, "add_task"):
        return None
    return progress.add_task(description, total=total)


def update_pbar(
    progress: Optional[Progress],
    pbar_id: Optional[int],
    advance: int = 1,
    advance_to_end: bool = False,
    remove: bool = True,
    total: Optional[int] = None,
):
    if progress is None or pbar_id is None:
        return
    task = next((t for t in progress.tasks if t.id == pbar_id), None)
    if task is None:
        return
    if advance_to_end:
        advance = task.remaining
    progress.update(pbar_id, advance=advance, total=total)
    task = next((t for t in progress.tasks if t.id == pbar_id), None)
    if task is not None and task.finished and remove:
        progress.remove_task(pbar_id)


def remove_pbars(
    progress: Optional[Progress], pbar_ids: List[int], cascade: bool = True
):
    if progress is None:
        return
    for pbar_id in pbar_ids:
        if cascade:
            time.sleep(0.1)
        task = next((t for t in progress.tasks if t.id == pbar_id), None)
        if task is not None:
            progress.remove_task(pbar_id)
