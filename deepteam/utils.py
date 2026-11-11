import time
import inspect
import os
from contextlib import contextmanager
from typing import Optional, List, Callable, Dict, Any
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from deepteam.test_case import RTTurn, ToolCall

PROGRESS_ENABLED = os.getenv("DEEPTEAM_SHOW_PROGRESS", "true").lower() in (
    "true",
    "1",
    "yes",
)

SPANS_CONTEXT_LIMIT = int(os.getenv("SPANS_CONTEXT_LIMIT", "40000"))


# Progress callback (issue #173): surface the progress deepteam shows on the
# console to a UI such as Streamlit. A single module-level observer fires from
# the `add_pbar` / `update_pbar` choke points, so every progress bar reports
# through it, with no callback threaded through the attack modules.
ProgressCallback = Callable[[Dict[str, Any]], None]
_progress_callback: Optional[ProgressCallback] = None


def set_progress_callback(callback: Optional[ProgressCallback]) -> None:
    """Register (or clear) a callback that receives assessment progress events.

    The callback is invoked with a dict, for example::

        {"event": "start", "description": "Simulating attacks",
         "completed": 0, "total": 12}
        {"event": "advance", "description": "Simulating attacks",
         "completed": 1, "total": 12}

    This mirrors the progress deepteam renders to the console so it can be shown
    in a custom UI such as Streamlit. Pass ``None`` to clear the callback.
    """
    global _progress_callback
    _progress_callback = callback


@contextmanager
def progress_callback_context(callback: Optional[ProgressCallback]):
    """Temporarily register a progress callback, restoring the previous one."""
    global _progress_callback
    previous = _progress_callback
    _progress_callback = callback
    try:
        yield
    finally:
        _progress_callback = previous


def _emit_progress(
    event: str,
    description: str,
    completed: Optional[int],
    total: Optional[int],
) -> None:
    callback = _progress_callback
    if callback is None:
        return
    try:
        callback(
            {
                "event": event,
                "description": description,
                "completed": completed,
                "total": total,
            }
        )
    except Exception:
        # A user-supplied callback must never break an assessment run.
        pass


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
        formatted_turns += f"Content: {turn.content} \n"
        formatted_turns += (
            f"Retrieved Context: {turn.retrieval_context} \n"
            if turn.retrieval_context
            else ""
        )
        formatted_turns += (
            f"Tools Called: {format_tools_called(turn.tools_called)} \n\n"
            if turn.tools_called
            else "\n"
        )
    formatted_turns += "End of conversation. \n"

    return formatted_turns


def format_tools_called(tools_called: List[ToolCall]) -> Optional[str]:
    if not tools_called:
        return None

    formatted_tools = []

    for idx, tool in enumerate(tools_called, start=1):
        tool_info = [
            f"Tool #{idx}:",
            f"  Name: {tool.name}",
            f"  Description: {tool.description}" if tool.description else "",
            f"  Reasoning: {tool.reasoning}" if tool.reasoning else "",
            (
                f"  Input Parameters: {tool.input_parameters}"
                if tool.input_parameters
                else ""
            ),
            f"  Output: {tool.output}" if tool.output is not None else "",
        ]

        formatted_tools.append("\n".join(tool_info))

    return "\n\n".join(formatted_tools)


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
    _emit_progress("start", description, 0, total)
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
    if task is not None:
        _emit_progress("advance", task.description, task.completed, task.total)
        if task.finished and remove:
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
