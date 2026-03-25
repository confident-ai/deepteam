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
from deepteam.test_case import RTTurn, ToolCall

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


import unicodedata

# Internal flag to control ASCII-only mode for encoding sanitization tests
_ASCII_STRICT = False


def set_ascii_strict_mode(enabled: bool) -> None:
    """Enable/disable strict ASCII-only mode for sanitize_prompt_for_encoding.

    When enabled, non-ASCII characters are dropped from the output. When
    disabled, common Unicode punctuation is normalized and some emoji are
    replaced with placeholders, but non-ASCII content is preserved where
    possible.
    """
    global _ASCII_STRICT
    _ASCII_STRICT = bool(enabled)


def sanitize_prompt_for_encoding(prompt: str) -> str:
    """Return an ASCII-safe version of the prompt for HTTP header encoding.

        This function replaces problematic Unicode characters with ASCII-safe
        equivalents while attempting to preserve essential non-ASCII content when
        possible. It also respects a strict ASCII mode for tests via
        set_ascii_strict_mode(enabled).

        Replacements (non-exhaustive):
    - Curly quotes are converted to ASCII quotes
    - Ellipsis … becomes ...
    - Em/En dashes are normalized to a plain hyphen with surrounding spaces
    - Non-breaking spaces are collapsed to regular spaces
    - Emoji and other symbol-like characters are replaced with the placeholder
      "[EMOJI]" in non-strict mode and dropped in strict mode
    """

    if prompt is None:
        return ""

    if not isinstance(prompt, str):
        prompt = str(prompt)

    s = prompt

    # Basic Unicode punctuation substitutions
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("‘", "'").replace("’", "'")
    s = s.replace("—", "-").replace("–", "-").replace("−", "-")
    s = s.replace("…", "...")
    s = s.replace("\u00a0", " ")  # non-breaking space to regular space

    # Build the output with per-character processing to handle strict mode and emoji
    out_chars = []
    i = 0
    while i < len(s):
        ch = s[i]
        code = ord(ch)
        if _ASCII_STRICT:
            if code <= 127:
                out_chars.append(ch)
            # else: drop non-ASCII characters
            i += 1
            continue
        else:
            # Check for emoji (category starting with 'So' or surrogate pairs for emoji)
            # Also handle emoji with skin tone modifiers (sequence of emoji + variation selector)
            cat = unicodedata.category(ch)
            if cat.startswith("So") or (0x1F300 <= code <= 0x1F9FF):
                # Skip this character and any following variation selectors or modifiers
                out_chars.append("[EMOJI]")
                i += 1
                # Skip variation selectors (U+FE00-U+FE0F) and skin tone modifiers (U+1F3FB-U+1F3FF)
                while i < len(s):
                    next_code = ord(s[i])
                    if (0xFE00 <= next_code <= 0xFE0F) or (
                        0x1F3FB <= next_code <= 0x1F3FF
                    ):
                        i += 1
                    else:
                        break
                continue
            out_chars.append(ch)
            i += 1

    result = "".join(out_chars)

    if _ASCII_STRICT:
        # Ensure ASCII-only output
        result = result.encode("ascii", "ignore").decode("ascii")

    return result
