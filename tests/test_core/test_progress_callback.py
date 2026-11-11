import pytest

from deepteam.utils import (
    create_progress,
    add_pbar,
    update_pbar,
    set_progress_callback,
    progress_callback_context,
)


class TestProgressCallback:
    """Unit tests for the progress callback (issue #173). No model/API key."""

    def setup_method(self):
        set_progress_callback(None)

    def teardown_method(self):
        set_progress_callback(None)

    def test_no_callback_is_noop(self):
        # With no callback registered, progress helpers must still work.
        with create_progress(enabled=True) as progress:
            pbar_id = add_pbar(progress, "Task", total=2)
            update_pbar(progress, pbar_id)

    def test_callback_receives_start_and_advance(self):
        events = []
        set_progress_callback(events.append)

        with create_progress(enabled=True) as progress:
            pbar_id = add_pbar(progress, "Simulating attacks", total=2)
            update_pbar(progress, pbar_id)
            update_pbar(progress, pbar_id)

        assert events[0]["event"] == "start"
        assert events[0]["description"] == "Simulating attacks"
        assert events[0]["completed"] == 0
        assert events[0]["total"] == 2

        advances = [e for e in events if e["event"] == "advance"]
        assert len(advances) == 2
        assert advances[-1]["completed"] == 2
        assert advances[-1]["total"] == 2

    def test_context_manager_scopes_and_restores(self):
        events = []
        with progress_callback_context(events.append):
            add_pbar(None, "Within", total=1)
        # Callback is restored (to None) after the context.
        add_pbar(None, "Outside", total=1)

        descriptions = [e["description"] for e in events]
        assert "Within" in descriptions
        assert "Outside" not in descriptions

    def test_start_emitted_even_when_progress_disabled(self):
        # A UI may want events without console rendering (progress disabled).
        events = []
        set_progress_callback(events.append)

        pbar_id = add_pbar(None, "Headless", total=3)

        assert pbar_id is None
        assert events and events[0]["description"] == "Headless"

    def test_callback_exception_does_not_break_run(self):
        def boom(event):
            raise RuntimeError("user callback error")

        set_progress_callback(boom)

        # A raising user callback must not propagate into the assessment.
        with create_progress(enabled=True) as progress:
            pbar_id = add_pbar(progress, "Task", total=1)
            update_pbar(progress, pbar_id)
