from __future__ import annotations

import ast
import io
import logging
import subprocess
import sys
from pathlib import Path

import diagnostic_script as ds
from rich.console import Console


def test_resolve_verbosity_level() -> None:
    assert ds.resolve_verbosity_level(0) == ds.VerbosityLevel.INFO
    assert ds.resolve_verbosity_level(1) == ds.VerbosityLevel.DEBUG
    assert ds.resolve_verbosity_level(2) == ds.VerbosityLevel.TRACE
    assert ds.resolve_verbosity_level(4) == ds.VerbosityLevel.TRACE


def test_build_sparkline_shapes() -> None:
    assert ds.build_sparkline([]) == "▁"
    assert ds.build_sparkline([1.0]) == "█"
    spark = ds.build_sparkline([1.0, 2.0, 3.0, 4.0])
    assert len(spark) == 4
    assert spark[0] == "▁"
    assert spark[-1] == "█"


def test_status_theme_covers_all_statuses() -> None:
    assert set(ds.STATUS_THEME) == {
        ds.CheckStatus.PASS,
        ds.CheckStatus.WARN,
        ds.CheckStatus.FAIL,
        ds.CheckStatus.INFO,
    }


def test_renderer_hides_pass_and_info_at_default_verbosity() -> None:
    output = io.StringIO()
    renderer = ds.DiagnosticRenderer(
        console=Console(file=output, force_terminal=False),
        verbosity=ds.VerbosityLevel.INFO,
    )
    renderer.render_event(
        ds.CheckEvent(section="x", status=ds.CheckStatus.PASS, message="pass event")
    )
    renderer.render_event(
        ds.CheckEvent(section="x", status=ds.CheckStatus.INFO, message="info event")
    )
    assert output.getvalue().strip() == ""


def test_renderer_shows_pass_at_debug_verbosity() -> None:
    output = io.StringIO()
    renderer = ds.DiagnosticRenderer(
        console=Console(file=output, force_terminal=False),
        verbosity=ds.VerbosityLevel.DEBUG,
    )
    renderer.render_event(
        ds.CheckEvent(section="x", status=ds.CheckStatus.PASS, message="pass event")
    )
    rendered = output.getvalue()
    assert "PASS" in rendered
    assert "pass event" in rendered


def test_issue_summary_caps_long_lists() -> None:
    output = io.StringIO()
    renderer = ds.DiagnosticRenderer(
        console=Console(file=output, force_terminal=False),
        verbosity=ds.VerbosityLevel.INFO,
    )
    events = [
        ds.CheckEvent(section="x", status=ds.CheckStatus.WARN, message=f"warn {idx}")
        for idx in range(ds.MAX_ISSUES_IN_SUMMARY + 2)
    ]
    renderer.render_issue_lists(events)
    rendered = output.getvalue()
    assert "+2 more" in rendered


def test_artifact_writer_writes_expected_files(tmp_path: Path) -> None:
    ctx = ds.create_run_context(ds.VerbosityLevel.INFO, tmp_path, json_report=True)
    try:
        ctx.events.append(
            ds.CheckEvent(
                section="unit-test",
                status=ds.CheckStatus.PASS,
                message="test event",
                duration_ms=12.5,
            )
        )
        ctx.durations.append(("unit-test", 12.5))
        ctx.logger.info("artifact writer smoke test")

        writer = ds.ArtifactWriter(ctx)
        writer.write_text_report(0.2)
        writer.write_json_report(0.2)
        writer.write_log_capture()
    finally:
        ds.teardown_logging(ctx.logger)

    assert (tmp_path / ds.DEFAULT_RESULTS_FILE).exists()
    assert (tmp_path / ds.DEFAULT_JSON_FILE).exists()
    assert (tmp_path / ds.DEFAULT_LOG_FILE).exists()


def test_logger_is_file_only_for_compact_console(tmp_path: Path) -> None:
    logger = ds._create_logger(tmp_path / ds.DEFAULT_LOG_FILE, ds.VerbosityLevel.INFO)
    try:
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.FileHandler)
    finally:
        ds.teardown_logging(logger)


def test_cli_help_renders() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "diagnostic_script.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    output = completed.stdout + completed.stderr
    assert "Usage" in output
    assert "--output-dir" in output
    assert "--json-report" in output
    assert "--verbose" in output


def test_no_bare_print_calls() -> None:
    source = Path("diagnostic_script.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    bare_print_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    assert bare_print_calls == []
