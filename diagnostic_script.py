#!/usr/bin/env python3
"""RapidShot DirectX diagnostics with a Typer + Rich CLI."""

from __future__ import annotations

import ctypes
import json
import logging
import os
import platform
import struct
import subprocess
import tempfile
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Callable, Sequence

import comtypes
import typer
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.traceback import Traceback

APP_NAME = "RapidShot DirectX Diagnostic Tool"
TRACE_LEVEL_NUM = 5
SPARK_CHARS = "▁▂▃▄▅▆▇█"
DEFAULT_LOG_FILE = "rapidshot_diagnostics.log"
DEFAULT_RESULTS_FILE = "rapidshot_diagnostics_results.txt"
DEFAULT_JSON_FILE = "rapidshot_diagnostics_results.json"


def _install_trace_level() -> None:
    """Register TRACE logging (below DEBUG) exactly once."""
    if logging.getLevelName(TRACE_LEVEL_NUM) != "TRACE":
        logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

    if not hasattr(logging.Logger, "trace"):

        def trace(
            self: logging.Logger, message: str, *args: Any, **kwargs: Any
        ) -> None:
            if self.isEnabledFor(TRACE_LEVEL_NUM):
                self._log(TRACE_LEVEL_NUM, message, args, **kwargs)

        setattr(logging.Logger, "trace", trace)


class VerbosityLevel(IntEnum):
    INFO = 0
    DEBUG = 1
    TRACE = 2


class CheckStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    INFO = "info"


STATUS_THEME: dict[CheckStatus, tuple[str, str, str]] = {
    CheckStatus.PASS: ("PASS", "green", "✓"),
    CheckStatus.WARN: ("WARN", "yellow", "!"),
    CheckStatus.FAIL: ("FAIL", "red", "✗"),
    CheckStatus.INFO: ("INFO", "cyan", "•"),
}


@dataclass
class CheckEvent:
    section: str
    status: CheckStatus
    message: str
    duration_ms: float | None = None
    details: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ArtifactRecord:
    artifact_type: str
    path: Path
    bytes_written: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RunContext:
    console: Console
    logger: logging.Logger
    verbosity: VerbosityLevel
    output_dir: Path
    json_report: bool
    log_path: Path
    events: list[CheckEvent] = field(default_factory=list)
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    durations: list[tuple[str, float]] = field(default_factory=list)
    renderer: "DiagnosticRenderer | None" = None


def resolve_verbosity_level(verbose_count: int) -> VerbosityLevel:
    """Map Typer count=True verbosity to the internal enum."""
    if verbose_count >= 2:
        return VerbosityLevel.TRACE
    if verbose_count == 1:
        return VerbosityLevel.DEBUG
    return VerbosityLevel.INFO


def build_sparkline(samples: Sequence[float]) -> str:
    """Build a compact sparkline from numeric samples."""
    if not samples:
        return SPARK_CHARS[0]
    if len(samples) == 1:
        return SPARK_CHARS[-1]

    minimum = min(samples)
    maximum = max(samples)
    spread = maximum - minimum
    if spread <= 0:
        return SPARK_CHARS[-1] * len(samples)

    points: list[str] = []
    ceiling = len(SPARK_CHARS) - 1
    for value in samples:
        normalized = (value - minimum) / spread
        index = max(0, min(ceiling, int(round(normalized * ceiling))))
        points.append(SPARK_CHARS[index])
    return "".join(points)


def _format_bytes(size: int) -> str:
    """Format bytes into readable units."""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.2f} MB"


def _parse_version(version: str) -> tuple[int, ...]:
    pieces: list[int] = []
    for raw_piece in version.split("."):
        digits = "".join(char for char in raw_piece if char.isdigit())
        if not digits:
            break
        pieces.append(int(digits))
    return tuple(pieces) if pieces else (0,)


def release_com_ptr(ptr: Any) -> None:
    """Release a COM pointer exactly once and clear it."""
    if ptr is None:
        return
    try:
        if not bool(ptr):
            return
    except Exception:
        return

    ptr.Release()

    try:
        ctypes.cast(ctypes.byref(ptr), ctypes.POINTER(ctypes.c_void_p))[0] = None
    except Exception:
        pass


class DiagnosticRenderer:
    """Single output pipeline for all user-visible diagnostics."""

    def __init__(self, console: Console, verbosity: VerbosityLevel) -> None:
        self.console = console
        self.verbosity = verbosity

    def render_banner(self) -> None:
        body = (
            "[bold cyan]DirectX + RapidShot health checks[/]\n"
            "[dim]Structured telemetry, rich diagnostics, and artifact traceability.[/]"
        )
        self.console.print(
            Panel.fit(
                body, title=f"[bold magenta]{APP_NAME}[/]", border_style="magenta"
            )
        )

    def render_section(self, title: str) -> None:
        self.console.rule(f"[bold blue]{title}[/]")

    def render_event(self, event: CheckEvent) -> None:
        label, style, icon = STATUS_THEME[event.status]
        duration_suffix = ""
        if event.duration_ms is not None:
            duration_suffix = f" [dim]({event.duration_ms:.1f} ms)[/]"
        self.console.print(
            f"[bold {style}]{icon} {label:<4}[/] {event.message}{duration_suffix}"
        )

        if event.details and self.verbosity >= VerbosityLevel.DEBUG:
            border = "bright_blue" if event.status == CheckStatus.INFO else style
            self.console.print(
                Panel(
                    event.details,
                    title=f"{event.section} details",
                    border_style=border,
                    padding=(0, 1),
                )
            )

    def render_kv_table(self, title: str, rows: Sequence[tuple[str, str]]) -> None:
        table = Table(
            title=title,
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
            show_lines=False,
        )
        table.add_column("Key", style="bold white", no_wrap=True)
        table.add_column("Value", style="bright_white")
        for key, value in rows:
            table.add_row(key, value)
        self.console.print(table)

    def render_block(self, title: str, content: str, style: str = "cyan") -> None:
        self.console.print(
            Panel(content, title=title, border_style=style, padding=(0, 1))
        )

    def render_exception(self, title: str, error: BaseException) -> None:
        self.console.print(
            Panel.fit(
                f"[bold red]{title}[/]\n[red]{error}[/]",
                title="Exception",
                border_style="red",
            )
        )
        if self.verbosity >= VerbosityLevel.TRACE:
            self.console.print(
                Traceback.from_exception(
                    type(error), error, error.__traceback__, show_locals=True
                )
            )

    def render_telemetry_cards(
        self,
        events: Sequence[CheckEvent],
        durations: Sequence[tuple[str, float]],
        total_seconds: float,
    ) -> None:
        pass_count = sum(1 for event in events if event.status == CheckStatus.PASS)
        warn_count = sum(1 for event in events if event.status == CheckStatus.WARN)
        fail_count = sum(1 for event in events if event.status == CheckStatus.FAIL)
        check_samples = [duration for _, duration in durations]
        spark = build_sparkline(check_samples)
        duration_text = "n/a"
        if check_samples:
            duration_text = (
                f"min {min(check_samples):.1f} ms | max {max(check_samples):.1f} ms"
            )

        cards = [
            Panel(
                f"[bold green]{pass_count}[/]\n[dim]Passing signals[/]",
                title="Pass",
                border_style="green",
                padding=(1, 2),
            ),
            Panel(
                f"[bold yellow]{warn_count}[/]\n[dim]Warnings[/]",
                title="Warn",
                border_style="yellow",
                padding=(1, 2),
            ),
            Panel(
                f"[bold red]{fail_count}[/]\n[dim]Failures[/]",
                title="Fail",
                border_style="red",
                padding=(1, 2),
            ),
            Panel(
                f"[bold cyan]{len(durations)}[/] checks\n"
                f"[bold white]{spark}[/]\n"
                f"[dim]{duration_text}[/]\n"
                f"[dim]Total {total_seconds:.2f}s[/]",
                title="Telemetry",
                border_style="cyan",
                padding=(1, 2),
            ),
        ]
        self.console.print(Columns(cards, expand=True, equal=True))

    def render_summary_table(self, events: Sequence[CheckEvent]) -> None:
        table = Table(
            title="Diagnostic Summary",
            box=box.MINIMAL_DOUBLE_HEAD,
            header_style="bold magenta",
        )
        table.add_column("Status", style="bold")
        table.add_column("Count", justify="right")

        for status in (CheckStatus.PASS, CheckStatus.WARN, CheckStatus.FAIL):
            label, style, _ = STATUS_THEME[status]
            count = sum(1 for event in events if event.status == status)
            table.add_row(f"[{style}]{label}[/]", str(count))

        self.console.print(table)

    def render_issue_lists(self, events: Sequence[CheckEvent]) -> None:
        failures = [
            event.message for event in events if event.status == CheckStatus.FAIL
        ]
        warnings = [
            event.message for event in events if event.status == CheckStatus.WARN
        ]

        if failures:
            self.render_block(
                "Failed Checks",
                "\n".join(f"- {line}" for line in failures),
                style="red",
            )
        if warnings:
            self.render_block(
                "Warnings", "\n".join(f"- {line}" for line in warnings), style="yellow"
            )
        if not failures:
            self.render_block(
                "Outcome", "[bold green]All critical checks passed.[/]", style="green"
            )

    def render_file_spotlight(
        self,
        artifact_type: str,
        path: Path,
        phase: str,
        bytes_written: int | None = None,
    ) -> None:
        parts = [
            f"[bold cyan]{phase}[/]",
            f"[bold white][link=file://{path}]{path}[/link][/bold white]",
        ]
        if bytes_written is not None:
            parts.append(f"[dim]{_format_bytes(bytes_written)}[/]")
        body = "\n".join(parts)
        self.console.print(
            Panel.fit(
                body, title=f"{artifact_type} artifact", border_style="bright_cyan"
            )
        )

    def render_artifacts(self, artifacts: Sequence[ArtifactRecord]) -> None:
        table = Table(
            title="Written Artifacts", box=box.SIMPLE_HEAVY, header_style="bold cyan"
        )
        table.add_column("Type", style="bold white")
        table.add_column("Path", style="cyan")
        table.add_column("Size", justify="right", style="green")
        table.add_column("Timestamp", style="dim")

        for artifact in artifacts:
            table.add_row(
                artifact.artifact_type,
                str(artifact.path),
                _format_bytes(artifact.bytes_written),
                artifact.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            )
        self.console.print(table)


class ArtifactWriter:
    """Persist diagnostics artifacts with visible output spotlighting."""

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx

    def _record_artifact(self, artifact_type: str, path: Path) -> None:
        bytes_written = path.stat().st_size if path.exists() else 0
        self.ctx.artifacts.append(
            ArtifactRecord(
                artifact_type=artifact_type, path=path, bytes_written=bytes_written
            )
        )

    def write_text_report(self, total_seconds: float) -> Path:
        path = self.ctx.output_dir / DEFAULT_RESULTS_FILE
        self.ctx.renderer.render_file_spotlight("Text summary", path, "Writing")

        pass_events = [
            event.message
            for event in self.ctx.events
            if event.status == CheckStatus.PASS
        ]
        warn_events = [
            event.message
            for event in self.ctx.events
            if event.status == CheckStatus.WARN
        ]
        fail_events = [
            event.message
            for event in self.ctx.events
            if event.status == CheckStatus.FAIL
        ]

        lines = [
            f"RapidShot DirectX Diagnostic Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Duration: {total_seconds:.2f}s",
            f"PASSED: {len(pass_events)}",
            f"WARNINGS: {len(warn_events)}",
            f"FAILED: {len(fail_events)}",
            "",
        ]

        if fail_events:
            lines.append("FAILED TESTS:")
            lines.extend(f"  - {entry}" for entry in fail_events)
            lines.append("")

        if warn_events:
            lines.append("WARNINGS:")
            lines.extend(f"  - {entry}" for entry in warn_events)
            lines.append("")

        lines.append("PASSED CHECKS:")
        lines.extend(f"  - {entry}" for entry in pass_events)
        lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        self._record_artifact("text-summary", path)
        self.ctx.renderer.render_file_spotlight(
            "Text summary", path, "Saved", bytes_written=path.stat().st_size
        )
        self.ctx.logger.info("Saved text summary report to %s", path)
        return path

    def write_json_report(self, total_seconds: float) -> Path:
        path = self.ctx.output_dir / DEFAULT_JSON_FILE
        self.ctx.renderer.render_file_spotlight("JSON report", path, "Writing")

        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "duration_seconds": round(total_seconds, 4),
            "verbosity": int(self.ctx.verbosity),
            "events": [
                {
                    "section": event.section,
                    "status": event.status.value,
                    "message": event.message,
                    "duration_ms": event.duration_ms,
                    "details": event.details,
                    "timestamp": event.timestamp.isoformat() + "Z",
                }
                for event in self.ctx.events
            ],
            "durations_ms": [
                {"section": section, "duration_ms": duration}
                for section, duration in self.ctx.durations
            ],
        }

        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._record_artifact("json-telemetry", path)
        self.ctx.renderer.render_file_spotlight(
            "JSON report", path, "Saved", bytes_written=path.stat().st_size
        )
        self.ctx.logger.info("Saved JSON telemetry report to %s", path)
        return path

    def write_log_capture(self) -> Path:
        for handler in self.ctx.logger.handlers:
            if hasattr(handler, "flush"):
                handler.flush()

        log_path = self.ctx.log_path
        if not any(record.path == log_path for record in self.ctx.artifacts):
            self._record_artifact("log", log_path)

        size = log_path.stat().st_size if log_path.exists() else 0
        self.ctx.renderer.render_file_spotlight(
            "Log", log_path, "Finalized", bytes_written=size
        )
        return log_path


class DiagnosticRunner:
    """Runs all diagnostics and emits structured events."""

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.current_section = "General"

    def _emit(
        self,
        status: CheckStatus,
        message: str,
        *,
        details: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        event = CheckEvent(
            section=self.current_section,
            status=status,
            message=message,
            duration_ms=duration_ms,
            details=details,
        )
        self.ctx.events.append(event)
        self.ctx.renderer.render_event(event)

        log_message = f"[{self.current_section}] {message}"
        if status == CheckStatus.FAIL:
            self.ctx.logger.error(log_message)
        elif status == CheckStatus.WARN:
            self.ctx.logger.warning(log_message)
        elif status == CheckStatus.PASS:
            self.ctx.logger.info(log_message)
        else:
            self.ctx.logger.debug(log_message)

        if details and self.ctx.verbosity >= VerbosityLevel.TRACE:
            self.ctx.logger.trace("[%s] details: %s", self.current_section, details)

    def _capture_exception(self, title: str, error: BaseException) -> None:
        self._emit(CheckStatus.FAIL, f"{title}: {error}")
        self.ctx.logger.exception("%s", title)
        self.ctx.renderer.render_exception(title, error)

    def run(self) -> int:
        self.ctx.renderer.render_banner()
        started = time.perf_counter()

        checks: list[tuple[str, Callable[[], bool]]] = [
            ("System Information", self.test_system_info),
            ("Dependency Verification", self.check_dependencies),
            ("DirectX DLL Verification", self.test_directx_dlls),
            ("GPU Information", self.test_gpu_info),
            ("DXGI Factory Creation", self.test_dxgi_factory_creation),
            ("D3D11 Device Creation", self.test_d3d11_device_creation),
            ("Desktop Duplication API", self.test_desktop_duplication),
            ("RapidShot Import Test", self.test_rapidshot_imports),
            ("RapidShot Device Info", self.test_rapidshot_device_info),
            ("RapidShot Output Info", self.test_rapidshot_output_info),
            ("RapidShot Create", self.test_rapidshot_create),
        ]

        for section_name, check in checks:
            self.current_section = section_name
            self.ctx.renderer.render_section(section_name)
            self.ctx.logger.info("Starting check: %s", section_name)

            section_start = time.perf_counter()
            with self.ctx.console.status(
                f"[bold cyan]Running {section_name}[/]", spinner="dots"
            ):
                try:
                    check()
                except Exception as error:
                    self._capture_exception(f"Unhandled error in {section_name}", error)
            duration_ms = (time.perf_counter() - section_start) * 1000.0
            self.ctx.durations.append((section_name, duration_ms))

            if self.ctx.verbosity >= VerbosityLevel.DEBUG:
                self._emit(
                    CheckStatus.INFO, "Section completed", duration_ms=duration_ms
                )

        total_seconds = time.perf_counter() - started

        self.current_section = "Telemetry"
        self.ctx.renderer.render_section("Telemetry")
        self.ctx.renderer.render_telemetry_cards(
            self.ctx.events, self.ctx.durations, total_seconds
        )

        self.current_section = "Summary"
        self.ctx.renderer.render_section("Summary")
        self.ctx.renderer.render_summary_table(self.ctx.events)
        self.ctx.renderer.render_issue_lists(self.ctx.events)

        writer = ArtifactWriter(self.ctx)
        writer.write_text_report(total_seconds)
        if self.ctx.json_report:
            writer.write_json_report(total_seconds)
        writer.write_log_capture()
        self.ctx.renderer.render_artifacts(self.ctx.artifacts)

        self.ctx.logger.info("Diagnostics finished in %.2fs", total_seconds)
        return 0

    def test_system_info(self) -> bool:
        rows = [
            ("OS", f"{platform.system()} {platform.release()} {platform.version()}"),
            ("Python", platform.python_version()),
            ("Architecture", platform.architecture()[0]),
            ("Processor", platform.processor() or "Unknown"),
        ]
        self.ctx.renderer.render_kv_table("Host Snapshot", rows)

        if platform.system() == "Windows":
            release = platform.release()
            try:
                release_major = int(release.split(".")[0])
            except ValueError:
                release_major = 10

            if release_major < 10:
                self._emit(
                    CheckStatus.WARN,
                    f"Windows {release} may have limited DirectX support. Windows 10+ is recommended.",
                )
            else:
                self._emit(
                    CheckStatus.PASS,
                    f"Windows {release} should support modern DirectX features.",
                )

        if platform.architecture()[0] == "32bit":
            self._emit(
                CheckStatus.WARN,
                "32-bit Python detected. 64-bit Python is recommended.",
            )
        else:
            self._emit(CheckStatus.PASS, "64-bit Python detected.")

        return True

    def check_dependencies(self) -> bool:
        dependencies = [
            ("numpy", "1.19.0", False),
            ("comtypes", "1.1.0", False),
            ("opencv-python", "4.5.0", True),
            ("pillow", "8.0.0", True),
            ("cupy", "11.0.0", True),
        ]

        all_required_ok = True
        for module_name, minimum, optional in dependencies:
            import_name = module_name.replace("-", "_")
            try:
                module = __import__(import_name)
                version = getattr(module, "__version__", "Unknown")
                message = f"{module_name} {version} installed"

                if module_name == "numpy" and version != "Unknown":
                    if _parse_version(version) < _parse_version(minimum):
                        self._emit(
                            CheckStatus.WARN,
                            f"{message} (minimum recommended: {minimum})",
                        )
                    else:
                        self._emit(CheckStatus.PASS, message)
                else:
                    self._emit(CheckStatus.PASS, message)
            except ImportError:
                if optional:
                    self._emit(
                        CheckStatus.WARN, f"Optional dependency missing: {module_name}"
                    )
                else:
                    self._emit(
                        CheckStatus.FAIL, f"Required dependency missing: {module_name}"
                    )
                    all_required_ok = False

        return all_required_ok

    def _read_dll_version(self, dll_name: str) -> str | None:
        try:
            get_size = ctypes.windll.version.GetFileVersionInfoSizeW
            get_info = ctypes.windll.version.GetFileVersionInfoW
            query_value = ctypes.windll.version.VerQueryValueW
            get_system_dir = ctypes.windll.kernel32.GetSystemDirectoryW

            buffer = ctypes.create_unicode_buffer(260)
            get_system_dir(buffer, len(buffer))
            dll_path = os.path.join(buffer.value, dll_name)

            size = get_size(dll_path, None)
            if size <= 0:
                return None

            blob = (ctypes.c_ubyte * size)()
            if not get_info(dll_path, 0, size, blob):
                return None

            info = ctypes.c_void_p()
            length = wintypes.UINT()
            if not query_value(
                blob,
                r"\VarFileInfo\Translation",
                ctypes.byref(info),
                ctypes.byref(length),
            ):
                return None

            struct_fmt = "hhhh"
            info_size = struct.calcsize(struct_fmt)
            info_data = ctypes.string_at(info.value, info_size)
            language, codepage = struct.unpack(struct_fmt, info_data)[:2]

            version_path = (
                f"\\StringFileInfo\\{language:04x}{codepage:04x}\\FileVersion"
            )
            if not query_value(
                blob, version_path, ctypes.byref(info), ctypes.byref(length)
            ):
                return None

            return ctypes.wstring_at(info.value, length.value - 1)
        except Exception:
            return None

    def test_directx_dlls(self) -> bool:
        dlls = ["d3d11.dll", "dxgi.dll", "d3dcompiler_47.dll"]
        essential = {"d3d11.dll", "dxgi.dll"}
        all_essential_available = True
        versions: list[tuple[str, str]] = []

        for dll in dlls:
            try:
                ctypes.windll.LoadLibrary(dll)
                self._emit(CheckStatus.PASS, f"{dll} is available")
                version = self._read_dll_version(dll)
                if version:
                    versions.append((dll, version))
            except Exception as error:
                if dll in essential:
                    self._emit(CheckStatus.FAIL, f"{dll} unavailable: {error}")
                    all_essential_available = False
                else:
                    self._emit(CheckStatus.WARN, f"{dll} unavailable: {error}")

        if versions:
            self.ctx.renderer.render_kv_table("Detected DLL versions", versions)

        if all_essential_available:
            self._emit(CheckStatus.PASS, "All essential DirectX DLLs are available.")
        else:
            self._emit(
                CheckStatus.FAIL, "One or more essential DirectX DLLs are missing."
            )

        return all_essential_available

    def test_gpu_info(self) -> bool:
        try:
            import wmi

            gpu_entries = list(wmi.WMI().Win32_VideoController())
            if not gpu_entries:
                self._emit(CheckStatus.WARN, "No GPUs reported by WMI.")
                return True

            known_vendor_found = False
            for index, gpu in enumerate(gpu_entries, start=1):
                name = str(getattr(gpu, "Name", "Unknown"))
                adapter_ram = getattr(gpu, "AdapterRAM", 0) or 0
                rows = [
                    ("Name", name),
                    ("Driver Version", str(getattr(gpu, "DriverVersion", "Unknown"))),
                    ("Driver Date", str(getattr(gpu, "DriverDate", "Unknown"))),
                    (
                        "Video Mode",
                        str(getattr(gpu, "VideoModeDescription", "Unknown")),
                    ),
                    ("Adapter RAM", f"{int(adapter_ram) / 1024 / 1024:.2f} MB"),
                    (
                        "Compatibility",
                        str(getattr(gpu, "AdapterCompatibility", "Unknown")),
                    ),
                ]
                self.ctx.renderer.render_kv_table(f"GPU {index}", rows)

                if any(vendor in name for vendor in ("NVIDIA", "AMD", "ATI", "Intel")):
                    known_vendor_found = True
                    self._emit(CheckStatus.PASS, f"Compatible GPU detected: {name}")
                else:
                    self._emit(CheckStatus.WARN, f"Unknown GPU vendor: {name}")

            if not known_vendor_found:
                self._emit(
                    CheckStatus.WARN, "No known DirectX-optimized GPU vendor detected."
                )

            return True

        except ImportError:
            self._emit(
                CheckStatus.WARN,
                "WMI package not installed; falling back to dxdiag parsing.",
            )
            return self._gpu_info_from_dxdiag()
        except Exception as error:
            self._capture_exception("Failed to gather GPU info", error)
            return False

    def _gpu_info_from_dxdiag(self) -> bool:
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".txt", delete=False
            ) as temporary_file:
                temp_path = Path(temporary_file.name)

            self._emit(
                CheckStatus.INFO, "Collecting dxdiag output for GPU fallback telemetry."
            )
            subprocess.run(["dxdiag", "/t", str(temp_path)], check=True)
            time.sleep(1.5)

            lines = temp_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            card_names: list[str] = []
            capture = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("Display Devices"):
                    capture = True
                    continue
                if capture and stripped.startswith("Card name"):
                    card_name = (
                        stripped.split(":", 1)[1].strip()
                        if ":" in stripped
                        else "Unknown"
                    )
                    card_names.append(card_name)

            if card_names:
                for card_name in card_names:
                    if any(
                        vendor in card_name
                        for vendor in ("NVIDIA", "AMD", "ATI", "Intel")
                    ):
                        self._emit(
                            CheckStatus.PASS, f"dxdiag GPU detected: {card_name}"
                        )
                    else:
                        self._emit(
                            CheckStatus.WARN,
                            f"dxdiag detected unknown GPU vendor: {card_name}",
                        )
                return True

            self._emit(
                CheckStatus.WARN, "dxdiag completed but no GPU card names were parsed."
            )
            return True

        except Exception as error:
            self._capture_exception("dxdiag GPU fallback failed", error)
            return False
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def test_dxgi_factory_creation(self) -> bool:
        def format_hresult(value: int) -> str:
            return f"0x{ctypes.c_uint32(value).value:08X}"

        from rapidshot._libs.dxgi import (
            DXGI_ADAPTER_DESC1 as RS_DXGI_ADAPTER_DESC1,
            DXGI_ERROR_NOT_FOUND as RS_DXGI_ERROR_NOT_FOUND,
            IDXGIAdapter1 as RS_IDXGIAdapter1,
            IDXGIFactory1 as RS_IDXGIFactory1,
        )

        factory_versions: list[dict[str, Any]] = [
            {
                "name": "DXGI 1.2+ (CreateDXGIFactory2)",
                "function": "CreateDXGIFactory2",
                "flags": 0,
            },
            {"name": "DXGI 1.1 (CreateDXGIFactory1)", "function": "CreateDXGIFactory1"},
            {"name": "DXGI 1.0 (CreateDXGIFactory)", "function": "CreateDXGIFactory"},
        ]

        factory_ptr = ctypes.c_void_p(0)
        selected_factory: dict[str, Any] | None = None

        for version in factory_versions:
            create_factory = getattr(ctypes.windll.dxgi, version["function"], None)
            if create_factory is None:
                self._emit(CheckStatus.WARN, f"{version['function']} export not found")
                continue

            try:
                create_factory.restype = comtypes.HRESULT
                if version["function"] == "CreateDXGIFactory2":
                    create_factory.argtypes = [
                        wintypes.UINT,
                        ctypes.POINTER(comtypes.GUID),
                        ctypes.POINTER(ctypes.c_void_p),
                    ]
                    hr = create_factory(
                        version["flags"],
                        ctypes.byref(RS_IDXGIFactory1._iid_),
                        ctypes.byref(factory_ptr),
                    )
                else:
                    create_factory.argtypes = [
                        ctypes.POINTER(comtypes.GUID),
                        ctypes.POINTER(ctypes.c_void_p),
                    ]
                    hr = create_factory(
                        ctypes.byref(RS_IDXGIFactory1._iid_), ctypes.byref(factory_ptr)
                    )

                if hr == 0 and factory_ptr.value:
                    selected_factory = version
                    self._emit(CheckStatus.PASS, f"Created {version['name']}")
                    break

                self._emit(
                    CheckStatus.WARN,
                    f"{version['name']} failed with HRESULT {format_hresult(hr)}",
                )
            except Exception as error:
                self._emit(CheckStatus.WARN, f"{version['name']} raised error: {error}")

        if selected_factory is None:
            self._emit(CheckStatus.FAIL, "Failed to create any DXGI factory.")
            return False

        factory = ctypes.cast(factory_ptr, ctypes.POINTER(RS_IDXGIFactory1))
        adapter_rows: list[tuple[str, str, str, str]] = []
        index = 0

        try:
            while True:
                adapter = ctypes.POINTER(RS_IDXGIAdapter1)()
                try:
                    factory.EnumAdapters1(index, ctypes.byref(adapter))
                except comtypes.COMError as error:
                    hresult = error.args[0] if error.args else None
                    if (
                        hresult is not None
                        and ctypes.c_int32(hresult).value
                        == ctypes.c_int32(RS_DXGI_ERROR_NOT_FOUND).value
                    ):
                        break
                    self._emit(
                        CheckStatus.WARN,
                        f"EnumAdapters1 stopped at index {index}: {error}",
                    )
                    break

                if not bool(adapter):
                    self._emit(
                        CheckStatus.WARN,
                        f"EnumAdapters1 returned null adapter at index {index}",
                    )
                    break

                try:
                    descriptor = RS_DXGI_ADAPTER_DESC1()
                    adapter.GetDesc1(ctypes.byref(descriptor))
                    adapter_rows.append(
                        (
                            str(index),
                            descriptor.Description,
                            f"{int(descriptor.DedicatedVideoMemory) / (1024 * 1024):.2f} MB",
                            str(descriptor.VendorId),
                        )
                    )
                except Exception as error:
                    self._emit(
                        CheckStatus.WARN, f"Failed to describe adapter {index}: {error}"
                    )
                finally:
                    release_com_ptr(adapter)

                index += 1

        except Exception as error:
            self._capture_exception("Error during adapter enumeration", error)
            return False
        finally:
            release_com_ptr(factory)

        if adapter_rows:
            table = Table(
                title="DXGI adapters", box=box.SIMPLE_HEAVY, header_style="bold cyan"
            )
            table.add_column("Index", justify="right")
            table.add_column("Description")
            table.add_column("VRAM")
            table.add_column("Vendor ID")
            for row in adapter_rows:
                table.add_row(*row)
            self.ctx.console.print(table)
            self._emit(
                CheckStatus.PASS, f"Found {len(adapter_rows)} graphics adapters."
            )
            return True

        self._emit(CheckStatus.FAIL, "No graphics adapters discovered by DXGI.")
        return False

    def test_d3d11_device_creation(self) -> bool:
        D3D_DRIVER_TYPE_HARDWARE = 1
        D3D_DRIVER_TYPE_WARP = 5

        D3D_FEATURE_LEVEL_11_1 = 0xB100
        D3D_FEATURE_LEVEL_11_0 = 0xB000
        D3D_FEATURE_LEVEL_10_1 = 0xA100
        D3D_FEATURE_LEVEL_10_0 = 0xA000

        D3D11_CREATE_DEVICE_BGRA_SUPPORT = 0x20
        D3D11_SDK_VERSION = 7

        from rapidshot._libs.d3d11 import ID3D11Device as RS_ID3D11Device
        from rapidshot._libs.d3d11 import ID3D11DeviceContext as RS_ID3D11DeviceContext

        try:
            d3d11_dll = ctypes.windll.d3d11
            create_device = d3d11_dll.D3D11CreateDevice
            create_device.restype = comtypes.HRESULT
            create_device.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.POINTER(ctypes.c_uint),
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_uint),
                ctypes.POINTER(ctypes.c_void_p),
            ]

            feature_levels = [
                D3D_FEATURE_LEVEL_11_1,
                D3D_FEATURE_LEVEL_11_0,
                D3D_FEATURE_LEVEL_10_1,
                D3D_FEATURE_LEVEL_10_0,
            ]
            feature_levels_array = (ctypes.c_uint * len(feature_levels))(
                *feature_levels
            )

            device_ptr = ctypes.c_void_p()
            context_ptr = ctypes.c_void_p()
            feature_level = ctypes.c_uint(0)

            hr = create_device(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                None,
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                feature_levels_array,
                len(feature_levels),
                D3D11_SDK_VERSION,
                ctypes.byref(device_ptr),
                ctypes.byref(feature_level),
                ctypes.byref(context_ptr),
            )

            if hr != 0:
                self._emit(
                    CheckStatus.WARN,
                    f"Hardware device creation failed ({hr:#010x}), retrying with WARP",
                )
                device_ptr = ctypes.c_void_p()
                context_ptr = ctypes.c_void_p()
                feature_level = ctypes.c_uint(0)

                hr = create_device(
                    None,
                    D3D_DRIVER_TYPE_WARP,
                    None,
                    D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                    feature_levels_array,
                    len(feature_levels),
                    D3D11_SDK_VERSION,
                    ctypes.byref(device_ptr),
                    ctypes.byref(feature_level),
                    ctypes.byref(context_ptr),
                )

            if hr == 0 and device_ptr.value and context_ptr.value:
                major = (feature_level.value >> 12) & 0xF
                minor = (feature_level.value >> 8) & 0xF
                self._emit(
                    CheckStatus.PASS,
                    f"D3D11 device created successfully at feature level {major}.{minor}",
                )

                device = ctypes.cast(device_ptr, ctypes.POINTER(RS_ID3D11Device))
                context = ctypes.cast(
                    context_ptr, ctypes.POINTER(RS_ID3D11DeviceContext)
                )
                release_com_ptr(context)
                release_com_ptr(device)
                return True

            self._emit(
                CheckStatus.FAIL,
                f"D3D11 device creation failed: 0x{ctypes.c_uint32(hr).value:08X}",
            )
            return False

        except Exception as error:
            self._capture_exception("D3D11 device creation error", error)
            return False

    def test_desktop_duplication(self) -> bool:
        try:
            from rapidshot._libs.dxgi import IDXGIFactory1 as RS_IDXGIFactory1

            create_factory = ctypes.windll.dxgi.CreateDXGIFactory1
            create_factory.restype = comtypes.HRESULT
            create_factory.argtypes = [
                ctypes.POINTER(comtypes.GUID),
                ctypes.POINTER(ctypes.c_void_p),
            ]

            factory_ptr = ctypes.c_void_p(0)
            hr = create_factory(
                ctypes.byref(RS_IDXGIFactory1._iid_), ctypes.byref(factory_ptr)
            )
            if hr != 0 or not factory_ptr.value:
                self._emit(
                    CheckStatus.FAIL,
                    f"Failed to create DXGI factory for duplication probe: {hr:#010x}",
                )
                return False

            self._emit(
                CheckStatus.PASS,
                "Desktop duplication prerequisite (DXGI factory) is operational.",
            )
            self._emit(
                CheckStatus.INFO,
                "Full DuplicateOutput probing is intentionally skipped in this lightweight diagnostic path.",
                details="Use runtime capture workflows for end-to-end duplication validation.",
            )

            factory = ctypes.cast(factory_ptr, ctypes.POINTER(RS_IDXGIFactory1))
            release_com_ptr(factory)
            return True

        except Exception as error:
            self._capture_exception("Desktop duplication probe error", error)
            return False

    def test_rapidshot_imports(self) -> bool:
        all_good = True
        try:
            import rapidshot

            self._emit(
                CheckStatus.PASS, f"Imported rapidshot from {rapidshot.__file__}"
            )
            self._emit(
                CheckStatus.PASS,
                f"RapidShot version: {getattr(rapidshot, '__version__', 'Unknown')}",
            )

            expected_attrs = [
                "create",
                "device_info",
                "output_info",
                "clean_up",
                "ScreenCapture",
            ]
            missing_attrs = [
                attr for attr in expected_attrs if not hasattr(rapidshot, attr)
            ]
            if missing_attrs:
                self._emit(
                    CheckStatus.FAIL,
                    f"Missing rapidshot attributes: {', '.join(missing_attrs)}",
                )
                all_good = False
            else:
                self._emit(CheckStatus.PASS, "Core rapidshot attributes are present.")

            try:
                import rapidshot.core

                self._emit(CheckStatus.PASS, "Imported rapidshot.core successfully.")
                expected_core = ["Device", "Output", "Duplicator", "StageSurface"]
                missing_core = [
                    name for name in expected_core if not hasattr(rapidshot.core, name)
                ]
                if missing_core:
                    self._emit(
                        CheckStatus.FAIL,
                        f"Missing core symbols: {', '.join(missing_core)}",
                    )
                    all_good = False
                else:
                    self._emit(
                        CheckStatus.PASS, "rapidshot.core exports expected symbols."
                    )
            except Exception as error:
                self._emit(
                    CheckStatus.FAIL, f"Failed importing rapidshot.core: {error}"
                )
                all_good = False

            try:
                from rapidshot._libs import d3d11, dxgi

                self._emit(
                    CheckStatus.PASS,
                    "Imported rapidshot._libs.d3d11 and rapidshot._libs.dxgi",
                )
                if not hasattr(d3d11, "D3D_FEATURE_LEVEL_11_0"):
                    self._emit(
                        CheckStatus.WARN,
                        "D3D_FEATURE_LEVEL_11_0 missing in d3d11 module",
                    )
                if not hasattr(dxgi, "IDXGIFactory1"):
                    self._emit(CheckStatus.WARN, "IDXGIFactory1 missing in dxgi module")
            except Exception as error:
                self._emit(
                    CheckStatus.FAIL,
                    f"Failed importing DirectX library wrappers: {error}",
                )
                all_good = False

            return all_good

        except Exception as error:
            self._capture_exception("rapidshot import failed", error)
            return False

    def test_rapidshot_device_info(self) -> bool:
        try:
            import rapidshot

            device_info = rapidshot.device_info()
            rendered = str(device_info).strip()
            if rendered:
                self._emit(CheckStatus.PASS, "rapidshot.device_info() returned data")
                self.ctx.renderer.render_block("Device info", rendered, style="cyan")
            else:
                self._emit(
                    CheckStatus.WARN, "rapidshot.device_info() returned empty data"
                )
            return True
        except Exception as error:
            self._capture_exception("rapidshot.device_info() failed", error)
            return False

    def test_rapidshot_output_info(self) -> bool:
        try:
            import rapidshot

            output_info = rapidshot.output_info()
            rendered = str(output_info).strip()
            if rendered:
                self._emit(CheckStatus.PASS, "rapidshot.output_info() returned data")
                self.ctx.renderer.render_block("Output info", rendered, style="blue")
            else:
                self._emit(
                    CheckStatus.WARN, "rapidshot.output_info() returned empty data"
                )
            return True
        except Exception as error:
            self._capture_exception("rapidshot.output_info() failed", error)
            return False

    def test_rapidshot_create(self) -> bool:
        try:
            import rapidshot

            screen = rapidshot.create(output_color="BGR")
            if screen is None:
                self._emit(CheckStatus.FAIL, "rapidshot.create() returned None")
                return False

            self._emit(
                CheckStatus.PASS, "rapidshot.create() returned a capture instance"
            )
            ok = True

            try:
                frame = screen.grab()
                if frame is not None:
                    height, width, channels = frame.shape
                    self._emit(
                        CheckStatus.PASS,
                        f"Initial frame captured: {width}x{height}x{channels}",
                    )
                else:
                    self._emit(CheckStatus.WARN, "Initial frame capture returned None")
            except Exception as error:
                self._emit(
                    CheckStatus.FAIL, f"Error during initial frame capture: {error}"
                )
                ok = False

            time.sleep(0.1)
            try:
                delayed_frame = screen.grab()
                if delayed_frame is not None:
                    self._emit(CheckStatus.PASS, "Delayed frame capture succeeded")
                else:
                    self._emit(CheckStatus.WARN, "Delayed frame capture returned None")
            except Exception as error:
                self._emit(
                    CheckStatus.FAIL, f"Error during delayed frame capture: {error}"
                )
                ok = False

            try:
                screen.release()
                self._emit(CheckStatus.PASS, "Capture resources released")
            except Exception as error:
                self._emit(
                    CheckStatus.WARN,
                    f"Failed releasing capture resources cleanly: {error}",
                )

            try:
                rapidshot.clean_up()
                self._emit(CheckStatus.PASS, "rapidshot.clean_up() completed")
            except Exception as error:
                self._emit(CheckStatus.WARN, f"rapidshot.clean_up() raised: {error}")

            return ok

        except Exception as error:
            self._capture_exception("rapidshot.create() workflow failed", error)
            return False


def _create_logger(
    console: Console, log_path: Path, verbosity: VerbosityLevel
) -> logging.Logger:
    _install_trace_level()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level_by_verbosity = {
        VerbosityLevel.INFO: logging.INFO,
        VerbosityLevel.DEBUG: logging.DEBUG,
        VerbosityLevel.TRACE: TRACE_LEVEL_NUM,
    }
    logger_level = level_by_verbosity[verbosity]

    logger = logging.getLogger("rapidshot_diagnostics")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logger_level)

    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=verbosity >= VerbosityLevel.TRACE,
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    rich_handler.setLevel(logger_level)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    file_handler.setLevel(TRACE_LEVEL_NUM)

    logger.addHandler(rich_handler)
    logger.addHandler(file_handler)
    return logger


def teardown_logging(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        if hasattr(handler, "flush"):
            handler.flush()
        if hasattr(handler, "close"):
            handler.close()
    logger.handlers.clear()


def create_run_context(
    verbosity: VerbosityLevel, output_dir: Path, json_report: bool
) -> RunContext:
    output_path = output_dir.expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    console = Console(markup=True)
    log_path = output_path / DEFAULT_LOG_FILE
    logger = _create_logger(console, log_path, verbosity)

    ctx = RunContext(
        console=console,
        logger=logger,
        verbosity=verbosity,
        output_dir=output_path,
        json_report=json_report,
        log_path=log_path,
    )
    ctx.renderer = DiagnosticRenderer(console, verbosity)
    ctx.renderer.render_file_spotlight("Log", log_path, "Opened")
    return ctx


def execute_run(ctx: RunContext) -> int:
    runner = DiagnosticRunner(ctx)
    return runner.run()


def render_saved_summary(ctx: RunContext) -> int:
    renderer = ctx.renderer
    renderer.render_banner()
    renderer.render_section("Saved Summary")

    summary_path = ctx.output_dir / DEFAULT_RESULTS_FILE
    log_path = ctx.output_dir / DEFAULT_LOG_FILE

    if summary_path.exists():
        summary_text = summary_path.read_text(
            encoding="utf-8", errors="replace"
        ).strip()
        renderer.render_block("Text summary", summary_text or "(empty)", style="green")
    else:
        renderer.render_event(
            CheckEvent(
                section="Saved Summary",
                status=CheckStatus.WARN,
                message=f"Summary file not found: {summary_path}",
            )
        )

    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        preview = "\n".join(lines[-40:]) if lines else "(empty)"
        renderer.render_block("Log tail (last 40 lines)", preview, style="blue")
    else:
        renderer.render_event(
            CheckEvent(
                section="Saved Summary",
                status=CheckStatus.WARN,
                message=f"Log file not found: {log_path}",
            )
        )

    renderer.render_file_spotlight("Summary view", summary_path, "Rendered")
    return 0


app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    rich_markup_mode="rich",
    help="""Run polished DirectX and RapidShot diagnostics with telemetry-rich terminal output.""",
    epilog="Use -v / -vv / -vvv to increase rendering and trace detail.",
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity. Repeat for deeper diagnostics.",
        rich_help_panel="Display and verbosity",
    ),
    output_dir: Path = typer.Option(
        Path("."),
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Directory where diagnostic artifacts are written.",
        rich_help_panel="Output artifacts",
    ),
    json_report: bool = typer.Option(
        False,
        "--json-report/--no-json-report",
        help="Write structured JSON telemetry alongside text and log reports.",
        rich_help_panel="Output artifacts",
    ),
) -> None:
    """Configure diagnostic execution context and run diagnostics by default."""
    if ctx.resilient_parsing:
        return

    verbosity = resolve_verbosity_level(verbose)
    run_ctx = create_run_context(verbosity, output_dir, json_report)
    ctx.obj = run_ctx

    if ctx.invoked_subcommand is None:
        try:
            code = execute_run(run_ctx)
        finally:
            teardown_logging(run_ctx.logger)
        raise typer.Exit(code)


@app.command("run")
def run_command(ctx: typer.Context) -> None:
    """Run the complete diagnostics suite."""
    run_ctx = ctx.obj
    if not isinstance(run_ctx, RunContext):
        raise typer.BadParameter("Run context is unavailable")

    try:
        code = execute_run(run_ctx)
    finally:
        teardown_logging(run_ctx.logger)
    raise typer.Exit(code)


@app.command("show-summary")
def show_summary_command(ctx: typer.Context) -> None:
    """Render saved diagnostic summary and recent log output in styled panels."""
    run_ctx = ctx.obj
    if not isinstance(run_ctx, RunContext):
        raise typer.BadParameter("Run context is unavailable")

    try:
        code = render_saved_summary(run_ctx)
    finally:
        teardown_logging(run_ctx.logger)
    raise typer.Exit(code)


if __name__ == "__main__":
    app()
