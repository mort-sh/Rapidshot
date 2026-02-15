# Diagnostic Console Patterns

## Context
Task Orchestrator integration was attempted for this workstream, but the MCP transport returned `Transport closed`. This document mirrors the same design decisions so implementation guidance remains durable and auditable in-repo.

## Styling Principles
- Use a single rendering pipeline (`DiagnosticRenderer`) for all user-facing output.
- Avoid bare `print()` calls; route output through Rich renderables and logger handlers.
- Use semantic status styling:
- `PASS` as green
- `WARN` as yellow
- `FAIL` as red
- `INFO` as cyan
- Use section rules and cards to keep scanability high during long diagnostic runs.

## Verbosity Contract
- `-v/--verbose` uses `count=True`.
- Verbosity mapping:
- `0` -> `INFO`
- `1` -> `DEBUG`
- `2+` -> `TRACE`
- At `DEBUG` and above, render richer details and per-section completion timing.
- At `TRACE`, include exception tracebacks with local variables.

## Telemetry Cards and Sparkline Rules
- Render telemetry in cards for:
- pass count
- warning count
- failure count
- execution telemetry
- Use sparkline-style bars based on `▁▂▃▄▅▆▇█`.
- Build bars from per-check durations (ms), normalized across min/max.
- Always show total diagnostic duration and check count.

## File Write Highlighting Standard
- Every artifact write must be visually spotlighted before and after write.
- Required artifact types:
- `rapidshot_diagnostics.log`
- `rapidshot_diagnostics_results.txt`
- Optional:
- `rapidshot_diagnostics_results.json` when `--json-report` is enabled.
- Spotlight cards include:
- artifact type
- full path
- phase (`Writing`, `Saved`, `Finalized`)
- byte size on completion

## Failure Rendering Standard
- Failures must emit both:
- a styled failure status line
- a dedicated exception panel
- At `TRACE` verbosity, render Rich traceback with local variables.
- Summary must always include explicit lists of failed checks and warnings.

## Reuse Guidance
- Keep console rendering in a renderer class and file persistence in an artifact writer class.
- Keep diagnostic probes focused on checks and event emission only.
- Make helper functions (`resolve_verbosity_level`, `build_sparkline`) unit-testable for reuse across CLIs.
