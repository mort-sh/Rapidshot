# Diagnostic Console Patterns

## Context
This diagnostic UI standard prioritizes signal density and readability. The default experience is intentionally compact, with richer detail available through verbosity.

## Core Principles
- One output pipeline: all user-facing output is rendered by `DiagnosticRenderer`.
- No duplicate console lines: file logging remains comprehensive, while console output stays signal-first.
- Default output favors summaries over event floods.
- Panels are sparse and intentional, not the default container for every message.

## Verbosity Contract
- `-v/--verbose` uses `count=True`.
- `0` (`INFO`): compact view.
- `1` (`DEBUG`): includes pass-level events and richer detail blocks.
- `2+` (`TRACE`): full details and tracebacks with locals.

## Default Compact Rules
- Always show section headers.
- Always show warnings and failures immediately.
- Hide routine pass/info event lines by default.
- Emit one section summary line:
- `SectionName  pass=X warn=Y fail=Z  duration=...ms`
- Keep key tables where they materially help interpretation.

## Panel Policy
- Allowed high-value panel usage:
- startup banner
- exception/failure detail
- final warnings/failures summary
- Routine data and artifact updates should use compact lines/tables first.

## Telemetry and Summary
- Telemetry is rendered as a compact strip:
- pass/warn/fail counts
- check count
- total duration
- sparkline of section durations
- Warning/failure lists are capped in console output.
- Overflow is represented as `+N more` to prevent wall-of-text output.

## Artifact Visibility
- Default mode prints one concise completion line per artifact.
- Debug and above can show lifecycle detail (`Writing`, `Saved`, `Finalized`).
- Artifact table remains available at the end for auditability.

## Reuse Guidance
- Keep rendering concerns in `DiagnosticRenderer`.
- Keep file persistence concerns in `ArtifactWriter`.
- Keep pure helpers (`resolve_verbosity_level`, `build_sparkline`) unit-testable and reusable.
