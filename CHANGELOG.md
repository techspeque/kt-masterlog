# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Run registry** — every `optimize()` call now writes a per-run JSON
  manifest to `~/.kt-masterlog/runs/<run_id>.json` (override with
  `$KT_MASTERLOG_REGISTRY_DIR`). The manifest contains project name,
  master CSV path, writing PID, hostname, start/end timestamps, and
  status (`running` / `completed` / `failed`). This is the cross-package
  contract that lets sister tools like
  [kt-masterviz](https://github.com/techspeque/kt-masterviz)
  auto-discover runs without the user typing CSV paths.
- `TunerConfig.register_run` (default `True`) — opt-out for sandboxed
  or CI environments where filesystem side effects are undesirable.
- New `kt_masterlog.registry` module with `register_run()` and
  `update_run_status()` for advanced / programmatic use.
- Test isolation: `tests/conftest.py` autouse fixture redirects the
  registry to `tmp_path` so the suite never writes to a real
  `~/.kt-masterlog/`.

### Schema

- Registry manifests use `schema_version: 1`. Readers are expected to
  skip unknown versions; this version is stable for the 0.x line.

## [0.1.0] - 2026-05-15

Initial release.

### Added

- **`MasterEpochLogger`** — Keras callback that appends one CSV row per
  epoch per trial. Thread-safe via per-path file locks; field order
  locked on first write.
- **`make_logging_tuner()`** — dynamically subclasses any KerasTuner
  tuner class to inject `MasterEpochLogger` into every trial. Cached
  per base class.
- **`TunerConfig`** — JSON-roundtrippable dataclass holding the full
  tuning configuration (strategy, objective, search epochs,
  early-stopping, ReduceLROnPlateau, extra fields/callbacks).
- **`optimize()`** — single-call orchestrator returning a `TuningResult`
  with the best model, best hyperparameters, elapsed time, master CSV
  path, and the underlying tuner.
- **`STRATEGY_REGISTRY`** — pluggable strategy lookup; built-ins are
  `bayesian`, `hyperband`, `random`. Custom tuners drop in.
- Tests at 98% line coverage on Python 3.12, full CI on every commit,
  tag-triggered PyPI release via trusted publishing.

[Unreleased]: https://github.com/techspeque/kt-masterlog/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/techspeque/kt-masterlog/releases/tag/v0.1.0
