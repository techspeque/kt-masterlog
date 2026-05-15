# Security Policy

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report them privately via [GitHub Security
Advisories](https://github.com/techspeque/kt-masterlog/security/advisories/new).
That channel is structured, traceable, and lets us coordinate a fix and
disclosure with you before the issue becomes public.

Please include:

- A description of the issue
- Steps to reproduce or a minimal proof-of-concept
- Affected version(s)

## Supported versions

This project is pre-1.0; only the latest released version on PyPI receives
security fixes.

## Scope

kt-masterlog is a thin orchestration layer over KerasTuner. Vulnerabilities
in TensorFlow, KerasTuner, scikit-learn, or other transitive dependencies
should be reported upstream to those projects.

Issues in our own code are in scope — examples:

- CSV writing or file-handling bugs in `MasterEpochLogger`
- The dynamic tuner subclassing in `make_logging_tuner`
- `optimize()` orchestrator logic
- Anything in `scripts/` or `.github/workflows/` that could affect
  release integrity
