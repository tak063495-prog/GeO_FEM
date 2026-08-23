# Contributing to GeoFEM

Thank you for helping improve GeoFEM. The public project language is English; Japanese material is welcome when it improves the engineering record.

## Before You Start

- Read the [README](README.md), the [developer guide](docs/DEVELOPER_GUIDE_JA.md), and the relevant validation/performance notes.
- For a numerical change, identify the affected analysis path, element/integration rule, material model, and expected accuracy impact.
- Do not include generated run directories, private input data, credentials, or large result archives in a source pull request.

## Development Workflow

1. Create a focused branch from `master`.
2. Make the smallest change that addresses the issue.
3. Add or update targeted tests.
4. Run the full regression suite when solver, GUI, output, or public API behavior changes.
5. Record benchmark changes with cold/warm state, worker count, solver/post/I/O split, and the commit.
6. For SRM changes, include the stable/failed bracket, `factor_tol`, convergence diagnostics, fallback evidence, and the YAML used.
7. Open a pull request with a concise summary, test commands, and known limitations.

## Numerical Changes

Performance must not silently trade away result quality. Prefer a feature flag or an explicit policy when behavior can change. Compare convergence status, FOS bracket, residual diagnostics, plasticity/fallback counters, mesh-quality indicators, and representative output fields before claiming a speedup.

## Commit and Pull Request Notes

Use a clear commit subject, for example `perf: cache SRM reduced matrix extraction`. Keep unrelated formatting or generated files out of the change. The pull request checklist in `.github/PULL_REQUEST_TEMPLATE.md` is the minimum review record.

## Questions and Bugs

Use the GitHub issue forms for reproducible bugs and feature requests. Include the OS, Python version, dependency versions, commit, YAML or a minimized equivalent, command, and relevant log excerpt. Never post secrets or confidential project data.
