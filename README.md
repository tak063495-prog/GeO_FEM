# GeoFEM 2D

[English](README.md) | [日本語](README.ja.md)

GeoFEM is a 2D finite-element analysis application with a PySide6 GUI and a command-line solver. It is intended for plane-strain geotechnical analysis, including elastic/plastic materials, strength-reduction method (SRM), consolidation/u-p analysis, large deformation, axisymmetric and dynamic workflows.

The current baseline is `v0.1.0`. The repository contains source code, tests, example YAML files, input YAML files, design notes, and a classified development-failure archive.

## What Is Included

- 2D `TRI3`、`TRI6`、`QUAD4`、`QUAD8` elements
- `FULL`、`SRI`、`B-bar` integration
- Plane-strain static/nonlinear analysis and large deformation paths
- Elastic and plastic material models, including Drucker-Prager and Mohr-Coulomb paths
- SRM FOS search with adaptive bracket, strict factor tolerance, diagnostics, lookahead, cancellation and worker policies
- Consolidation, u-p, axisymmetric, dynamic, Riks, MPC/Lagrange and interface workflows
- CSV, VTK, JSON and deferred HTML/PDF result artifacts
- GUI preview caching, result summary view, read-only result reference and large-mesh display policies
- NumPy/SciPy sparse assembly and Numba acceleration for selected hot paths

## Important Scope And Accuracy Note

This project does not claim commercial GeoFEAS/VGFlow feature parity or identical published-paper results. In particular, an SRM result can be marked `material_fallback_evidence` with limited confidence when regularization or a constitutive fallback was used. `converged=true` alone is not a proof that the FOS is fully validated.

For engineering review, retain the FOS value together with the stable/failed bracket, failure class, factor tolerance, material fallback counters, convergence diagnostics, mesh quality, YAML snapshot and `analysis_log`.

## Quick Start

### CLI

```powershell
python -m geofem_app.cli solve examples\plane_strain_quad4_bbar.yaml --out runs\sample_quad4
```

For SRM batch execution, the worker policy can be selected explicitly:

```powershell
python -m geofem_app.cli solve examples\sustainability_2024_case2_quad4_sri_auto_srm.yaml --out runs\case2 --srm-parallel-policy auto --srm-workers auto
```

Typical output includes `summary.json`, `run.log`, `analysis_log.json`, CSV files and VTK results. Generated results belong in a run directory or release archive, not in the source commit.

### GUI

```powershell
run_gui.bat
```

The GUI uses the same solver layer as the CLI. During a long analysis, the main supported action is cooperative cancellation; result reports and result views are generated lazily when requested.

## Validation

Run the full Python regression suite:

```powershell
python -W error::ResourceWarning -m unittest discover -s tests -p "test_*.py" -v
```

Useful checks:

```powershell
python -m geofem_app.cli doctor --out runs\startup_check
python -m geofem_app.cli api-contracts --out runs\api_contracts
python -m geofem_app.cli benchmarks --out runs\geofem_standard_benchmarks
```

The repository also contains Case1-4 SRM YAMLs under `examples/`. Full Case1-4 runs are hardware- and configuration-dependent and should be recorded with the same YAML, commit, worker policy and environment.

## Documentation

- [Developer guide (Japanese)](docs/DEVELOPER_GUIDE_JA.md)
- [Installation and startup (Japanese)](docs/INSTALL_STARTUP_JA.md)
- [API contracts (Japanese)](docs/API_CONTRACTS_JA.md)
- [Performance review (Japanese)](docs/PERFORMANCE_REVIEW_JA.md)
- [Case2/4 Mohr-Coulomb Numba validation (Japanese)](docs/CASE2_CASE4_MC_NUMBA_VALIDATION_JA.md)
- [Development history and failure classification (Japanese)](development_history_failures_20260821/DEVELOPMENT_FAILURE_HISTORY_JA.md)
- [Deliverable manifest](docs/DELIVERABLE_MANIFEST_20260821.md)
- [Changelog](CHANGELOG.md)
- [Project Wiki](https://github.com/tak063495-prog/GeO_FEM/wiki)

## Version And Contribution Workflow

### Clone And Update On Another PC

Clone the repository:

```powershell
git clone https://github.com/tak063495-prog/GeO_FEM.git
cd GeO_FEM
```

Update an existing working copy:

```powershell
git switch master
git pull --ff-only origin master
```

If the working copy has local changes, commit or stash them before pulling. Do not use `reset --hard` for a routine update.

The repository remote may be named `GeOFEM` instead of the conventional `origin`:

```powershell
git clone -o GeOFEM https://github.com/tak063495-prog/GeO_FEM.git
cd GeO_FEM
git pull --ff-only GeOFEM master
```

To rename an existing remote:

```powershell
git remote rename origin GeOFEM
git pull --ff-only GeOFEM master
```

1. Create a feature branch from `master`.
2. Make a focused change and update the relevant tests/docs.
3. Run targeted tests, then the full regression suite when the change crosses solver, GUI or output boundaries.
4. Record performance changes with cold/warm, solver/post/I/O and worker=1/auto distinctions.
5. Commit with a message that states the user-visible or numerical effect.
6. Push the branch and review the diff before merging.
7. Update `geofem_app/__init__.py`, create a matching `vX.Y.Z` tag, and attach executable or large run archives as GitHub Release assets rather than committing build directories.

The initial baseline commit is `dc7249e18e7fb4e7730f9282bdce590a43727fca`, tagged `v0.1.0`.

## License And External Tools

The GeoFEM source repository is distributed under the [MIT License](LICENSE). Dependencies, GUI libraries, Numba/LLVM components, and external DWG conversion tools retain their own licenses and notices. Native DWG decoding is not claimed; the documented external conversion policy applies.
