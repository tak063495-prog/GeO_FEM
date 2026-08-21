# GeoFEM 2D MVP

GeoFEM is now a 2D plane-strain FEM application. The former 3D/v26 backend and
3D delegation path have been removed.

## Current Solver

- Dimension: 2D plane strain only
- Elements: `TRI3`, `TRI6`, `QUAD4`, `QUAD8`
- Integration: `FULL`, `SRI`, `B-bar`
- Core features: elastic/plastic materials, SRM, geostatic/K0,
  excavation/death, u-p consolidation, Riks, interfaces, iterative solver
  controls, CSV/VTK/HTML/JSON outputs
- Acceleration: NumPy/SciPy sparse assembly with required Numba kernels for
  hot QUAD4 paths

Inputs that request a 3D/v26 backend now fail fast. Plane-strain outputs still
include `eps_z=0` and `sigma_z` because those are part of the 2D plane-strain
stress state, not a 3D solver path.

## CLI

```powershell
cd C:\Users\link_\Downloads\WORK\GeoFEM
python -m geofem_app.cli solve examples\plane_strain_quad4_bbar.yaml --out runs\sample_quad4
```

Generated outputs include:

- `summary.json`
- `run.log`
- `Stage-1/displacements.csv`
- `Stage-1/element_stress.csv`
- `Stage-1/reactions.csv`
- `Stage-1/results.vtk`
- `Stage-1/report.html`

## GUI

For the source distribution, `run_gui.bat` creates a local `.venv` and installs
the Python 3.12 dependency set pinned by `requirements-lock.txt` on first launch
if the GUI dependencies are missing:

```powershell
run_gui.bat
```

The GUI runs the same 2D CLI solver.

## Japanese User And Developer Docs

- `docs/INSTALL_STARTUP_JA.md`: Windows setup, startup check, GUI packaging verification.
- `docs/TUTORIAL_BASIC_JA.md`: first-run tutorial from sample creation to result review.
- `docs/API_CONTRACTS_JA.md`: internal data contracts between solver, mesh, materials, output, and GUI.
- `docs/DEVELOPER_GUIDE_JA.md`: module responsibilities, test layout, and performance workflow.

Machine-readable startup and contract checks:

```powershell
python -m geofem_app.cli doctor --out runs\startup_check
python -m geofem_app.cli api-contracts --out runs\api_contracts
```

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```
