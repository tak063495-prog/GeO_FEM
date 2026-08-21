"""Built-in sample configurations used by CLI and GUI."""

from __future__ import annotations

from typing import Any


def plane_strain_quad4_sample(*, integration: str = "B-bar") -> dict[str, Any]:
    """Small 2D cantilever-like elastic sample."""

    return {
        "analysis": {"dimension": "2D", "type": "static_plane_strain", "unit_system": "m-kN"},
        "mesh": {
            "generator": "rectangle",
            "x_range": [0.0, 10.0],
            "y_range": [0.0, 2.0],
            "nx": 8,
            "ny": 2,
            "element_type": "QUAD4",
            "integration": integration,
            "material": "soil",
        },
        "materials": {
            "soil": {"model": "elastic", "E": 50000.0, "nu": 0.33, "gamma": 18.0},
        },
        "boundary_conditions": [
            {"set": "left", "ux": 0.0, "uy": 0.0},
            {"set": "bottom", "uy": 0.0},
        ],
        "loads": [
            {"edge": ["9", "18"], "ty": -50.0},
        ],
        "output": {"directory": "runs/sample_2d"},
    }


def plane_strain_patch_sample(element_type: str = "QUAD4", integration: str = "FULL") -> dict[str, Any]:
    """One-element linear-displacement patch sample.

    The prescribed displacement field is:
        ux = 0.001*x + 0.0002*y
        uy = -0.0003*y + 0.0004*x
    """

    etype = element_type.upper()
    if etype == "TRI3":
        nodes = {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [0.0, 1.0]}
        elements = [{"id": "1", "type": "TRI3", "nodes": ["1", "2", "3"], "material": "soil", "integration": integration}]
    elif etype == "TRI6":
        nodes = {
            "1": [0.0, 0.0],
            "2": [1.0, 0.0],
            "3": [0.0, 1.0],
            "4": [0.5, 0.0],
            "5": [0.5, 0.5],
            "6": [0.0, 0.5],
        }
        elements = [{"id": "1", "type": "TRI6", "nodes": ["1", "2", "3", "4", "5", "6"], "material": "soil", "integration": integration}]
    elif etype == "QUAD4":
        nodes = {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]}
        elements = [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil", "integration": integration}]
    elif etype == "QUAD8":
        nodes = {
            "1": [0.0, 0.0],
            "2": [1.0, 0.0],
            "3": [1.0, 1.0],
            "4": [0.0, 1.0],
            "5": [0.5, 0.0],
            "6": [1.0, 0.5],
            "7": [0.5, 1.0],
            "8": [0.0, 0.5],
        }
        elements = [{"id": "1", "type": "QUAD8", "nodes": ["1", "2", "3", "4", "5", "6", "7", "8"], "material": "soil", "integration": integration}]
    else:
        raise ValueError(f"unsupported sample element type: {element_type}")

    bcs = []
    for nid, xy in nodes.items():
        x, y = xy
        bcs.append({"node": nid, "ux": 0.001 * x + 0.0002 * y, "uy": -0.0003 * y + 0.0004 * x})

    return {
        "analysis": {"dimension": "2D", "type": "static_plane_strain", "unit_system": "m-kN"},
        "mesh": {"nodes": nodes, "elements": elements},
        "materials": {"soil": {"model": "elastic", "E": 50000.0, "nu": 0.30, "gamma": 0.0}},
        "boundary_conditions": bcs,
        "loads": [],
        "output": {"directory": f"runs/patch_{etype.lower()}_{integration.lower().replace('-', '_')}"},
    }

