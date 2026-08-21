"""Mesh generation, set resolution, and mesh validation."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from .fem2d_elements import integration_points, strain_displacement_matrix
from .fem2d_interfaces import interface_stiffness
from .fem2d_types import ElasticPlaneStrainMaterial, Element2D, FEM2DError, Interface2D, Mesh2D, SUPPORTED_ELEMENTS, normalize_integration
from .fem2d_utils import _as_xy, _ensure_list, _range2, _require_sequence, _sets_from_mapping


def mesh_from_config(cfg: Mapping[str, Any]) -> Mesh2D:
    mesh_cfg = cfg.get("mesh", {})
    if not isinstance(mesh_cfg, Mapping):
        raise FEM2DError("mesh must be a mapping")
    if str(mesh_cfg.get("source", "")).strip().lower() in {"external", "file"}:
        mesh_cfg = _load_external_mesh_config(mesh_cfg)

    generator = str(mesh_cfg.get("generator", "")).strip().lower()
    if generator in {"rectangle", "structured_rectangle", "quad_rect", "tri_rect"}:
        mesh = _generate_rectangle_mesh(mesh_cfg)
        _apply_config_sets(mesh, cfg, mesh_cfg)
        _validate_mesh(mesh)
        return mesh

    raw_nodes = mesh_cfg.get("nodes")
    raw_elements = mesh_cfg.get("elements")
    if raw_nodes is None or raw_elements is None:
        raise FEM2DError("mesh.nodes and mesh.elements are required without a generator")

    node_ids: list[str] = []
    coords: list[tuple[float, float]] = []
    if isinstance(raw_nodes, Mapping):
        for node_id, value in raw_nodes.items():
            xy = _as_xy(value, f"mesh.nodes.{node_id}")
            node_ids.append(str(node_id))
            coords.append(xy)
    elif isinstance(raw_nodes, list):
        for i, value in enumerate(raw_nodes, start=1):
            if isinstance(value, Mapping):
                node_id = str(value.get("id", i))
                xy = _as_xy(value.get("coord", value.get("coords", value)), f"mesh.nodes[{i}]")
            else:
                node_id = str(i)
                xy = _as_xy(value, f"mesh.nodes[{i}]")
            node_ids.append(node_id)
            coords.append(xy)
    else:
        raise FEM2DError("mesh.nodes must be a mapping or list")

    default_element_type = str(mesh_cfg.get("element_type", mesh_cfg.get("type", "QUAD4"))).upper()
    default_material = str(mesh_cfg.get("material", cfg.get("default_material", "soil")))
    default_integration = normalize_integration(mesh_cfg.get("integration", "FULL"))
    elements: list[Element2D] = []
    for i, raw in enumerate(raw_elements, start=1):
        if isinstance(raw, Mapping):
            eid = str(raw.get("id", i))
            etype = str(raw.get("type", raw.get("element_type", default_element_type))).upper()
            nodes = raw.get("nodes", raw.get("connectivity"))
            material = str(raw.get("material", default_material))
            integration = normalize_integration(raw.get("integration", default_integration))
            active = bool(raw.get("active", True))
        else:
            eid = str(i)
            etype = default_element_type
            nodes = raw
            material = default_material
            integration = default_integration
            active = True
        if etype not in SUPPORTED_ELEMENTS:
            raise FEM2DError(f"element {eid}: unsupported element type '{etype}'")
        node_tuple = tuple(str(v) for v in _require_sequence(nodes, f"element {eid}.nodes"))
        _check_node_count(eid, etype, node_tuple)
        elements.append(Element2D(eid, etype, node_tuple, material, integration, active))

    mesh = Mesh2D(node_ids=node_ids, coords=np.asarray(coords, dtype=float), elements=elements)
    _apply_config_sets(mesh, cfg, mesh_cfg)
    _validate_mesh(mesh)
    return mesh


def _load_external_mesh_config(mesh_cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw_path = mesh_cfg.get("path", mesh_cfg.get("file", mesh_cfg.get("filename")))
    if not raw_path:
        raise FEM2DError("mesh.source=external requires mesh.path")
    base_dir = Path(str(mesh_cfg.get("base_dir", mesh_cfg.get("root", "."))))
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        raise FEM2DError(f"external mesh file does not exist: {path}")
    fmt = str(mesh_cfg.get("format", path.suffix.lstrip(".") or "")).strip().lower()
    if fmt in {"yaml", "yml", "json"}:
        text = path.read_text(encoding="utf-8")
        loaded = json.loads(text) if fmt == "json" else yaml.safe_load(text)
        if not isinstance(loaded, Mapping):
            raise FEM2DError(f"external mesh file must contain a mapping: {path}")
        external_mesh = loaded.get("mesh", loaded)
        if not isinstance(external_mesh, Mapping):
            raise FEM2DError(f"external mesh mapping is invalid: {path}")
        merged = {key: value for key, value in mesh_cfg.items() if key not in {"source", "path", "file", "filename", "format"}}
        merged.update(dict(external_mesh))
        merged.pop("source", None)
        return merged
    if fmt in {"msh", "gmsh"}:
        parsed = _read_gmsh_v2_mesh(path, mesh_cfg)
        return parsed
    raise FEM2DError(f"unsupported external mesh format '{fmt}' for {path}")


def _read_gmsh_v2_mesh(path: Path, mesh_cfg: Mapping[str, Any]) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    nodes: dict[str, list[float]] = {}
    elements: list[dict[str, Any]] = []
    default_material = str(mesh_cfg.get("material", "soil"))
    default_integration = normalize_integration(mesh_cfg.get("integration", "FULL"))
    gmsh_type_map = {
        2: ("TRI3", 3),
        3: ("QUAD4", 4),
        9: ("TRI6", 6),
        16: ("QUAD8", 8),
    }
    i = 0
    while i < len(lines):
        token = lines[i].strip()
        if token == "$Nodes":
            i += 1
            if i >= len(lines):
                raise FEM2DError(f"invalid Gmsh node block: {path}")
            count = int(lines[i].strip().split()[0])
            for _ in range(count):
                i += 1
                parts = lines[i].strip().split()
                if len(parts) < 4:
                    raise FEM2DError(f"invalid Gmsh node row: {path}")
                nodes[str(parts[0])] = [float(parts[1]), float(parts[2])]
        elif token == "$Elements":
            i += 1
            if i >= len(lines):
                raise FEM2DError(f"invalid Gmsh element block: {path}")
            count = int(lines[i].strip().split()[0])
            for _ in range(count):
                i += 1
                parts = lines[i].strip().split()
                if len(parts) < 4:
                    raise FEM2DError(f"invalid Gmsh element row: {path}")
                eid = str(parts[0])
                gmsh_type = int(parts[1])
                num_tags = int(parts[2])
                type_info = gmsh_type_map.get(gmsh_type)
                if type_info is None:
                    continue
                etype, node_count = type_info
                node_start = 3 + num_tags
                conn = [str(value) for value in parts[node_start : node_start + node_count]]
                if len(conn) != node_count:
                    raise FEM2DError(f"element {eid}: invalid Gmsh connectivity length")
                elements.append({"id": eid, "type": etype, "nodes": conn, "material": default_material, "integration": default_integration})
        i += 1
    if not nodes or not elements:
        raise FEM2DError(f"external Gmsh mesh must contain supported 2D nodes and elements: {path}")
    return {
        "nodes": nodes,
        "elements": elements,
        "element_type": str(mesh_cfg.get("element_type", elements[0]["type"])).upper(),
        "material": default_material,
        "integration": default_integration,
        "node_sets": dict(mesh_cfg.get("node_sets", {}) if isinstance(mesh_cfg.get("node_sets", {}), Mapping) else {}),
        "element_sets": dict(mesh_cfg.get("element_sets", {}) if isinstance(mesh_cfg.get("element_sets", {}), Mapping) else {}),
    }


def interfaces_from_config(cfg: Mapping[str, Any], mesh: Mesh2D) -> list[Interface2D]:
    raw_interfaces = cfg.get("interfaces", cfg.get("interface_elements", []))
    interfaces: list[Interface2D] = []
    for i, raw in enumerate(_ensure_list(raw_interfaces), start=1):
        if not isinstance(raw, Mapping):
            raise FEM2DError("each interface must be a mapping")
        nodes = raw.get("nodes")
        if nodes is not None:
            node_list = [str(v) for v in _require_sequence(nodes, f"interface {i}.nodes")]
            if len(node_list) != 4:
                raise FEM2DError(f"interface {i}.nodes must contain 4 nodes")
            minus_nodes = (node_list[0], node_list[1])
            plus_nodes = (node_list[2], node_list[3])
        else:
            minus_nodes = tuple(str(v) for v in _require_sequence(raw.get("minus_nodes", raw.get("nodes_minus")), f"interface {i}.minus_nodes"))
            plus_nodes = tuple(str(v) for v in _require_sequence(raw.get("plus_nodes", raw.get("nodes_plus")), f"interface {i}.plus_nodes"))
            if len(minus_nodes) != 2 or len(plus_nodes) != 2:
                raise FEM2DError(f"interface {i}: minus_nodes and plus_nodes must contain 2 nodes each")
        behavior = raw.get("behavior", raw.get("law", raw))
        if not isinstance(behavior, Mapping):
            behavior = raw
        hydro_behavior = behavior.get("hydro", behavior.get("hydraulic", {})) if isinstance(behavior, Mapping) else {}
        if not isinstance(hydro_behavior, Mapping):
            hydro_behavior = {}
        friction = float(behavior.get("friction", behavior.get("mu", behavior.get("friction_coefficient", raw.get("friction", raw.get("mu", 0.0))))))
        cohesion = float(behavior.get("cohesion", behavior.get("c", raw.get("cohesion", raw.get("c", 0.0)))))
        material_model = str(behavior.get("material_model", behavior.get("model", raw.get("material_model", raw.get("model", "")))) or "").strip()
        if not material_model:
            material_model = "mohr_coulomb" if friction > 0.0 or cohesion > 0.0 else "linear"
        interface = Interface2D(
            id=str(raw.get("id", i)),
            minus_nodes=(minus_nodes[0], minus_nodes[1]),
            plus_nodes=(plus_nodes[0], plus_nodes[1]),
            kn=float(raw.get("kn", raw.get("normal_stiffness", raw.get("normal", 0.0)))),
            kt=float(raw.get("kt", raw.get("shear_stiffness", raw.get("tangential_stiffness", 0.0)))),
            thickness=float(raw.get("thickness", 1.0)),
            friction=friction,
            cohesion=cohesion,
            no_tension=bool(behavior.get("no_tension", behavior.get("compression_only", raw.get("no_tension", False)))),
            material_model=material_model,
            roughness=float(behavior.get("roughness", behavior.get("jrc", raw.get("roughness", raw.get("jrc", 0.0))))),
            dilatancy_angle=float(behavior.get("dilatancy_angle", behavior.get("dilation_angle", behavior.get("psi", raw.get("dilatancy_angle", raw.get("dilation_angle", raw.get("psi", 0.0))))))),
            roughness_degradation=float(behavior.get("roughness_degradation", behavior.get("jrc_degradation", raw.get("roughness_degradation", raw.get("jrc_degradation", 0.0))))),
            residual_roughness_ratio=float(behavior.get("residual_roughness_ratio", behavior.get("roughness_residual_ratio", raw.get("residual_roughness_ratio", raw.get("roughness_residual_ratio", 0.2))))),
            hydraulic_transfer=float(hydro_behavior.get("transfer", hydro_behavior.get("conductance", behavior.get("hydraulic_transfer", raw.get("hydraulic_transfer", 0.0))))),
            active=bool(raw.get("active", True)),
            history=dict(behavior.get("history", raw.get("history", {}))) if isinstance(behavior.get("history", raw.get("history", {})), Mapping) else {},
        )
        _validate_interface(mesh, interface)
        interfaces.append(interface)
    return interfaces


def _generate_rectangle_mesh(mesh_cfg: Mapping[str, Any]) -> Mesh2D:
    x0, x1 = _range2(mesh_cfg.get("x_range", [0.0, 1.0]), "mesh.x_range")
    y0, y1 = _range2(mesh_cfg.get("y_range", [0.0, 1.0]), "mesh.y_range")
    nx = int(mesh_cfg.get("nx", 1))
    ny = int(mesh_cfg.get("ny", 1))
    if nx <= 0 or ny <= 0:
        raise FEM2DError("mesh.nx and mesh.ny must be positive")
    etype = str(mesh_cfg.get("element_type", mesh_cfg.get("type", "QUAD4"))).upper()
    if etype not in SUPPORTED_ELEMENTS:
        raise FEM2DError(f"unsupported generated 2D element type '{etype}'")
    material = str(mesh_cfg.get("material", "soil"))
    integration = normalize_integration(mesh_cfg.get("integration", "FULL"))

    node_ids: list[str] = []
    coords: list[tuple[float, float]] = []
    corner: dict[tuple[int, int], str] = {}
    for j in range(ny + 1):
        y = y0 + (y1 - y0) * j / ny
        for i in range(nx + 1):
            x = x0 + (x1 - x0) * i / nx
            nid = str(len(node_ids) + 1)
            node_ids.append(nid)
            coords.append((x, y))
            corner[(i, j)] = nid

    edge_mid: dict[tuple[str, str], str] = {}

    def midpoint(a: str, b: str) -> str:
        key = tuple(sorted((a, b)))
        if key in edge_mid:
            return edge_mid[key]
        ia = node_ids.index(a)
        ib = node_ids.index(b)
        xy = (np.asarray(coords[ia]) + np.asarray(coords[ib])) / 2.0
        nid = str(len(node_ids) + 1)
        node_ids.append(nid)
        coords.append((float(xy[0]), float(xy[1])))
        edge_mid[key] = nid
        return nid

    elements: list[Element2D] = []
    for j in range(ny):
        for i in range(nx):
            n00 = corner[(i, j)]
            n10 = corner[(i + 1, j)]
            n11 = corner[(i + 1, j + 1)]
            n01 = corner[(i, j + 1)]
            if etype == "QUAD4":
                elements.append(Element2D(str(len(elements) + 1), "QUAD4", (n00, n10, n11, n01), material, integration))
            elif etype == "QUAD8":
                elements.append(
                    Element2D(
                        str(len(elements) + 1),
                        "QUAD8",
                        (n00, n10, n11, n01, midpoint(n00, n10), midpoint(n10, n11), midpoint(n11, n01), midpoint(n01, n00)),
                        material,
                        integration,
                    )
                )
            elif etype == "TRI3":
                elements.append(Element2D(str(len(elements) + 1), "TRI3", (n00, n10, n11), material, integration))
                elements.append(Element2D(str(len(elements) + 1), "TRI3", (n00, n11, n01), material, integration))
            elif etype == "TRI6":
                tri1 = (n00, n10, n11)
                tri2 = (n00, n11, n01)
                elements.append(Element2D(str(len(elements) + 1), "TRI6", (*tri1, midpoint(tri1[0], tri1[1]), midpoint(tri1[1], tri1[2]), midpoint(tri1[2], tri1[0])), material, integration))
                elements.append(Element2D(str(len(elements) + 1), "TRI6", (*tri2, midpoint(tri2[0], tri2[1]), midpoint(tri2[1], tri2[2]), midpoint(tri2[2], tri2[0])), material, integration))

    if etype == "QUAD8":
        node_sets = {
            "left": _quad8_boundary_sequence([corner[(0, j)] for j in range(ny + 1)], midpoint),
            "right": _quad8_boundary_sequence([corner[(nx, j)] for j in range(ny + 1)], midpoint),
            "bottom": _quad8_boundary_sequence([corner[(i, 0)] for i in range(nx + 1)], midpoint),
            "top": _quad8_boundary_sequence([corner[(i, ny)] for i in range(nx + 1)], midpoint),
            "all": list(node_ids),
        }
    else:
        node_sets = {
            "left": [corner[(0, j)] for j in range(ny + 1)],
            "right": [corner[(nx, j)] for j in range(ny + 1)],
            "bottom": [corner[(i, 0)] for i in range(nx + 1)],
            "top": [corner[(i, ny)] for i in range(nx + 1)],
            "all": list(node_ids),
        }
    element_sets = {"all": [element.id for element in elements]}
    mesh = Mesh2D(node_ids=node_ids, coords=np.asarray(coords, dtype=float), elements=elements, node_sets=node_sets, element_sets=element_sets)
    _validate_mesh(mesh)
    return mesh


def _apply_config_sets(mesh: Mesh2D, cfg: Mapping[str, Any], mesh_cfg: Mapping[str, Any]) -> None:
    sets_cfg = cfg.get("sets", {})
    if isinstance(sets_cfg, Mapping):
        mesh.node_sets.update(_sets_from_mapping(sets_cfg.get("nodes", {})))
        mesh.element_sets.update(_sets_from_mapping(sets_cfg.get("elements", {})))
    if isinstance(mesh_cfg.get("node_sets"), Mapping):
        mesh.node_sets.update(_sets_from_mapping(mesh_cfg.get("node_sets", {})))
    if isinstance(mesh_cfg.get("element_sets"), Mapping):
        mesh.element_sets.update(_sets_from_mapping(mesh_cfg.get("element_sets", {})))


def _target_elements(mesh: Mesh2D, spec: Mapping[str, Any]) -> list[str]:
    if "element" in spec:
        return [str(spec["element"])]
    if "elements" in spec:
        raw = spec["elements"]
        if isinstance(raw, str) and raw in mesh.element_sets:
            return list(mesh.element_sets[raw])
        return [str(v) for v in _require_sequence(raw, "elements")]
    for key in ("element_set", "elementSet", "set"):
        if key in spec:
            set_name = str(spec[key])
            if set_name not in mesh.element_sets:
                raise FEM2DError(f"unknown element set '{set_name}'")
            return list(mesh.element_sets[set_name])
    if bool(spec.get("all", False)):
        return [element.id for element in mesh.elements]
    raise FEM2DError("deactivation stage must define element, elements, element_set, set, or all=true")


def _mesh_with_active_elements(mesh: Mesh2D, active_element_ids: set[str]) -> Mesh2D:
    active = set(active_element_ids)
    return Mesh2D(
        node_ids=list(mesh.node_ids),
        coords=mesh.coords.copy(),
        elements=[replace(element, active=element.id in active) for element in mesh.elements],
        node_sets={key: list(value) for key, value in mesh.node_sets.items()},
        element_sets={key: list(value) for key, value in mesh.element_sets.items()},
    )


def _target_nodes(mesh: Mesh2D, spec: Mapping[str, Any]) -> list[str]:
    if "node" in spec:
        return [str(spec["node"])]
    if "nodes" in spec:
        raw = spec["nodes"]
        if isinstance(raw, str) and raw in mesh.node_sets:
            return list(mesh.node_sets[raw])
        return [str(v) for v in _require_sequence(raw, "nodes")]
    if "set" in spec:
        key = str(spec["set"])
        if key not in mesh.node_sets:
            raise FEM2DError(f"unknown node set '{key}'")
        return list(mesh.node_sets[key])
    raise FEM2DError("boundary/load target must define node, nodes, or set")


def _edge_set(mesh: Mesh2D, name: str) -> list[list[str]]:
    # A simple convenience for generated rectangle boundaries.
    if name not in mesh.node_sets:
        raise FEM2DError(f"unknown edge/node set '{name}'")
    ids = mesh.node_sets[name]
    return [[ids[i], ids[i + 1]] for i in range(len(ids) - 1)]


def _quad8_boundary_sequence(corners: list[str], midpoint: Any) -> list[str]:
    out: list[str] = []
    for left, right in zip(corners, corners[1:]):
        out.append(left)
        out.append(midpoint(left, right))
    if corners:
        out.append(corners[-1])
    return out


def _pressure_edges(mesh: Mesh2D, spec: Mapping[str, Any]) -> list[tuple[str, ...]]:
    edges_raw = spec.get("edges", spec.get("edge"))
    if edges_raw is not None:
        if isinstance(edges_raw, str):
            return [tuple(edge) for edge in _edge_set(mesh, edges_raw)]  # type: ignore[misc]
        edges = _ensure_list(edges_raw)
        if len(edges) in {2, 3} and not isinstance(edges[0], (list, tuple, Mapping)):
            return [tuple(str(node) for node in edges)]
        out: list[tuple[str, ...]] = []
        for edge in edges:
            nodes = [str(v) for v in _require_sequence(edge, "pore boundary edge")]
            if len(nodes) not in {2, 3}:
                raise FEM2DError("pore boundary edges must contain 2 or 3 nodes")
            out.append(tuple(nodes))
        return out
    if "nodes" in spec:
        nodes = [str(v) for v in _require_sequence(spec["nodes"], "pore boundary nodes")]
        if len(nodes) not in {2, 3}:
            raise FEM2DError("pore boundary nodes must contain exactly 2 or 3 nodes")
        return [tuple(nodes)]
    if "set" in spec:
        return [tuple(edge) for edge in _edge_set(mesh, str(spec["set"]))]  # type: ignore[misc]
    raise FEM2DError("pore boundary condition must define edge, edges, nodes, or set")


def _edge_length(mesh: Mesh2D, edge: tuple[str, ...]) -> float:
    node_index = mesh.node_index
    missing = [nid for nid in edge if nid not in node_index]
    if missing:
        raise FEM2DError(f"unknown edge node(s): {missing}")
    length = 0.0
    for left, right in zip(edge, edge[1:]):
        p0 = mesh.coords[node_index[left]]
        p1 = mesh.coords[node_index[right]]
        length += float(np.linalg.norm(p1 - p0))
    if length <= 0.0:
        raise FEM2DError("edge length must be positive")
    return length


def _edge_lumped_weights(edge: Iterable[str]) -> np.ndarray:
    node_count = len(tuple(edge))
    if node_count == 2:
        return np.array([0.5, 0.5], dtype=float)
    if node_count == 3:
        return np.array([1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0], dtype=float)
    raise FEM2DError("pore boundary edges must contain 2 or 3 nodes")


def _edge_consistent_robin_matrix(edge: Iterable[str]) -> np.ndarray:
    node_count = len(tuple(edge))
    if node_count == 2:
        return np.array([[1.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 3.0]], dtype=float)
    if node_count == 3:
        return np.array(
            [
                [2.0 / 15.0, 1.0 / 15.0, -1.0 / 30.0],
                [1.0 / 15.0, 8.0 / 15.0, 1.0 / 15.0],
                [-1.0 / 30.0, 1.0 / 15.0, 2.0 / 15.0],
            ],
            dtype=float,
        )
    raise FEM2DError("pore Robin boundary edges must contain 2 or 3 nodes")


def _validate_mesh(mesh: Mesh2D) -> None:
    if mesh.coords.ndim != 2 or mesh.coords.shape[1] != 2:
        raise FEM2DError("mesh coordinates must have shape (n, 2)")
    if len(mesh.node_ids) != mesh.coords.shape[0]:
        raise FEM2DError("mesh node id count does not match coordinate count")
    node_set = set(mesh.node_ids)
    if len(node_set) != len(mesh.node_ids):
        raise FEM2DError("mesh node ids must be unique")
    element_ids = [element.id for element in mesh.elements]
    if len(set(element_ids)) != len(element_ids):
        raise FEM2DError("mesh element ids must be unique")
    for element in mesh.elements:
        _check_node_count(element.id, element.type, element.nodes)
        missing = [nid for nid in element.nodes if nid not in node_set]
        if missing:
            raise FEM2DError(f"element {element.id}: unknown nodes {missing}")
        coords = np.array([mesh.coords[mesh.node_index[nid]] for nid in element.nodes], dtype=float)
        for gp in integration_points(element.type, "FULL"):
            _B4, _detJ, _N = strain_displacement_matrix(element.type, coords, gp)
    element_set = set(element_ids)
    for name, ids in mesh.node_sets.items():
        missing = [nid for nid in ids if nid not in node_set]
        if missing:
            raise FEM2DError(f"node set {name}: unknown nodes {missing}")
    for name, ids in mesh.element_sets.items():
        missing = [eid for eid in ids if eid not in element_set]
        if missing:
            raise FEM2DError(f"element set {name}: unknown elements {missing}")


def validate_mesh_quality_for_solve(mesh: Mesh2D, cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Reject numerically singular meshes using dimensionless quality measures."""

    raw: Mapping[str, Any] = {}
    if isinstance(cfg, Mapping):
        solver = cfg.get("solver", {})
        checks = cfg.get("checks", {})
        if isinstance(solver, Mapping):
            candidate = solver.get("mesh_quality_preflight", solver.get("mesh_preflight", {}))
            if isinstance(candidate, Mapping):
                raw = candidate
        if isinstance(checks, Mapping):
            quality = checks.get("mesh_quality", {})
            if isinstance(quality, Mapping):
                candidate = quality.get("solve_preflight", quality.get("preflight", {}))
                if isinstance(candidate, Mapping):
                    raw = {**raw, **candidate}
    enabled = bool(raw.get("enabled", True))
    if not enabled:
        return {"enabled": False, "passed": True, "reason": "disabled_by_config"}

    min_normalized_area = max(float(raw.get("min_normalized_area", 1.0e-12)), 0.0)
    min_normalized_jacobian = max(
        float(raw.get("min_normalized_jacobian", raw.get("min_relative_jacobian", 1.0e-12))),
        0.0,
    )
    max_coordinate_ulp_ratio = max(float(raw.get("max_coordinate_ulp_ratio", 1.0e-8)), 0.0)
    max_reported_failures = max(int(raw.get("max_reported_failures", 20)), 1)
    coords = np.asarray(mesh.coords, dtype=float)
    if coords.size == 0 or not np.all(np.isfinite(coords)):
        raise FEM2DError(
            "mesh solve preflight failed: coordinates must be finite and non-empty",
            diagnostics={"status": "mesh_quality_preflight_failed", "reason": "nonfinite_or_empty_coordinates"},
        )

    span = np.ptp(coords, axis=0)
    model_scale = float(np.linalg.norm(span))
    max_coordinate = float(np.max(np.abs(coords)))
    coordinate_ulp = float(abs(np.spacing(max_coordinate))) if max_coordinate > 0.0 else float(np.finfo(float).tiny)
    coordinate_ulp_ratio = coordinate_ulp / model_scale if model_scale > 0.0 else math.inf
    failures: list[dict[str, Any]] = []
    min_area_ratio = math.inf
    min_jacobian_ratio = math.inf
    node_index = mesh.node_index
    checked_elements = 0
    for element in mesh.elements:
        if not element.active:
            continue
        checked_elements += 1
        element_coords = np.asarray([coords[node_index[nid]] for nid in element.nodes], dtype=float)
        corner_count = 3 if element.type.startswith("TRI") else 4
        corner_coords = element_coords[:corner_count]
        edge_lengths = np.linalg.norm(np.roll(corner_coords, -1, axis=0) - corner_coords, axis=1)
        element_scale = float(np.max(edge_lengths)) if edge_lengths.size else 0.0
        scale_squared = element_scale * element_scale
        if not math.isfinite(scale_squared) or scale_squared <= 0.0:
            failures.append({"element": str(element.id), "reason": "zero_or_nonfinite_element_scale"})
            if len(failures) >= max_reported_failures:
                break
            continue
        x = corner_coords[:, 0]
        y = corner_coords[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        area_ratio = area / scale_squared
        min_area_ratio = min(min_area_ratio, area_ratio)
        element_min_jacobian_ratio = math.inf
        try:
            for gp in integration_points(element.type, "FULL"):
                _b4, det_j, _shape = strain_displacement_matrix(element.type, element_coords, gp)
                element_min_jacobian_ratio = min(element_min_jacobian_ratio, float(det_j) / scale_squared)
        except FEM2DError as exc:
            failures.append({"element": str(element.id), "reason": "invalid_jacobian", "message": str(exc)})
            if len(failures) >= max_reported_failures:
                break
            continue
        min_jacobian_ratio = min(min_jacobian_ratio, element_min_jacobian_ratio)
        reasons: list[str] = []
        if not math.isfinite(area_ratio) or area_ratio < min_normalized_area:
            reasons.append("normalized_area")
        if not math.isfinite(element_min_jacobian_ratio) or element_min_jacobian_ratio < min_normalized_jacobian:
            reasons.append("normalized_jacobian")
        if reasons:
            failures.append(
                {
                    "element": str(element.id),
                    "reason": ",".join(reasons),
                    "normalized_area": area_ratio,
                    "normalized_jacobian": element_min_jacobian_ratio,
                    "element_scale": element_scale,
                }
            )
            if len(failures) >= max_reported_failures:
                break

    if checked_elements == 0:
        coordinate_ulp_ratio = 0.0
        min_area_ratio = 0.0
        min_jacobian_ratio = 0.0
    if checked_elements > 0 and model_scale <= 0.0:
        failures.append({"reason": "zero_model_extent", "model_scale": model_scale})
    if checked_elements > 0 and (not math.isfinite(coordinate_ulp_ratio) or coordinate_ulp_ratio > max_coordinate_ulp_ratio):
        failures.append(
            {
                "reason": "coordinate_precision_loss",
                "coordinate_ulp_ratio": coordinate_ulp_ratio,
                "max_coordinate_ulp_ratio": max_coordinate_ulp_ratio,
            }
        )
    summary = {
        "enabled": True,
        "passed": not failures,
        "checked_element_count": checked_elements,
        "geometry_checks_skipped": checked_elements == 0,
        "model_scale": model_scale,
        "coordinate_ulp_ratio": coordinate_ulp_ratio,
        "min_normalized_area": min_area_ratio,
        "min_normalized_jacobian": min_jacobian_ratio,
        "thresholds": {
            "min_normalized_area": min_normalized_area,
            "min_normalized_jacobian": min_normalized_jacobian,
            "max_coordinate_ulp_ratio": max_coordinate_ulp_ratio,
        },
        "failures": failures,
    }
    if failures:
        first = failures[0]
        element_label = f" element {first.get('element')}" if first.get("element") else ""
        raise FEM2DError(
            f"mesh solve preflight failed:{element_label} {first.get('reason', 'numerically unsafe mesh')}",
            diagnostics={"status": "mesh_quality_preflight_failed", **summary},
        )
    return summary


def _validate_material_references(mesh: Mesh2D, materials: Mapping[str, ElasticPlaneStrainMaterial]) -> None:
    for element in mesh.elements:
        if element.material not in materials:
            raise FEM2DError(f"element {element.id}: material '{element.material}' is not defined")


def _validate_interface(mesh: Mesh2D, interface: Interface2D) -> None:
    node_set = set(mesh.node_ids)
    missing = [nid for nid in (*interface.minus_nodes, *interface.plus_nodes) if nid not in node_set]
    if missing:
        raise FEM2DError(f"interface {interface.id}: unknown nodes {missing}")
    interface_stiffness(interface, mesh)


def _collect_warnings(mesh: Mesh2D, materials: Mapping[str, ElasticPlaneStrainMaterial]) -> list[str]:
    warnings: list[str] = []
    for element in mesh.elements:
        material = materials.get(element.material)
        if not material:
            continue
        if material.nu >= 0.45 and element.type == "QUAD4" and normalize_integration(element.integration) == "FULL":
            warnings.append(f"element {element.id}: nu={material.nu:g} with QUAD4/FULL may cause volumetric locking; SRI or B-bar is recommended")
        if material.nu >= 0.49 and normalize_integration(element.integration) == "FULL":
            warnings.append(f"element {element.id}: nu={material.nu:g}; do not use FULL-only results for design judgment")
    return warnings


def _check_node_count(eid: str, etype: str, nodes: Iterable[str]) -> None:
    expected = {"TRI3": 3, "TRI6": 6, "QUAD4": 4, "QUAD8": 8}[etype]
    actual = len(tuple(nodes))
    if actual != expected:
        raise FEM2DError(f"element {eid}: {etype} requires {expected} nodes, got {actual}")

__all__ = [
    "mesh_from_config",
    "interfaces_from_config",
    "_generate_rectangle_mesh",
    "_apply_config_sets",
    "_target_elements",
    "_mesh_with_active_elements",
    "_target_nodes",
    "_edge_set",
    "_pressure_edges",
    "_edge_length",
    "_edge_lumped_weights",
    "_edge_consistent_robin_matrix",
    "_validate_mesh",
    "validate_mesh_quality_for_solve",
    "_validate_material_references",
    "_validate_interface",
    "_collect_warnings",
    "_check_node_count",
]

