"""Configuration parsing for the 2D FEM core."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .fem2d_types import (
    CONSOLIDATION_2D_STAGE_TYPES,
    DEACTIVATION_2D_STAGE_TYPES,
    FEM2DError,
    GEOSTATIC_2D_STAGE_TYPES,
    PLANNED_2D_CORE_BLOCKS,
    PLANNED_2D_CORE_STAGE_TYPES,
    RIKS_2D_STAGE_TYPES,
    SRM_2D_STAGE_TYPES,
    SUPPORTED_2D_CORE_STAGE_TYPES,
    ElasticPlaneStrainMaterial,
)

_BASE_MATERIAL_MODELS = {"elastic", "linear_elastic", "plane_strain_elastic", "drucker_prager", "dp", "von_mises", "j2", "mohr_coulomb", "mc"}
_ADVANCED_MATERIAL_ALIASES = {
    "nonlinear_elastic": "nonlinear_elastic",
    "nonlinear-elastic": "nonlinear_elastic",
    "hardin_drnevich": "hardin_drnevich",
    "hardin-drnevich": "hardin_drnevich",
    "hd": "hardin_drnevich",
    "duncan_chang": "duncan_chang",
    "duncan-chang": "duncan_chang",
    "ramberg_osgood": "ramberg_osgood",
    "ramberg-osgood": "ramberg_osgood",
    "ro": "ramberg_osgood",
    "uw_clay": "uw_clay",
    "uw-clay": "uw_clay",
    "pastor_zienkiewicz_sand": "pastor_zienkiewicz_sand",
    "pastor-zienkiewicz-sand": "pastor_zienkiewicz_sand",
    "pastor_zienkiewicz_clay": "pastor_zienkiewicz_clay",
    "pastor-zienkiewicz-clay": "pastor_zienkiewicz_clay",
    "liquefaction": "liquefaction",
    "bilinear_liquefaction": "bilinear_liquefaction",
    "bilinear-liquefaction": "bilinear_liquefaction",
}
_MATERIAL_KNOWN_KEYS = {
    "model",
    "type",
    "E",
    "young",
    "young_modulus",
    "nu",
    "poisson",
    "gamma",
    "unit_weight",
    "thickness",
    "k0",
    "K0",
    "cohesion",
    "c",
    "friction_angle",
    "phi",
    "dilation_angle",
    "psi",
    "yield_stress",
    "sigma_y",
    "sy",
    "yield",
    "hardening",
    "H",
    "tension_cutoff",
    "ft",
    "tensile_strength",
    "tension_cutoff_stage",
    "mohr_coulomb_apex_policy",
    "apex_policy",
    "apex_return",
}


def validate_2d_core_scope(cfg: Mapping[str, Any]) -> None:
    """Fail fast when a 2D input asks for features not supported by the 2D core."""

    analysis = cfg.get("analysis", {})
    if isinstance(analysis, Mapping):
        backend = str(analysis.get("backend", analysis.get("solver_backend", ""))).lower().strip()
        if backend in {"v26", "v26_plane_strain", "plane_strain_v26", "3d"}:
            raise FEM2DError("3D/v26 backends have been removed; use the native 2D core")

        analysis_type = str(analysis.get("type", "")).lower().strip()
        if analysis_type and analysis_type not in SUPPORTED_2D_CORE_STAGE_TYPES:
            raise FEM2DError(f"2D core feature not implemented yet: analysis.type='{analysis_type}'")

        fields = analysis.get("fields", analysis.get("field", ["u"]))
        fields_list = [fields] if isinstance(fields, str) else list(fields) if isinstance(fields, (list, tuple)) else ["u"]
        fields_norm = {str(v).lower().strip() for v in fields_list}
    for block_name in ("stages", "steps"):
        raw_steps = cfg.get(block_name)
        if raw_steps is None:
            continue
        if not isinstance(raw_steps, list):
            raise FEM2DError(f"{block_name} must be a list")
        for idx, raw in enumerate(raw_steps, start=1):
            if not isinstance(raw, Mapping):
                raise FEM2DError(f"{block_name}[{idx}] must be a mapping")
            stype = str(raw.get("type", "static")).lower().strip()
            if stype in PLANNED_2D_CORE_STAGE_TYPES:
                raise FEM2DError(f"2D core stage type '{stype}' is planned but not implemented yet")
            if stype not in SUPPORTED_2D_CORE_STAGE_TYPES:
                raise FEM2DError(f"unsupported 2D core stage type '{stype}'")
            requested_blocks = sorted(k for k in raw if str(k).lower().strip() in PLANNED_2D_CORE_BLOCKS)
            if requested_blocks:
                raise FEM2DError(f"2D core stage blocks not implemented yet: {', '.join(requested_blocks)}")


def plane_strain_materials(cfg: Mapping[str, Any]) -> dict[str, ElasticPlaneStrainMaterial]:
    raw_materials = cfg.get("materials", cfg.get("material", {}))
    if not isinstance(raw_materials, Mapping) or not raw_materials:
        raise FEM2DError("at least one material is required")
    out: dict[str, ElasticPlaneStrainMaterial] = {}
    for name, raw in raw_materials.items():
        if not isinstance(raw, Mapping):
            raise FEM2DError(f"material {name}: must be a mapping")
        model = str(raw.get("model", raw.get("type", "elastic"))).lower().strip()
        advanced_model = _advanced_material_model(model, raw)
        if model not in _BASE_MATERIAL_MODELS and advanced_model == "":
            raise FEM2DError(f"material {name}: unsupported 2D core material model '{model}'")
        core_model = "elastic" if model in {"linear_elastic", "plane_strain_elastic"} else model
        if advanced_model:
            core_model = advanced_model
        tc_raw = raw.get("tension_cutoff", False)
        if isinstance(tc_raw, Mapping):
            tension_cutoff = bool(tc_raw.get("enabled", True))
            tensile_strength = float(tc_raw.get("ft", tc_raw.get("tensile_strength", raw.get("ft", 0.0))))
            tension_cutoff_stage = str(tc_raw.get("stage", tc_raw.get("mode", raw.get("tension_cutoff_stage", "corrector"))))
        elif isinstance(tc_raw, (int, float)) and not isinstance(tc_raw, bool):
            tension_cutoff = True
            tensile_strength = float(tc_raw)
            tension_cutoff_stage = str(raw.get("tension_cutoff_stage", "corrector"))
        else:
            tension_cutoff = bool(tc_raw)
            tensile_strength = float(raw.get("ft", raw.get("tensile_strength", 0.0 if tension_cutoff else math.inf)))
            tension_cutoff_stage = str(raw.get("tension_cutoff_stage", "corrector"))
        apex_raw = raw.get(
            "mohr_coulomb_apex_policy",
            raw.get("apex_policy", raw.get("apex_return", "legacy_bounded")),
        )
        if isinstance(apex_raw, Mapping):
            apex_policy = str(apex_raw.get("mode", apex_raw.get("policy", "legacy_bounded")))
        else:
            apex_policy = str(apex_raw)
        apex_policy = apex_policy.lower().strip().replace("-", "_")
        apex_aliases = {
            "associated": "associated_multisurface",
            "multisurface": "associated_multisurface",
            "strict": "strict_nonassociated",
            "legacy": "legacy_bounded",
            "rankine": "rankine_cap",
        }
        apex_policy = apex_aliases.get(apex_policy, apex_policy)
        apex_policy_explicit = any(
            key in raw
            for key in ("mohr_coulomb_apex_policy", "apex_policy", "apex_return")
        )
        if apex_policy == "rankine_cap":
            tension_cutoff = True
            if not math.isfinite(tensile_strength):
                tensile_strength = 0.0
        out[str(name)] = ElasticPlaneStrainMaterial(
            name=str(name),
            E=float(raw.get("E", raw.get("young", raw.get("young_modulus", 0.0)))),
            nu=float(raw.get("nu", raw.get("poisson", 0.3))),
            gamma=float(raw.get("gamma", raw.get("unit_weight", 0.0))),
            thickness=float(raw.get("thickness", 1.0)),
            k0=None if raw.get("k0", raw.get("K0")) is None else float(raw.get("k0", raw.get("K0"))),
            model=core_model,
            cohesion=float(raw.get("cohesion", raw.get("c", 0.0))),
            friction_angle=float(raw.get("friction_angle", raw.get("phi", 0.0))),
            dilation_angle=float(raw.get("dilation_angle", raw.get("psi", raw.get("phi", 0.0)))),
            yield_stress=float(raw.get("yield_stress", raw.get("sigma_y", raw.get("sy", 0.0)))),
            hardening=float(raw.get("hardening", raw.get("H", 0.0))),
            tension_cutoff=tension_cutoff,
            tensile_strength=tensile_strength,
            tension_cutoff_stage=tension_cutoff_stage,
            advanced_model=advanced_model,
            advanced_params=_advanced_material_params(raw),
            mohr_coulomb_apex_policy=apex_policy,
            mohr_coulomb_apex_policy_explicit=apex_policy_explicit,
        )
    return out


def _advanced_material_model(model: str, raw: Mapping[str, Any]) -> str:
    if model in _ADVANCED_MATERIAL_ALIASES:
        return _ADVANCED_MATERIAL_ALIASES[model]
    gui_model = str(raw.get("gui_model", raw.get("constitutive_model", ""))).lower().strip().replace(" ", "_")
    if gui_model in _ADVANCED_MATERIAL_ALIASES:
        return _ADVANCED_MATERIAL_ALIASES[gui_model]
    family = str(raw.get("model_family", raw.get("family", ""))).lower().strip().replace(" ", "_")
    if family == "liquefaction":
        return "liquefaction"
    if family == "nonlinear_elastic" and model in {"elastic", "linear_elastic", "plane_strain_elastic"}:
        return "nonlinear_elastic"
    nested_liq = raw.get("liquefaction")
    if isinstance(nested_liq, Mapping):
        return "liquefaction"
    return ""


def _advanced_material_params(raw: Mapping[str, Any]) -> dict[str, Any]:
    params = {str(key): value for key, value in raw.items() if str(key) not in _MATERIAL_KNOWN_KEYS}
    curve = raw.get("test_curve", raw.get("material_test_curve"))
    if curve is not None:
        params["test_curve"] = curve
    return params

__all__ = [
    "validate_2d_core_scope",
    "plane_strain_materials",
    "_advanced_material_model",
    "_advanced_material_params",
]

