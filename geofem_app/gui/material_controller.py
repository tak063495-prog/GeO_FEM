"""Material GUI operation controller functions split from MainWindow.

The MainWindow remains the owner of Qt widgets, project state, and notifications.
This module owns material-library table synchronization, curve fitting helpers,
river liquefaction parameter estimation, and material-panel apply operations.
"""

from __future__ import annotations

import html
import math
from typing import Any, Mapping

import yaml

from geofem_app.material_models import material_form_schema, material_model_catalog, material_validation_issues, normalize_material_model_name


MATERIAL_CONTROLLER_METHODS = (
    "add_material_row",
    "material_library_model_changed",
    "estimate_material_constants_from_curve",
    "apply_river_seismic_parameters",
    "_parse_material_curve_text",
    "_parse_material_curve_sets",
    "_fit_material_curve",
    "_fit_material_curve_global",
    "_fit_confidence_from_jacobian",
    "_material_curve_unit_conversion",
    "_material_fit_report_text",
    "_material_fit_report_html",
    "add_material_from_library",
    "add_material_preset_row",
    "_material_library_spec",
    "_material_extra_yaml",
    "remove_selected_material_rows",
    "apply_materials_panel",
)


def material_controller_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.gui.material_controller.v1",
        "method_count": len(MATERIAL_CONTROLLER_METHODS),
        "methods": list(MATERIAL_CONTROLLER_METHODS),
        "pure_helper_count": 8,
        "owner_boundary": "controller mutates owner material widgets/config; MainWindow delegates material input-domain actions",
    }


def _material_model_options() -> list[tuple[str, str]]:
    japanese_labels = {
        "elastic": "線形弾性(平面ひずみ)",
        "drucker_prager": "ドラッカー・プラガー",
        "mohr_coulomb": "モール・クーロン",
        "von_mises": "von Mises / J2塑性",
        "no_tension": "引張なし弾性",
        "nonlinear_elastic": "非線形弾性",
        "hardin_drnevich": "非線形弾性: Hardin-Drnevich",
        "duncan_chang": "非線形弾性: Duncan-Chang",
        "ramberg_osgood": "非線形弾性: Ramberg-Osgood",
        "liquefaction": "液状化(ru-FL代替)",
        "bilinear_liquefaction": "液状化: バイリニア",
        "uw_clay": "粘性土代替モデル",
        "pastor_zienkiewicz_sand": "砂質土代替モデル",
        "pastor_zienkiewicz_clay": "粘性土代替モデル(PZ)",
    }
    options: list[tuple[str, str]] = []
    for row in material_model_catalog():
        value = str(row.get("model", row.get("name", "")) or row.get("name", ""))
        if not value:
            continue
        label = japanese_labels.get(value, str(row.get("label", value) or value))
        options.append((value, label))
    for value, label in [
        ("hardin_drnevich", japanese_labels["hardin_drnevich"]),
        ("duncan_chang", japanese_labels["duncan_chang"]),
        ("ramberg_osgood", japanese_labels["ramberg_osgood"]),
        ("bilinear_liquefaction", japanese_labels["bilinear_liquefaction"]),
    ]:
        if value not in {item[0] for item in options}:
            options.append((value, label))
    preferred = ["elastic", "mohr_coulomb", "drucker_prager", "von_mises", "no_tension", "hardin_drnevich", "duncan_chang", "ramberg_osgood", "bilinear_liquefaction", "nonlinear_elastic", "liquefaction"]
    by_value = {value: label for value, label in options}
    ordered: list[tuple[str, str]] = []
    for value in preferred:
        if value in by_value:
            ordered.append((value, by_value.pop(value)))
    ordered.extend(sorted(by_value.items(), key=lambda item: item[0]))
    return ordered


def _table_model_value(value: Any) -> str:
    text = str(value or "elastic").strip()
    if text.lower() in {"no-tension", "tension_cutoff", "tension-cutoff"}:
        return "no_tension"
    normalized_text = text.lower().replace("-", "_")
    if normalized_text in {"hardin_drnevich", "duncan_chang", "ramberg_osgood", "bilinear_liquefaction"}:
        return normalized_text
    return normalize_material_model_name(text)


def _required_material_fields(model_text: str) -> set[str]:
    return {str(field.get("name", "")) for field in material_form_schema(model_text).get("fields", []) if bool(field.get("required", False))}


def _material_cell_float(owner: Any, row: int, col: int, field: str, *, required: bool, default: float | None = None) -> float | None:
    self = owner
    text = self._table_text(self.material_table, row, col).strip()
    if not text:
        if required:
            raise ValueError(f"材料 {row + 1} 行目: 必須項目 {field} を入力してください。")
        return default
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"材料 {row + 1} 行目: {field} は数値で入力してください。") from exc


def add_material_row(owner: Any, qt: Mapping[str, Any], *_args: Any, **values: Any) -> None:
    self = owner
    QComboBox = qt.get("QComboBox")
    QTableWidgetItem = qt["QTableWidgetItem"]
    row = self.material_table.rowCount()
    self.material_table.insertRow(row)
    defaults = {
        "name": "soil",
        "model": "elastic",
        "E": 50000.0,
        "nu": 0.3,
        "gamma": 18.0,
        "cohesion": 0.0,
        "phi": 0.0,
        "psi": 0.0,
        "yield_stress": 0.0,
        "hardening": 0.0,
        "k0": "",
        "tension": False,
        "ft": "",
        "extra": "",
    }
    defaults.update(values)
    for col, key in enumerate(["name", "model", "E", "nu", "gamma", "cohesion", "phi", "psi", "yield_stress", "hardening", "k0", "tension", "ft", "extra"]):
        value = str(defaults[key])
        self.material_table.setItem(row, col, QTableWidgetItem(value))
        if col == 1 and QComboBox is not None:
            combo = QComboBox()
            combo.setEditable(True)
            combo.setMinimumWidth(220)
            combo.setMinimumContentsLength(26)
            try:
                combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                combo.view().setMinimumWidth(340)
            except Exception:
                pass
            for option_value, label in _material_model_options():
                combo.addItem(label, option_value)
            current = _table_model_value(value)
            index = combo.findData(current)
            if index < 0:
                combo.addItem(current, current)
                index = combo.findData(current)
            combo.setCurrentIndex(index)
            self.material_table.setCellWidget(row, col, combo)


def material_library_model_changed(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    combo = getattr(self, "material_library_model", None)
    if combo is None:
        return
    key = str(combo.currentData() or combo.currentText())
    spec = self._material_library_spec(key)
    self.material_library_name.setText(str(spec["name"]))
    self.material_library_E.setText(str(spec.get("E", 50000.0)))
    self.material_library_nu.setText(str(spec.get("nu", 0.3)))
    self.material_library_gamma.setText(str(spec.get("gamma", 18.0)))
    self.material_library_cohesion.setText(str(spec.get("cohesion", 0.0)))
    self.material_library_phi.setText(str(spec.get("phi", 0.0)))
    self.material_library_psi.setText(str(spec.get("psi", 0.0)))
    self.material_library_yield.setText(str(spec.get("yield_stress", 0.0)))
    self.material_library_hardening.setText(str(spec.get("hardening", 0.0)))
    self.material_library_k0.setText("" if spec.get("k0", "") in (None, "") else str(spec.get("k0")))
    self.material_library_ft.setText("" if spec.get("ft", "") in (None, "") else str(spec.get("ft")))
    extra = dict(self._mapping(spec.get("extra", {})))
    self.material_library_g0.setText("" if extra.get("G0", "") in (None, "") else str(extra.get("G0")))
    self.material_library_gamma_ref.setText("" if extra.get("gamma_ref", "") in (None, "") else str(extra.get("gamma_ref")))
    liq = self._mapping(extra.get("liquefaction", {}))
    self.material_library_liq_crr.setText("" if liq.get("cyclic_resistance_ratio", liq.get("CRR", "")) in (None, "") else str(liq.get("cyclic_resistance_ratio", liq.get("CRR", ""))))
    self.material_library_extra.setPlainText(yaml.safe_dump(extra, allow_unicode=True, sort_keys=False).strip())
    curve = extra.get("test_curve", "")
    if isinstance(curve, list):
        self.material_test_curve.setPlainText("\n".join(",".join(str(v) for v in row) if isinstance(row, (list, tuple)) else str(row) for row in curve))
    else:
        self.material_test_curve.setPlainText(str(curve) if curve else "")


def estimate_material_constants_from_curve(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    try:
        curve_sets, value_kind = self._parse_material_curve_sets(self.material_test_curve.toPlainText())
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return
    points = [point for values in curve_sets.values() for point in values]
    if len(points) < 2:
        QMessageBox.warning(self, "GeoFEM", "材料試験カーブは2点以上入力してください。")
        return
    fit_result = self._fit_material_curve_global(curve_sets, value_kind)
    fits = dict(fit_result["curve_fits"])
    global_fit = dict(fit_result["global_fit"])
    g0 = float(global_fit["G0"])
    gamma_ref = float(global_fit["gamma_ref"])
    self.material_library_g0.setText(f"{g0:.8g}")
    self.material_library_gamma_ref.setText(f"{gamma_ref:.8g}")
    extra = self._yaml_mapping_text(self.material_library_extra.toPlainText(), "material library extra YAML")
    extra["G0"] = g0
    extra["gamma_ref"] = gamma_ref
    extra["fit_exponent"] = float(global_fit.get("exponent", 1.0))
    extra["test_curve"] = [[float(gamma), float(value)] for gamma, value in points]
    extra["test_curve_sets"] = {name: [[float(gamma), float(value)] for gamma, value in values] for name, values in curve_sets.items()}
    extra["curve_value"] = value_kind
    extra["curve_fits"] = fits
    extra["global_fit"] = global_fit
    extra["fit_confidence"] = fit_result["confidence"]
    extra["fit_constraints"] = fit_result["constraints"]
    extra["unit_conversion"] = self._material_curve_unit_conversion(self.material_test_curve.toPlainText())
    warnings = list(fit_result.get("warnings", []))
    if max(gamma for gamma, _value in points) > 0.2:
        warnings.append("strain values appear larger than engineering shear strain; check percent/unit conversion")
    if min(gamma for gamma, _value in points) < 1.0e-8:
        warnings.append("very small strain values may cause unstable G0 estimation")
    if warnings:
        extra["unit_warnings"] = warnings
    extra["fit_report"] = self._material_fit_report_text(fits, value_kind, warnings, global_fit=global_fit, confidence=fit_result["confidence"])
    extra["fit_report_html"] = self._material_fit_report_html(fits, value_kind, warnings, global_fit, fit_result["confidence"])
    self.material_library_extra.setPlainText(yaml.safe_dump(extra, allow_unicode=True, sort_keys=False).strip())


def apply_river_seismic_parameters(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    try:
        n_value = self._float_text(self.material_river_n.text().strip() or "10.0", "N値")
        fines = self._float_text(self.material_river_fc.text().strip() or "0.0", "Fc")
        sigma_v = self._float_text(self.material_river_sigma_v.text().strip() or "100.0", "sigma_v")
        sigma_v_eff = self._float_text(self.material_river_sigma_v_eff.text().strip() or str(max(sigma_v * 0.6, 1.0)), "sigma_v_eff")
        csr = self._optional_float(self.material_river_csr.text().strip()) or 0.0
        if n_value <= 0.0 or sigma_v_eff <= 0.0:
            raise ValueError("N値と有効上載圧は正値にしてください。")
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return
    g0 = 7000.0 * max(n_value, 1.0) ** 0.75 * math.sqrt(max(sigma_v_eff, 1.0) / 100.0)
    phi = max(24.0, min(42.0, 20.0 + 3.0 * math.sqrt(n_value) - 0.03 * fines))
    crr = max(0.05, min(0.6, 0.088 * math.sqrt(max(n_value, 1.0) / 1.7) + 0.001 * fines))
    self.material_library_g0.setText(f"{g0:.8g}")
    self.material_library_gamma_ref.setText(self.material_library_gamma_ref.text().strip() or "0.001")
    self.material_library_phi.setText(f"{phi:.8g}")
    self.material_library_liq_crr.setText(f"{crr:.8g}")
    if csr > 0.0:
        self.material_river_csr.setText(f"{csr:.8g}")
    extra = self._yaml_mapping_text(self.material_library_extra.toPlainText(), "material library extra YAML")
    extra["G0"] = g0
    extra["gamma_ref"] = float(self.material_library_gamma_ref.text() or 0.001)
    liq = dict(self._mapping(extra.get("liquefaction", {})))
    liq["cyclic_resistance_ratio"] = crr
    if csr > 0.0:
        liq["cyclic_stress_ratio"] = csr
    extra["liquefaction"] = liq
    extra["river_seismic_guideline"] = {
        "N_value": n_value,
        "fines_content": fines,
        "sigma_v": sigma_v,
        "sigma_v_eff": sigma_v_eff,
        "estimated_G0": g0,
        "estimated_phi": phi,
        "estimated_CRR": crr,
    }
    self.material_library_extra.setPlainText(yaml.safe_dump(extra, allow_unicode=True, sort_keys=False).strip())


def _parse_material_curve_text(text: str) -> tuple[list[tuple[float, float]], str]:
    curve_sets, value_kind = _parse_material_curve_sets(text)
    return [point for values in curve_sets.values() for point in values], value_kind


def _parse_material_curve_sets(text: str) -> tuple[dict[str, list[tuple[float, float]]], str]:
    lines = [line.strip() for line in text.replace("\t", ",").splitlines() if line.strip() and not line.strip().startswith("#")]
    if not lines:
        raise ValueError("材料試験カーブを入力してください。")
    header = [part.strip().lower() for part in lines[0].split(",")]
    has_header = any(any(ch.isalpha() for ch in part) for part in header)
    value_kind = "tau" if has_header and any("tau" in part or "stress" in part for part in header[1:]) else "G"
    gamma_header = next((part for part in header if "gamma" in part or "strain" in part), "")
    gamma_scale = 0.01 if has_header and ("%" in gamma_header or "percent" in gamma_header) else 1.0
    data_lines = lines[1:] if has_header else lines
    sets: dict[str, list[tuple[float, float]]] = {}
    for index, line in enumerate(data_lines, start=2 if has_header else 1):
        parts = [part.strip() for part in line.split(",") if part.strip()]
        if len(parts) < 2:
            raise ValueError(f"材料試験カーブ {index} 行目: gamma,value の2列が必要です。")
        curve_name = "curve_1"
        offset = 0
        if len(parts) >= 3:
            try:
                float(parts[0])
            except ValueError:
                curve_name = parts[0]
                offset = 1
        try:
            gamma = float(parts[offset]) * gamma_scale
            value = float(parts[offset + 1])
        except ValueError as exc:
            raise ValueError(f"材料試験カーブ {index} 行目: 数値に変換できません。") from exc
        if gamma <= 0.0 or value <= 0.0:
            raise ValueError(f"材料試験カーブ {index} 行目: gamma/value は正値にしてください。")
        sets.setdefault(curve_name, []).append((gamma, value))
    for values in sets.values():
        values.sort(key=lambda item: item[0])
    return sets, value_kind


def _fit_material_curve(points: list[tuple[float, float]], value_kind: str) -> dict[str, float]:
    result = _fit_material_curve_global({"curve_1": points}, value_kind)
    return dict(result["curve_fits"]["curve_1"])


def _fit_material_curve_global(curve_sets: Mapping[str, list[tuple[float, float]]], value_kind: str) -> dict[str, Any]:
    points = [point for values in curve_sets.values() for point in values]
    gamma_values = [max(float(p[0]), 1.0e-12) for p in points]
    if value_kind == "tau":
        g_values = [max(float(value) / gamma, 1.0e-12) for gamma, value in zip(gamma_values, [p[1] for p in points])]
    else:
        g_values = [max(float(p[1]), 1.0e-12) for p in points]
    min_gamma = min(gamma_values)
    max_gamma = max(gamma_values)
    max_g = max(g_values)
    constraints = {
        "G0": [max(max_g * 0.1, 1.0e-12), max(max_g * 20.0, max_g + 1.0)],
        "gamma_ref": [max(min_gamma * 0.01, 1.0e-12), max(max_gamma * 100.0, min_gamma * 10.0)],
        "exponent": [0.25, 3.0],
    }
    warnings: list[str] = []
    try:
        import numpy as _np
        from scipy.optimize import least_squares

        gamma_arr = _np.asarray(gamma_values, dtype=float)
        g_arr = _np.asarray(g_values, dtype=float)
        target = 0.5 * max_g
        gamma_ref0 = gamma_arr[-1]
        for gamma, g in zip(gamma_arr, g_arr):
            if g <= target:
                gamma_ref0 = gamma
                break
        x0 = _np.log([max_g, max(float(gamma_ref0), constraints["gamma_ref"][0]), 1.0])
        lb = _np.log([constraints["G0"][0], constraints["gamma_ref"][0], constraints["exponent"][0]])
        ub = _np.log([constraints["G0"][1], constraints["gamma_ref"][1], constraints["exponent"][1]])

        def residual(x: Any) -> Any:
            g0, gamma_ref, exponent = _np.exp(x)
            pred = g0 / (1.0 + (gamma_arr / gamma_ref) ** exponent)
            return _np.log(_np.maximum(pred, 1.0e-12)) - _np.log(g_arr)

        opt = least_squares(residual, x0, bounds=(lb, ub), max_nfev=2000)
        g0, gamma_ref, exponent = [float(value) for value in _np.exp(opt.x)]
        pred = g0 / (1.0 + (gamma_arr / gamma_ref) ** exponent)
        rmse = float(_np.sqrt(_np.mean((pred - g_arr) ** 2)))
        confidence = _fit_confidence_from_jacobian(opt.x, opt.fun, opt.jac, ("G0", "gamma_ref", "exponent"))
        confidence["curve_count"] = len(curve_sets)
        optimizer = {"method": "scipy.optimize.least_squares", "success": bool(opt.success), "cost": float(opt.cost), "message": str(opt.message)}
    except Exception as exc:
        warnings.append(f"nonlinear optimizer fallback used: {exc}")
        g0 = max_g
        target = 0.5 * g0
        gamma_ref = gamma_values[-1]
        for gamma, g in zip(gamma_values, g_values):
            if g <= target:
                gamma_ref = gamma
                break
        exponent = 1.0
        rmse = math.sqrt(sum((g - g0 / (1.0 + gamma / max(gamma_ref, 1.0e-12))) ** 2 for gamma, g in zip(gamma_values, g_values)) / len(g_values))
        confidence = {
            "curve_count": len(curve_sets),
            "point_count": len(points),
            "G0_95": [g0, g0],
            "gamma_ref_95": [gamma_ref, gamma_ref],
            "exponent_95": [exponent, exponent],
        }
        optimizer = {"method": "hyperbolic_half_stiffness_fallback", "success": False, "cost": float(rmse)}
    curve_fits: dict[str, dict[str, float]] = {}
    for name, curve_points in curve_sets.items():
        gammas = [max(float(point[0]), 1.0e-12) for point in curve_points]
        gs = [max(float(point[1]) / gamma, 1.0e-12) for gamma, point in zip(gammas, curve_points)] if value_kind == "tau" else [max(float(point[1]), 1.0e-12) for point in curve_points]
        factors = [1.0 / (1.0 + (gamma / max(gamma_ref, 1.0e-12)) ** exponent) for gamma in gammas]
        denom = sum(f * f for f in factors)
        curve_g0 = sum(g * f for g, f in zip(gs, factors)) / max(denom, 1.0e-12)
        residuals = [g - curve_g0 * f for g, f in zip(gs, factors)]
        curve_fits[str(name)] = {
            "G0": float(curve_g0),
            "gamma_ref": float(gamma_ref),
            "exponent": float(exponent),
            "rmse": math.sqrt(sum(value * value for value in residuals) / max(len(residuals), 1)),
            "point_count": float(len(curve_points)),
        }
    return {
        "global_fit": {"G0": g0, "gamma_ref": gamma_ref, "exponent": exponent, "rmse": rmse, "point_count": float(len(points)), "curve_count": float(len(curve_sets)), "optimizer": optimizer},
        "curve_fits": curve_fits,
        "confidence": confidence,
        "constraints": constraints,
        "warnings": warnings,
    }


def _fit_confidence_from_jacobian(x: Any, residual: Any, jacobian: Any, names: tuple[str, ...]) -> dict[str, Any]:
    try:
        import numpy as _np

        jtj = _np.asarray(jacobian, dtype=float).T @ _np.asarray(jacobian, dtype=float)
        dof = max(int(_np.asarray(residual).size) - len(names), 1)
        variance = float(_np.sum(_np.asarray(residual, dtype=float) ** 2) / dof)
        cov = _np.linalg.pinv(jtj) * variance
        se = _np.sqrt(_np.maximum(_np.diag(cov), 0.0))
        out: dict[str, Any] = {"point_count": int(_np.asarray(residual).size), "dof": dof}
        for i, name in enumerate(names):
            lo = float(math.exp(float(x[i]) - 1.96 * float(se[i])))
            hi = float(math.exp(float(x[i]) + 1.96 * float(se[i])))
            out[f"{name}_95"] = [lo, hi]
            out[f"{name}_std_log"] = float(se[i])
        return out
    except Exception:
        values = [float(math.exp(float(v))) for v in x]
        return {**{"point_count": 0, "dof": 0}, **{f"{name}_95": [values[i], values[i]] for i, name in enumerate(names)}}


def _material_curve_unit_conversion(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.replace("\t", ",").splitlines() if line.strip() and not line.strip().startswith("#")]
    if not lines:
        return {"gamma_scale": 1.0, "note": "none"}
    header = [part.strip().lower() for part in lines[0].split(",")]
    has_header = any(any(ch.isalpha() for ch in part) for part in header)
    gamma_header = next((part for part in header if "gamma" in part or "strain" in part), "")
    scale = 0.01 if has_header and ("%" in gamma_header or "percent" in gamma_header) else 1.0
    return {"gamma_scale": scale, "gamma_unit": "percent" if scale != 1.0 else "strain", "converted_to": "engineering_strain"}


def _material_fit_report_text(
    fits: Mapping[str, Mapping[str, float]],
    value_kind: str,
    warnings: list[str],
    *,
    global_fit: Mapping[str, Any] | None = None,
    confidence: Mapping[str, Any] | None = None,
) -> str:
    lines = [f"value={value_kind}", f"curve_count={len(fits)}"]
    if global_fit:
        lines.append(
            "global: "
            f"G0={float(global_fit.get('G0', 0.0)):.8g}, "
            f"gamma_ref={float(global_fit.get('gamma_ref', 0.0)):.8g}, "
            f"exponent={float(global_fit.get('exponent', 1.0)):.8g}, "
            f"rmse={float(global_fit.get('rmse', 0.0)):.8g}"
        )
    if confidence:
        for key in ("G0_95", "gamma_ref_95", "exponent_95"):
            if key in confidence:
                lines.append(f"{key}={confidence[key]}")
    for name, fit in fits.items():
        lines.append(f"{name}: G0={fit.get('G0', 0.0):.8g}, gamma_ref={fit.get('gamma_ref', 0.0):.8g}, exponent={fit.get('exponent', 1.0):.8g}, rmse={fit.get('rmse', 0.0):.8g}")
    for warning in warnings:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)


def _material_fit_report_html(
    fits: Mapping[str, Mapping[str, float]],
    value_kind: str,
    warnings: list[str],
    global_fit: Mapping[str, Any],
    confidence: Mapping[str, Any],
) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(str(name))}</td><td>{float(fit.get('G0', 0.0)):.8g}</td><td>{float(fit.get('gamma_ref', 0.0)):.8g}</td><td>{float(fit.get('exponent', 1.0)):.8g}</td><td>{float(fit.get('rmse', 0.0)):.8g}</td></tr>"
        for name, fit in fits.items()
    )
    warning_html = "".join(f"<li>{html.escape(str(warning))}</li>" for warning in warnings)
    return (
        "<section><h2>Material parameter fit</h2>"
        f"<p>value={html.escape(value_kind)}, G0={float(global_fit.get('G0', 0.0)):.8g}, gamma_ref={float(global_fit.get('gamma_ref', 0.0)):.8g}, exponent={float(global_fit.get('exponent', 1.0)):.8g}</p>"
        f"<p>G0_95={html.escape(str(confidence.get('G0_95', '')))}, gamma_ref_95={html.escape(str(confidence.get('gamma_ref_95', '')))}, exponent_95={html.escape(str(confidence.get('exponent_95', '')))}</p>"
        "<table><thead><tr><th>curve</th><th>G0</th><th>gamma_ref</th><th>exponent</th><th>rmse</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        f"<ul>{warning_html}</ul></section>"
    )


def add_material_from_library(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    key = str(self.material_library_model.currentData() or self.material_library_model.currentText())
    spec = self._material_library_spec(key)
    try:
        extra = self._yaml_mapping_text(self.material_library_extra.toPlainText(), "material library extra YAML")
        g0 = self._optional_float(self.material_library_g0.text())
        gamma_ref = self._optional_float(self.material_library_gamma_ref.text())
        liq_crr = self._optional_float(self.material_library_liq_crr.text())
        if g0 is not None:
            extra["G0"] = g0
        if gamma_ref is not None:
            extra["gamma_ref"] = gamma_ref
        if liq_crr is not None:
            liq = dict(self._mapping(extra.get("liquefaction", {})))
            liq["cyclic_resistance_ratio"] = liq_crr
            extra["liquefaction"] = liq
        curve_text = self.material_test_curve.toPlainText().strip()
        if curve_text and "test_curve" not in extra:
            points, value_kind = self._parse_material_curve_text(curve_text)
            extra["test_curve"] = [[float(gamma), float(value)] for gamma, value in points]
            extra["curve_value"] = value_kind
        self.add_material_row(
            name=self.material_library_name.text().strip() or str(spec["name"]),
            model=str(spec["model"]),
            E=self._float_text(self.material_library_E.text(), "E"),
            nu=self._float_text(self.material_library_nu.text(), "nu"),
            gamma=self._float_text(self.material_library_gamma.text(), "gamma"),
            cohesion=self._float_text(self.material_library_cohesion.text(), "cohesion"),
            phi=self._float_text(self.material_library_phi.text(), "phi"),
            psi=self._float_text(self.material_library_psi.text(), "psi"),
            yield_stress=self._float_text(self.material_library_yield.text(), "yield"),
            hardening=self._float_text(self.material_library_hardening.text(), "hardening"),
            k0=self.material_library_k0.text().strip(),
            tension=bool(spec.get("tension", False)) or bool(self.material_library_ft.text().strip()),
            ft=self.material_library_ft.text().strip(),
            extra=yaml.safe_dump(extra, allow_unicode=True, sort_keys=False).strip(),
        )
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return


def add_material_preset_row(owner: Any, qt: Mapping[str, Any], model: str) -> None:
    self = owner
    presets = {
        "elastic": {"name": "soil_elastic", "model": "elastic", "cohesion": 0.0, "phi": 0.0, "psi": 0.0},
        "mohr_coulomb": {"name": "soil_mc", "model": "mohr_coulomb", "cohesion": 10.0, "phi": 30.0, "psi": 0.0},
        "drucker_prager": {"name": "soil_dp", "model": "drucker_prager", "cohesion": 10.0, "phi": 30.0, "psi": 0.0},
        "von_mises": {"name": "soil_j2", "model": "von_mises", "yield_stress": 30.0, "hardening": 0.0},
        "no_tension": {"name": "soil_no_tension", "model": "elastic", "tension": True, "ft": 0.0},
    }
    if model in presets:
        self.add_material_row(**presets[model])
        return
    spec = self._material_library_spec(model)
    self.add_material_row(
        name=spec["name"],
        model=spec["model"],
        E=spec.get("E", 50000.0),
        nu=spec.get("nu", 0.3),
        gamma=spec.get("gamma", 18.0),
        cohesion=spec.get("cohesion", 0.0),
        phi=spec.get("phi", 0.0),
        psi=spec.get("psi", 0.0),
        yield_stress=spec.get("yield_stress", 0.0),
        hardening=spec.get("hardening", 0.0),
        k0="" if spec.get("k0", "") in (None, "") else spec.get("k0"),
        tension=spec.get("tension", False),
        ft="" if spec.get("ft", "") in (None, "") else spec.get("ft"),
        extra=yaml.safe_dump(spec.get("extra", {}), allow_unicode=True, sort_keys=False).strip(),
    )


def _material_library_spec(owner: Any, qt: Mapping[str, Any], key: str) -> dict[str, Any]:
    self = owner
    common = {"E": 50000.0, "nu": 0.3, "gamma": 18.0, "cohesion": 0.0, "phi": 0.0, "psi": 0.0, "yield_stress": 0.0, "hardening": 0.0}
    specs: dict[str, dict[str, Any]] = {
        "elastic_linear": {"name": "soil_elastic", "model": "elastic", **common, "extra": {"gui_model": "linear_elastic", "model_family": "elastic", "catalog": "GeoFEAS elastic"}},
        "elastic_orthotropic": {"name": "soil_orthotropic", "model": "elastic", **common, "extra": {"gui_model": "orthotropic_elastic", "model_family": "elastic", "Ex": 50000.0, "Ey": 30000.0, "Gxy": 15000.0, "nuxy": 0.25, "solver_status": "catalog_only_elastic_fallback"}},
        "elastic_undrained": {"name": "clay_undrained_elastic", "model": "elastic", **common, "nu": 0.49, "extra": {"gui_model": "undrained_elastic", "model_family": "elastic", "undrained": True, "solver_status": "elastic_core"}},
        "elastic_k0": {"name": "soil_k0_elastic", "model": "elastic", **common, "k0": 0.5, "extra": {"gui_model": "k0_elastic", "model_family": "elastic", "initial_stress": "K0"}},
        "nonlinear_elastic_hardin_drnevich": {"name": "soil_hd", "model": "hardin_drnevich", **common, "extra": {"gui_model": "hardin_drnevich", "model_family": "nonlinear_elastic", "G0": 25000.0, "gamma_ref": 0.001, "solver_status": "2d_core_nonlinear_secant"}},
        "nonlinear_elastic_duncan_chang": {"name": "soil_duncan_chang", "model": "duncan_chang", **common, "extra": {"gui_model": "duncan_chang", "model_family": "nonlinear_elastic", "K": 500.0, "n": 0.5, "Rf": 0.9, "solver_status": "2d_core_nonlinear_secant"}},
        "nonlinear_elastic_ramberg_osgood": {"name": "soil_ramberg_osgood", "model": "ramberg_osgood", **common, "extra": {"gui_model": "ramberg_osgood", "model_family": "nonlinear_elastic", "G0": 25000.0, "alpha": 1.0, "r": 2.0, "solver_status": "2d_core_nonlinear_secant"}},
        "uw_clay": {"name": "clay_uw", "model": "uw_clay", **common, "cohesion": 20.0, "phi": 0.0, "extra": {"gui_model": "uw_clay", "model_family": "critical_state", "G0": 20000.0, "gamma_ref": 0.002, "su": 20.0}},
        "pastor_zienkiewicz_sand": {"name": "sand_pz", "model": "pastor_zienkiewicz_sand", **common, "phi": 34.0, "psi": 5.0, "extra": {"gui_model": "pastor_zienkiewicz_sand", "model_family": "sand_plasticity", "G0": 30000.0, "gamma_ref": 0.001, "phi_cs": 34.0}},
        "pastor_zienkiewicz_clay": {"name": "clay_pz", "model": "pastor_zienkiewicz_clay", **common, "cohesion": 15.0, "phi": 24.0, "extra": {"gui_model": "pastor_zienkiewicz_clay", "model_family": "clay_plasticity", "G0": 18000.0, "gamma_ref": 0.002, "su": 15.0, "phi_cs": 24.0}},
        "perfect_plastic_mohr_coulomb": {"name": "soil_mc", "model": "mohr_coulomb", **common, "cohesion": 10.0, "phi": 30.0, "extra": {"gui_model": "mohr_coulomb_perfect_plastic", "model_family": "perfect_plastic"}},
        "perfect_plastic_drucker_prager": {"name": "soil_dp", "model": "drucker_prager", **common, "cohesion": 10.0, "phi": 30.0, "extra": {"gui_model": "drucker_prager_perfect_plastic", "model_family": "perfect_plastic"}},
        "perfect_plastic_von_mises": {"name": "soil_j2", "model": "von_mises", **common, "yield_stress": 30.0, "extra": {"gui_model": "von_mises_perfect_plastic", "model_family": "perfect_plastic"}},
        "elastoplastic_mohr_coulomb_hardening": {"name": "soil_mc_hardening", "model": "mohr_coulomb", **common, "cohesion": 10.0, "phi": 30.0, "hardening": 100.0, "extra": {"gui_model": "mohr_coulomb_hardening", "model_family": "elastoplastic"}},
        "elastoplastic_drucker_prager_hardening": {"name": "soil_dp_hardening", "model": "drucker_prager", **common, "cohesion": 10.0, "phi": 30.0, "hardening": 100.0, "extra": {"gui_model": "drucker_prager_hardening", "model_family": "elastoplastic"}},
        "no_tension": {"name": "soil_no_tension", "model": "elastic", **common, "tension": True, "ft": 0.0, "extra": {"gui_model": "no_tension", "model_family": "tension_cutoff", "tension_cutoff_stage": "corrector"}},
        "bilinear_liquefaction": {"name": "sand_bilinear_liquefaction", "model": "bilinear_liquefaction", **common, "phi": 32.0, "extra": {"gui_model": "bilinear_liquefaction", "model_family": "liquefaction", "G0": 25000.0, "gamma_ref": 0.001, "liquefaction": {"model": "bilinear", "cyclic_resistance_ratio": 0.18, "post_liquefaction_stiffness_ratio": 0.02, "trigger": "excess_pore_pressure_ratio"}}},
    }
    return dict(specs.get(key, specs["elastic_linear"]))


def _material_extra_yaml(owner: Any, qt: Mapping[str, Any], material: Mapping[str, Any]) -> str:
    self = owner
    known = {
        "model",
        "type",
        "E",
        "young",
        "young_modulus",
        "nu",
        "poisson",
        "gamma",
        "unit_weight",
        "cohesion",
        "c",
        "friction_angle",
        "phi",
        "dilation_angle",
        "psi",
        "yield_stress",
        "sigma_y",
        "sy",
        "hardening",
        "H",
        "k0",
        "K0",
        "tension_cutoff",
        "ft",
        "tensile_strength",
    }
    extra = {str(key): value for key, value in material.items() if str(key) not in known}
    return yaml.safe_dump(extra, allow_unicode=True, sort_keys=False).strip() if extra else ""


def remove_selected_material_rows(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    rows = sorted({index.row() for index in self.material_table.selectedIndexes()}, reverse=True)
    for row in rows:
        self.material_table.removeRow(row)


def apply_materials_panel(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    materials: dict[str, dict[str, Any]] = {}
    try:
        for row in range(self.material_table.rowCount()):
            name = self._table_text(self.material_table, row, 0).strip()
            if not name:
                continue
            model_text = _table_model_value(self._table_text(self.material_table, row, 1).strip() or "elastic")
            no_tension_model = model_text.lower() in {"no_tension", "no-tension", "tension_cutoff"}
            required = _required_material_fields(model_text)
            extra = self._yaml_mapping_text(self._table_text(self.material_table, row, 13), "material extra YAML")
            e_value = _material_cell_float(self, row, 2, "E", required="E" in required)
            nu_value = _material_cell_float(self, row, 3, "nu", required="nu" in required, default=0.3)
            material = {
                "model": "elastic" if no_tension_model else model_text,
            }
            if e_value is not None:
                material["E"] = e_value
            if nu_value is not None:
                material["nu"] = nu_value
            optional_columns = [
                (4, "gamma", "gamma", 0.0),
                (5, "cohesion", "cohesion", 0.0),
                (6, "friction_angle", "friction_angle", 0.0),
                (7, "dilation_angle", "dilation_angle", 0.0),
                (8, "yield_stress", "yield_stress", 0.0),
                (9, "hardening", "hardening", 0.0),
            ]
            for col, key, field_name, default in optional_columns:
                value = _material_cell_float(self, row, col, field_name, required=field_name in required, default=default)
                if value is not None:
                    material[key] = value
            k0_text = self._table_text(self.material_table, row, 10).strip()
            if k0_text:
                material["k0"] = self._float_text(k0_text, "k0")
            tension_text = self._table_text(self.material_table, row, 11).strip().lower()
            ft_text = self._table_text(self.material_table, row, 12).strip()
            tension_enabled = no_tension_model or tension_text in {"true", "1", "yes", "on", "no_tension", "enabled"} or bool(ft_text)
            if tension_enabled:
                material["tension_cutoff"] = True
                material["tensile_strength"] = self._float_text(ft_text or "0.0", "ft")
            material.update(extra)
            if no_tension_model:
                material["model"] = "elastic"
                material.setdefault("tension_cutoff", True)
            validation = material_validation_issues(name, material)
            if validation:
                first = validation[0]
                raise ValueError(f"材料 {name}: {first.get('message', '入力が不足しています')} ({first.get('path', '')})")
            materials[name] = material
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return
    if not materials:
        QMessageBox.warning(self, "GeoFEM", "材料を1件以上入力してください。")
        return
    self.cfg["materials"] = materials
    pending_regions = [str(value) for value in getattr(self, "_pending_material_assignment_regions", []) if str(value).strip()]
    pending_name = str(getattr(self, "_pending_material_assignment_name", "") or "").strip()
    if pending_regions and pending_name in materials and hasattr(self, "_assign_material_to_regions") and hasattr(self, "_finish_material_assignment_change"):
        assigned = self._assign_material_to_regions(pending_name, pending_regions)
        self._pending_material_assignment_regions = []
        self._pending_material_assignment_name = ""
        if assigned:
            self._finish_material_assignment_change(assigned, f"材料を反映し、選択対象へ割り当てました: {pending_name}")
            return
    self._after_form_change("材料を反映しました")


__all__ = [
    "MATERIAL_CONTROLLER_METHODS",
    "material_controller_contract",
    *MATERIAL_CONTROLLER_METHODS,
]
