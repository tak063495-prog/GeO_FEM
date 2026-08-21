"""User-facing labels for values that retain stable solver/YAML identifiers."""

from __future__ import annotations

import re
from typing import Any, Iterable


ChoiceLabels = dict[str, dict[str, tuple[str, str]]]


CHOICE_LABELS: ChoiceLabels = {
    "analysis_type": {
        "static_plane_strain": ("静的解析（平面ひずみ）", "Static (Plane Strain)"),
        "axisymmetric_static": ("静的解析（軸対称）", "Static (Axisymmetric)"),
        "geostatic": ("初期応力・自重解析", "Geostatic / Self Weight"),
        "srm": ("安全率解析（SRM）", "Safety Factor (SRM)"),
        "consolidation": ("圧密解析", "Consolidation"),
        "riks": ("Riks・弧長法解析", "Riks / Arc Length"),
    },
    "analysis_geometry": {
        "plane_strain": ("平面ひずみ", "Plane Strain"),
        "axisymmetric": ("軸対称", "Axisymmetric"),
    },
    "deformation_mode": {
        "small_deformation": ("微小変形", "Small Deformation"),
        "large_deformation": ("大変形", "Large Deformation"),
    },
    "mesh_generator": {
        "rectangle": ("矩形メッシュ", "Rectangular Mesh"),
    },
    "geometry_boolean": {
        "union": ("結合", "Union"),
        "intersection": ("共通部分", "Intersection"),
        "expression": ("条件式で指定", "Expression"),
    },
    "mpc_dof": {
        "ux": ("X方向変位", "X Displacement"),
        "uy": ("Y方向変位", "Y Displacement"),
    },
    "mpc_method": {
        "lagrange": ("ラグランジュ乗数法", "Lagrange Multiplier"),
        "elimination": ("消去法", "Elimination"),
    },
    "stage_type": {
        "static": ("静的", "Static"),
        "large_deformation": ("大変形", "Large Deformation"),
        "geostatic": ("初期応力・自重", "Geostatic / Self Weight"),
        "k0": ("K0初期応力", "K0 Initial Stress"),
        "excavation": ("掘削", "Excavation"),
        "death": ("要素無効化", "Deactivate Elements"),
        "deactivate": ("要素無効化", "Deactivate Elements"),
        "srm": ("安全率（SRM）", "Safety Factor (SRM)"),
        "safety_factor": ("安全率", "Safety Factor"),
        "consolidation": ("圧密", "Consolidation"),
        "u-p": ("変位・水圧連成", "Displacement-Pore Pressure"),
        "riks": ("Riks法", "Riks"),
        "arc_length": ("弧長法", "Arc Length"),
    },
    "boolean": {
        "true": ("使用する", "Use"),
        "false": ("使用しない", "Do Not Use"),
    },
    "result_component": {
        "ux": ("X方向変位", "X Displacement"),
        "uy": ("Y方向変位", "Y Displacement"),
        "u_norm": ("変位量", "Displacement Magnitude"),
        "settlement": ("沈下量", "Settlement"),
        "pore_pressure": ("間隙水圧", "Pore Pressure"),
        "q": ("偏差応力 q", "Deviatoric Stress q"),
        "p": ("平均応力 p", "Mean Stress p"),
        "sigma_x": ("X方向応力 σx", "X Stress σx"),
        "sigma_y": ("Y方向応力 σy", "Y Stress σy"),
        "sigma_z": ("Z方向応力 σz", "Z Stress σz"),
        "tau_xy": ("せん断応力 τxy", "Shear Stress τxy"),
        "sigma_1": ("最大主応力 σ1", "Major Principal Stress σ1"),
        "sigma_2": ("中間主応力 σ2", "Intermediate Principal Stress σ2"),
        "sigma_3": ("最小主応力 σ3", "Minor Principal Stress σ3"),
        "tau_max": ("最大せん断応力", "Maximum Shear Stress"),
        "eps_x": ("X方向ひずみ εx", "X Strain εx"),
        "eps_y": ("Y方向ひずみ εy", "Y Strain εy"),
        "gamma_xy": ("せん断ひずみ γxy", "Shear Strain γxy"),
        "plastic": ("塑性状態", "Plastic State"),
        "yield_value": ("降伏関数値", "Yield Function"),
        "FL": ("局所安全率 FL", "Local Safety Factor FL"),
        "safety_factor": ("安全率", "Safety Factor"),
        "residual_norm": ("残差ノルム", "Residual Norm"),
        "elapsed_seconds": ("経過時間", "Elapsed Time"),
    },
    "post_mode": {
        "none": ("未表示", "Not Displayed"),
        "contour": ("要素コンター", "Element Contour"),
        "node_contour": ("節点コンター", "Node Contour"),
        "vector": ("ベクトル表示", "Vector View"),
        "deformed": ("変形図", "Deformed Shape"),
        "plastic": ("塑性状態", "Plastic State"),
        "srm": ("SRM結果", "SRM Results"),
        "distribution": ("分布図", "Distribution Plot"),
        "table": ("数値表", "Value Table"),
    },
    "srm_aggregation": {
        "mean": ("平均", "Mean"),
        "min": ("最小", "Minimum"),
        "max": ("最大", "Maximum"),
    },
    "srm_search_mode": {
        "all": ("すべて", "All"),
        "circular": ("円弧", "Circular"),
        "non-circular": ("非円弧", "Non-Circular"),
        "optimized path": ("最適経路", "Optimized Path"),
    },
    "srm_direction": {
        "auto": ("自動", "Automatic"),
        "left-to-right": ("左から右", "Left to Right"),
        "right-to-left": ("右から左", "Right to Left"),
    },
    "srm_trial_search_mode": {
        "auto": ("自動探索", "Automatic Search"),
        "adaptive_bracket": ("適応ブラケット探索", "Adaptive Bracket Search"),
        "explicit_factors": ("指定係数の試行", "Specified Factor Trials"),
        "two_branch": ("上下分岐探索", "Two-Branch Search"),
        "two_branch_bisection": ("上下分岐・二分探索", "Two-Branch Bisection"),
        "bisection": ("二分探索", "Bisection Search"),
        "coarse_to_fine": ("粗い探索から細かい探索", "Coarse-to-Fine Search"),
    },
}


def choice_label(group: str, value: Any, *, locale: str = "ja") -> str:
    raw = str(value)
    labels = CHOICE_LABELS.get(str(group), {}).get(raw)
    if labels is None:
        return raw
    return labels[1] if str(locale).startswith("en") else labels[0]


def populate_labeled_combo(combo: Any, group: str, values: Iterable[str], *, locale: str = "ja") -> None:
    for value in values:
        raw = str(value)
        combo.addItem(choice_label(group, raw, locale=locale), raw)
    combo.setProperty("presentationChoiceGroup", str(group))


def combo_value(combo: Any, default: str = "") -> str:
    data = combo.currentData() if combo is not None and hasattr(combo, "currentData") else None
    if data not in (None, ""):
        return str(data)
    if combo is not None and hasattr(combo, "currentText"):
        return str(combo.currentText())
    return str(default)


def relabel_combo(combo: Any, *, locale: str = "ja") -> None:
    group = str(combo.property("presentationChoiceGroup") or "")
    if not group:
        return
    for index in range(combo.count()):
        raw = combo.itemData(index)
        if raw in (None, ""):
            raw = combo.itemText(index)
            combo.setItemData(index, str(raw))
        combo.setItemText(index, choice_label(group, raw, locale=locale))


_ROOT_LABELS = {
    "analysis": ("解析条件", "Analysis settings"),
    "geometry": ("形状", "Geometry"),
    "mesh": ("メッシュ", "Mesh"),
    "materials": ("材料", "Materials"),
    "boundary_conditions": ("境界条件", "Boundary conditions"),
    "bc": ("境界条件", "Boundary conditions"),
    "loads": ("荷重", "Loads"),
    "stages": ("ステージ", "Stages"),
    "solver": ("解析設定", "Solver settings"),
    "results": ("解析結果", "Results"),
}

_FIELD_LABELS = {
    "type": ("種別", "type"),
    "geometry": ("解析形状", "geometry"),
    "deformation_mode": ("変形モード", "deformation mode"),
    "dimension": ("解析次元", "dimension"),
    "unit_system": ("単位系", "unit system"),
    "nodes": ("節点", "nodes"),
    "elements": ("要素", "elements"),
    "regions": ("領域", "regions"),
    "material": ("材料割当", "material assignment"),
    "E": ("ヤング率", "Young's modulus"),
    "young": ("ヤング率", "Young's modulus"),
    "nu": ("ポアソン比", "Poisson's ratio"),
    "poisson": ("ポアソン比", "Poisson's ratio"),
    "gamma": ("単位体積重量", "unit weight"),
    "unit_weight": ("単位体積重量", "unit weight"),
    "cohesion": ("粘着力", "cohesion"),
    "friction_angle": ("内部摩擦角", "friction angle"),
    "dilation_angle": ("ダイレイタンシー角", "dilation angle"),
    "nx": ("X方向分割数", "X divisions"),
    "ny": ("Y方向分割数", "Y divisions"),
    "x_range": ("X範囲", "X range"),
    "y_range": ("Y範囲", "Y range"),
    "closure": ("形状の閉合", "geometry closure"),
    "fixed": ("拘束", "constraint"),
    "set": ("対象セット", "target set"),
}

_RESULT_ARTIFACT_LABELS = {
    "results/summary.json": ("解析結果サマリ", "Result summary"),
    "summary.json": ("解析結果サマリ", "Result summary"),
    "results/failure_report.json": ("解析失敗レポート", "Analysis failure report"),
    "failure_report.json": ("解析失敗レポート", "Analysis failure report"),
    "results/standard_report.html": ("標準帳票", "Standard report"),
    "results/calculation_report.html": ("計算書", "Calculation report"),
}


def friendly_stage_name(value: Any, *, locale: str = "ja") -> str:
    """Humanize generated stage identifiers while preserving user-entered names."""

    raw = str(value or "").strip()
    if not raw or not re.search(r"[_-]", raw):
        return raw
    tokens = [token for token in re.split(r"[_-]+", raw) if token]
    japanese = not str(locale).startswith("en")
    ja_tokens = {
        "srm": "SRM",
        "strength": "強度",
        "reduction": "低減",
        "geostatic": "初期応力",
        "static": "静的",
        "large": "大変形",
        "deformation": "",
        "consolidation": "圧密",
    }
    rendered: list[str] = []
    for token in tokens:
        lowered = token.lower()
        case_match = re.fullmatch(r"case(\d+)", lowered)
        if case_match:
            rendered.append(f"Case {case_match.group(1)}")
        elif japanese and lowered in ja_tokens:
            if ja_tokens[lowered]:
                rendered.append(ja_tokens[lowered])
        elif lowered == "srm":
            rendered.append("SRM")
        else:
            rendered.append(token[:1].upper() + token[1:] if str(locale).startswith("en") else token)
    return " ".join(rendered) or raw


def friendly_input_reference(value: Any, *, locale: str = "ja") -> str:
    """Convert a YAML/model path to a concise user-facing field name."""

    raw = str(value or "").strip()
    if not raw:
        return raw
    artifact = _RESULT_ARTIFACT_LABELS.get(raw.replace("\\", "/"))
    if artifact is not None:
        return artifact[1] if str(locale).startswith("en") else artifact[0]
    prefix, separator, suffix = raw.partition(":")
    path = prefix.strip()
    if not re.fullmatch(r"[A-Za-z_][\w-]*(?:\[[^\]]+\]|\.[\w-]+|\[\d+\])*", path):
        return raw

    tokens = [token for token in re.split(r"\.|\[(?:'|\")?([^\]\"']+)(?:'|\")?\]", path) if token]
    if not tokens:
        return raw
    root = tokens[0]
    root_pair = _ROOT_LABELS.get(root, (root.replace("_", " "), root.replace("_", " ")))
    root_label = root_pair[1] if str(locale).startswith("en") else root_pair[0]
    leaf = tokens[-1]
    field_pair = _FIELD_LABELS.get(leaf)
    field_label = (field_pair[1] if str(locale).startswith("en") else field_pair[0]) if field_pair else ""

    if root == "materials" and len(tokens) >= 2:
        material_name = tokens[1]
        if field_label and len(tokens) >= 3:
            friendly = f"Material '{material_name}': {field_label}" if str(locale).startswith("en") else f"材料「{material_name}」の{field_label}"
        else:
            friendly = f"Material '{material_name}'" if str(locale).startswith("en") else f"材料「{material_name}」"
    elif field_label and field_label != root_label:
        friendly = f"{root_label}: {field_label}" if str(locale).startswith("en") else f"{root_label}の{field_label}"
    else:
        friendly = root_label
    if separator and suffix.strip():
        friendly = f"{friendly}: {suffix.strip()}"
    return friendly
