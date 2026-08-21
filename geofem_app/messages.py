"""Centralized UI/report/diagnostic messages for GeoFEM.

The solver and data contracts should not depend on display text.  This module
keeps user-facing wording behind stable keys so wording changes can be tested
without changing analysis behavior.
"""

from __future__ import annotations

from typing import Any, Mapping


DEFAULT_LOCALE = "ja"


MESSAGES: dict[str, dict[str, str]] = {
    "ja": {
        "diagnostics.unused_top_level_key.message": "未使用の可能性があるトップレベルキーです。",
        "diagnostics.unused_top_level_key.suggestion": "綴りを確認し、必要なら analysis、mesh、materials、steps などの既知キーへ移してください。",
        "diagnostics.dimension.message": "GeoFEM 2D では 2D 解析のみ対応しています。",
        "diagnostics.dimension.suggestion": "analysis.dimension を 2D にしてください。",
        "diagnostics.unit_system.message": "単位系が標準候補に含まれていません。",
        "diagnostics.unit_system.suggestion": "m-kN、m-N、SI、mm-N、user/custom のいずれかを明示することを推奨します。",
        "diagnostics.mesh.missing.message": "メッシュ定義がありません。",
        "diagnostics.mesh.missing.suggestion": "mesh に generator または nodes/elements を定義してください。",
        "diagnostics.mesh.generator_or_explicit.message": "メッシュ生成方法または明示メッシュが不足しています。",
        "diagnostics.mesh.generator_or_explicit.suggestion": "mesh.generator または mesh.nodes/mesh.elements を定義してください。",
        "diagnostics.mesh.unsupported_element.message": "未対応の2D要素タイプです。",
        "diagnostics.mesh.unsupported_element.suggestion": "対応要素: {supported}",
        "diagnostics.mesh.unsupported_integration.message": "積分指定が標準候補に含まれていません。",
        "diagnostics.mesh.unsupported_integration.suggestion": "対応候補: {supported}",
        "diagnostics.mesh.division_positive.message": "分割数は正の整数である必要があります。",
        "diagnostics.mesh.division_positive.suggestion": "{name} を 1 以上にしてください。",
        "diagnostics.mesh.division_integer.message": "分割数を整数として解釈できません。",
        "diagnostics.mesh.division_integer.suggestion": "{name} を整数で指定してください。",
        "diagnostics.materials.missing.message": "材料定義がありません。",
        "diagnostics.materials.missing.suggestion": "materials に少なくとも1つの材料を定義してください。",
        "diagnostics.material.mapping.message": "材料定義はマッピングである必要があります。",
        "diagnostics.material.mapping.suggestion": "E、nu、model などを持つ辞書にしてください。",
        "diagnostics.material.required.message": "必須材料パラメータがありません。",
        "diagnostics.material.required.suggestion": "{key} を指定してください。",
        "diagnostics.material.positive.message": "正の値が必要です。",
        "diagnostics.material.positive.suggestion": "{key} を 0 より大きくしてください。",
        "diagnostics.material.nu_range.message": "ポアソン比は 0 <= nu < 0.5 である必要があります。",
        "diagnostics.material.nu_range.suggestion": "nu を有効範囲にしてください。",
        "diagnostics.mesh.material_missing.message": "メッシュの既定材料が materials に存在しません。",
        "diagnostics.mesh.material_missing.suggestion": "materials に追加するか、mesh.material を修正してください。",
        "diagnostics.mesh.element_material_missing.message": "要素の材料参照が materials に存在しません。",
        "diagnostics.mesh.element_material_missing.suggestion": "材料名を修正するか materials に追加してください。",
        "diagnostics.bc.mapping.message": "境界条件はマッピングである必要があります。",
        "diagnostics.bc.mapping.suggestion": "set、node、nodes のいずれかと拘束値を指定してください。",
        "diagnostics.bc.unknown_set.message": "存在しない節点セットを参照しています。",
        "diagnostics.bc.unknown_set.suggestion": "sets.nodes または mesh の既定セット名を確認してください。",
        "diagnostics.bc.no_component.message": "拘束成分が見つかりません。",
        "diagnostics.bc.no_component.suggestion": "ux、uy、fixed などを指定してください。",
        "diagnostics.loads.mapping.message": "荷重はマッピングである必要があります。",
        "diagnostics.loads.mapping.suggestion": "node、set、edge、type などを指定してください。",
        "diagnostics.loads.unknown_set.message": "存在しないセットを参照しています。",
        "diagnostics.loads.unknown_set.suggestion": "sets.nodes または sets.elements を確認してください。",
        "diagnostics.loads.no_component.message": "荷重成分または対象が見つかりません。",
        "diagnostics.loads.no_component.suggestion": "荷重成分と対象を指定してください。",
        "diagnostics.stage.mapping.message": "ステージはマッピングである必要があります。",
        "diagnostics.stage.mapping.suggestion": "name、type、loads、boundary_conditions などを持つ辞書にしてください。",
        "diagnostics.stage.unsupported_type.message": "未対応のステージ種別です。",
        "diagnostics.stage.unsupported_type.suggestion": "static、geostatic、excavation、consolidation、riks、srm、dynamic などを指定してください。",
        "diagnostics.stage.unknown_set.message": "存在しないセットを参照しています。",
        "diagnostics.stage.unknown_set.suggestion": "ステージ対象セットを確認してください。",
        "diagnostics.constraints.none.message": "境界条件がありません。剛体運動で特異行列になる可能性があります。",
        "diagnostics.constraints.none.suggestion": "少なくとも水平・鉛直の剛体運動を止める拘束を設定してください。",
        "diagnostics.constraints.no_x.message": "水平剛体運動を止める拘束が見つかりません。",
        "diagnostics.constraints.no_x.suggestion": "ux または fixed を含む拘束を追加してください。",
        "diagnostics.constraints.no_y.message": "鉛直剛体運動を止める拘束が見つかりません。",
        "diagnostics.constraints.no_y.suggestion": "uy または fixed を含む拘束を追加してください。",
        "diagnostics.constraints.over_fixed.message": "全節点拘束の可能性があります。",
        "diagnostics.constraints.over_fixed.suggestion": "解析目的に対して過拘束でないか確認してください。",
        "diagnostics.html.title": "GeoFEM 入力診断",
        "diagnostics.html.heading": "入力診断",
        "reports.standard.title": "GeoFEM 2D 標準帳票",
        "reports.standard.input": "入力条件",
        "reports.standard.mesh_quality": "メッシュ品質",
        "reports.standard.stages": "ステージ",
        "reports.standard.performance": "性能",
        "startup.heading": "GeoFEM 起動確認",
        "startup.completed": "起動確認を完了しました。",
    },
    "en": {
        "diagnostics.unused_top_level_key.message": "This top-level key may be unused.",
        "diagnostics.unused_top_level_key.suggestion": "Check the spelling or move it under a known key such as analysis, mesh, materials, or steps.",
        "diagnostics.dimension.message": "GeoFEM 2D supports only 2D analyses.",
        "diagnostics.dimension.suggestion": "Set analysis.dimension to 2D.",
        "diagnostics.unit_system.message": "The unit system is not in the standard candidates.",
        "diagnostics.unit_system.suggestion": "Use m-kN, m-N, SI, mm-N, user, or custom.",
        "diagnostics.mesh.missing.message": "Mesh definition is missing.",
        "diagnostics.mesh.missing.suggestion": "Define mesh.generator or mesh.nodes/mesh.elements.",
        "diagnostics.mesh.generator_or_explicit.message": "Mesh generator or explicit mesh data is missing.",
        "diagnostics.mesh.generator_or_explicit.suggestion": "Define mesh.generator or mesh.nodes/mesh.elements.",
        "diagnostics.mesh.unsupported_element.message": "Unsupported 2D element type.",
        "diagnostics.mesh.unsupported_element.suggestion": "Supported elements: {supported}",
        "diagnostics.mesh.unsupported_integration.message": "Integration option is not in the standard candidates.",
        "diagnostics.mesh.unsupported_integration.suggestion": "Supported options: {supported}",
        "diagnostics.mesh.division_positive.message": "The division count must be a positive integer.",
        "diagnostics.mesh.division_positive.suggestion": "Set {name} to 1 or greater.",
        "diagnostics.mesh.division_integer.message": "The division count cannot be parsed as an integer.",
        "diagnostics.mesh.division_integer.suggestion": "Specify {name} as an integer.",
        "diagnostics.materials.missing.message": "Material definitions are missing.",
        "diagnostics.materials.missing.suggestion": "Define at least one material under materials.",
        "diagnostics.material.mapping.message": "A material definition must be a mapping.",
        "diagnostics.material.mapping.suggestion": "Use a dictionary with E, nu, model, and related parameters.",
        "diagnostics.material.required.message": "A required material parameter is missing.",
        "diagnostics.material.required.suggestion": "Specify {key}.",
        "diagnostics.material.positive.message": "A positive value is required.",
        "diagnostics.material.positive.suggestion": "Set {key} greater than 0.",
        "diagnostics.material.nu_range.message": "Poisson's ratio must satisfy 0 <= nu < 0.5.",
        "diagnostics.material.nu_range.suggestion": "Set nu in the valid range.",
        "diagnostics.mesh.material_missing.message": "The mesh default material is not present in materials.",
        "diagnostics.mesh.material_missing.suggestion": "Add it to materials or fix mesh.material.",
        "diagnostics.mesh.element_material_missing.message": "An element references a material that is not present in materials.",
        "diagnostics.mesh.element_material_missing.suggestion": "Fix the material name or add it to materials.",
        "diagnostics.bc.mapping.message": "A boundary condition must be a mapping.",
        "diagnostics.bc.mapping.suggestion": "Specify set, node, or nodes plus constraint values.",
        "diagnostics.bc.unknown_set.message": "The boundary condition references an unknown node set.",
        "diagnostics.bc.unknown_set.suggestion": "Check sets.nodes or built-in mesh set names.",
        "diagnostics.bc.no_component.message": "No constrained component was found.",
        "diagnostics.bc.no_component.suggestion": "Specify ux, uy, fixed, or similar fields.",
        "diagnostics.loads.mapping.message": "A load must be a mapping.",
        "diagnostics.loads.mapping.suggestion": "Specify node, set, edge, type, or related fields.",
        "diagnostics.loads.unknown_set.message": "The load references an unknown set.",
        "diagnostics.loads.unknown_set.suggestion": "Check sets.nodes or sets.elements.",
        "diagnostics.loads.no_component.message": "No load component or target was found.",
        "diagnostics.loads.no_component.suggestion": "Specify load components and targets.",
        "diagnostics.stage.mapping.message": "A stage must be a mapping.",
        "diagnostics.stage.mapping.suggestion": "Use a dictionary with name, type, loads, or boundary_conditions.",
        "diagnostics.stage.unsupported_type.message": "Unsupported stage type.",
        "diagnostics.stage.unsupported_type.suggestion": "Use static, geostatic, excavation, consolidation, riks, srm, dynamic, or related aliases.",
        "diagnostics.stage.unknown_set.message": "The stage references an unknown set.",
        "diagnostics.stage.unknown_set.suggestion": "Check the stage target set.",
        "diagnostics.constraints.none.message": "No boundary condition is defined, so rigid-body modes may make the matrix singular.",
        "diagnostics.constraints.none.suggestion": "Add enough constraints to prevent horizontal and vertical rigid-body motion.",
        "diagnostics.constraints.no_x.message": "No constraint prevents horizontal rigid-body motion.",
        "diagnostics.constraints.no_x.suggestion": "Add a constraint containing ux or fixed.",
        "diagnostics.constraints.no_y.message": "No constraint prevents vertical rigid-body motion.",
        "diagnostics.constraints.no_y.suggestion": "Add a constraint containing uy or fixed.",
        "diagnostics.constraints.over_fixed.message": "All-node constraints may be present.",
        "diagnostics.constraints.over_fixed.suggestion": "Check whether the model is over-constrained for the analysis purpose.",
        "diagnostics.html.title": "GeoFEM input diagnostics",
        "diagnostics.html.heading": "Input diagnostics",
        "reports.standard.title": "GeoFEM 2D Standard Report",
        "reports.standard.input": "Input",
        "reports.standard.mesh_quality": "Mesh Quality",
        "reports.standard.stages": "Stages",
        "reports.standard.performance": "Performance",
        "startup.heading": "GeoFEM startup check",
        "startup.completed": "Startup check completed.",
    },
}


def message(message_key: str, *, locale: str = DEFAULT_LOCALE, **values: Any) -> str:
    """Return a localized message by stable key."""

    table = MESSAGES.get(locale, MESSAGES[DEFAULT_LOCALE])
    text = table.get(message_key, MESSAGES[DEFAULT_LOCALE].get(message_key, message_key))
    if values:
        try:
            return text.format(**values)
        except Exception:
            return text
    return text


def message_catalog(locale: str = DEFAULT_LOCALE) -> dict[str, str]:
    """Return a copy of the message catalog for a locale."""

    return dict(MESSAGES.get(locale, MESSAGES[DEFAULT_LOCALE]))


def available_message_keys(locale: str = DEFAULT_LOCALE) -> list[str]:
    """Return stable message keys for tests and documentation generators."""

    return sorted(message_catalog(locale))


def localize_mapping(keys: Mapping[str, str], *, locale: str = DEFAULT_LOCALE) -> dict[str, str]:
    """Resolve a small mapping of logical names to message keys."""

    return {name: message(key, locale=locale) for name, key in keys.items()}


__all__ = [
    "DEFAULT_LOCALE",
    "MESSAGES",
    "available_message_keys",
    "localize_mapping",
    "message",
    "message_catalog",
]
