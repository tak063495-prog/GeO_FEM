"""VGFlow 2D-like UI state/profile exports for the public substitute."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fem2d_types import Mesh2D
from .fem2d_utils import _ensure_list
from .html_report_utils import html_escape, report_css, table


def write_vgflow_ui_profile_outputs(
    out: Path,
    mesh: Mesh2D,
    seepage: Mapping[str, Any],
    artifacts: Mapping[str, str],
) -> dict[str, str]:
    paths = {
        "ui_profile_json": str(out / "vgflow_ui_profile.json"),
        "ui_profile_csv": str(out / "vgflow_ui_profile.csv"),
        "ui_profile_html": str(out / "vgflow_ui_profile.html"),
    }
    profile = vgflow_ui_profile(mesh, seepage, artifacts)
    Path(paths["ui_profile_json"]).write_text(json.dumps(profile, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_profile_csv(Path(paths["ui_profile_csv"]), profile)
    Path(paths["ui_profile_html"]).write_text(_profile_html(profile), encoding="utf-8")
    return paths


def vgflow_ui_profile(mesh: Mesh2D, seepage: Mapping[str, Any], artifacts: Mapping[str, str]) -> dict[str, Any]:
    pre = _mapping(seepage, "pre", "vgflow_pre", "preprocess", "pre_operation")
    post = _mapping(seepage, "post", "vgflow_post")
    report = _mapping(seepage, "report", "vgflow_report")
    model_ready = len(mesh.node_ids) > 0 and len(mesh.elements) > 0
    has_boundary = any(_ensure_list(seepage.get(key)) for key in ("known_head_bcs", "head_boundaries", "rainfall", "flux_boundaries", "flow_boundaries", "seepage_faces"))
    run_ready = model_ready and has_boundary
    transient = str(seepage.get("mode", seepage.get("analysis_mode", "steady"))).lower() == "transient"
    return {
        "schema": "geofem.vgflow2d.ui_profile.public_substitute.v1",
        "profile": "VGFlow 2D public-operation and GeoFEAS-like UI substitute",
        "source_policy": {
            "primary": "FORUM8 public product page and operation guidance PDF",
            "fallback": "GeoFEAS-like workflow where VGFlow 2D commercial UI details are not public",
            "commercial_pixel_equivalence": False,
        },
        "state": {
            "analysis_type": str(seepage.get("analysis_type", "saturated_unsaturated_seepage")),
            "analysis_mode": str(seepage.get("mode", seepage.get("analysis_mode", "steady"))),
            "problem_type": str(seepage.get("problem_type", "vertical")),
            "mesh_mode": str(pre.get("mesh_mode", seepage.get("mesh_mode", seepage.get("vgflow_mesh_mode", "auto_mixed")))),
            "model_ready": model_ready,
            "boundary_ready": has_boundary,
            "run_ready": run_ready,
            "post_ready": all(key in artifacts for key in ("post_contours", "flow_vectors", "flowlines")),
            "report_ready": "vgflow_report_manifest" in artifacts,
        },
        "toolbar_groups": _toolbar_groups(model_ready, run_ready, transient),
        "context_menus": _context_menus(),
        "modal_tables": _modal_tables(pre, post, report),
        "button_enablement": _button_enablement(model_ready, has_boundary, run_ready, transient),
        "selection_palette": _selection_palette(seepage),
        "screen_transitions": _screen_transitions(transient),
        "post_playback": _post_playback(post, artifacts, transient),
        "report_operations": _report_operations(report),
        "artifacts": {key: value for key, value in sorted(artifacts.items()) if key.startswith(("pre_", "mesh_", "post_", "flow", "section", "time_history", "vgflow_report"))},
    }


def _toolbar_groups(model_ready: bool, run_ready: bool, transient: bool) -> list[dict[str, Any]]:
    return [
        {
            "id": "file",
            "label": "ファイル",
            "movable": True,
            "commands": [
                _command("new_project", "新規入力", True, "空のモデルを作成する"),
                _command("open_project", "開く", True, "公開代替YAML/JSONまたはCAD/CSVを開く"),
                _command("save_project", "保存", model_ready, "公開代替プロジェクトを保存する"),
            ],
        },
        {
            "id": "basic_conditions",
            "label": "基本条件",
            "movable": True,
            "commands": [
                _command("mesh_mode", "メッシュモード", True, "オート混合/四角形のみ/三角形のみ/セミオートを選択する"),
                _command("problem_type", "解析種別", True, "鉛直問題/軸対称/水平問題を選択する"),
                _command("analysis_mode", "解析モード", True, "定常/非定常を選択する"),
            ],
        },
        {
            "id": "model_mesh",
            "label": "モデル・メッシュ",
            "movable": True,
            "commands": [
                _command("grid_setting", "グリッド設定", True, "作成タブのグリッド表示と間隔を設定する"),
                _command("snap_toggle", "スナップ", True, "節点・線分の入力補助を切り替える"),
                _command("block_decision", "ブロック化", model_ready, "閉じた地層領域をブロックとして決定する"),
                _command("mesh_split", "メッシュ分割", model_ready, "分割数を反映してメッシュを作成する"),
            ],
        },
        {
            "id": "analysis_post",
            "label": "解析・Post",
            "movable": True,
            "commands": [
                _command("boundary_condition", "境界条件", model_ready, "水頭既知/降雨/流量/浸出面を設定する"),
                _command("run", "解析実行", run_ready, "収束条件と出力先を確認して解析する"),
                _command("post_play", "時刻歴再生", run_ready and transient, "非定常結果のコンター/流線タイムラインを再生する"),
                _command("report_preview", "計算書プレビュー", run_ready, "Pre/Post項目を選択して帳票を表示する"),
            ],
        },
    ]


def _context_menus() -> list[dict[str, Any]]:
    return [
        {
            "id": "model_canvas",
            "target": "モデル作成キャンバス",
            "items": [
                _menu_item("insert_node", "節点追加", "クリック位置または数値座標で節点を追加する"),
                _menu_item("register_line", "直線登録", "選択した2点をメッシュ分割用補助線に登録する"),
                _menu_item("set_origin", "原点に設定", "選択節点をローカル原点として扱う"),
                _menu_item("node_coordinate_correction", "座標修正", "節点座標の数値表を開く"),
            ],
        },
        {
            "id": "mesh_selection",
            "target": "線分・ブロック選択",
            "items": [
                _menu_item("select_single", "選択モード", "線分またはブロックを個別選択する"),
                _menu_item("box_crossing", "矩形選択（掛け）", "矩形に交差する対象を選択する"),
                _menu_item("box_inside", "矩形選択（囲み）", "矩形内の対象を選択する"),
                _menu_item("division_setting", "分割数の設定", "分割数または分割幅の入力表を開く"),
            ],
        },
        {
            "id": "boundary_selection",
            "target": "解析条件キャンバス",
            "items": [
                _menu_item("known_head", "水頭既知境界", "選択辺へ水頭値または時刻曲線を割り当てる"),
                _menu_item("rainfall", "降雨境界", "選択辺へ降雨強度を割り当てる"),
                _menu_item("flux", "流量境界", "選択辺または節点へ流量を割り当てる"),
                _menu_item("seepage_face", "浸出面境界", "圧力水頭上限を持つ浸出面を設定する"),
            ],
        },
        {
            "id": "post_canvas",
            "target": "Post表示",
            "items": [
                _menu_item("probe_value", "数値確認", "節点/要素の値確認表へ選択行を送る"),
                _menu_item("copy_table", "表をコピー", "TSV形式でクリップボード相当の表を出力する"),
                _menu_item("export_frame", "表示フレーム出力", "選択時刻の図化データをCSV/HTMLに保存する"),
                _menu_item("video_export", "動画化", "HTMLタイムラインまたは外部エンコーダ用手順を開く"),
            ],
        },
    ]


def _modal_tables(pre: Mapping[str, Any], post: Mapping[str, Any], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "material_property",
            "label": "浸透要素のプロパティ設定",
            "columns": ["材料名", "Kx", "Ky", "Ss", "地層傾斜角", "不飽和モデル"],
            "commit_button": "確定",
        },
        {
            "id": "unsaturated_property",
            "label": "不飽和浸透特性",
            "columns": ["alpha", "n", "theta_r", "theta_s", "表入力theta", "表入力Kr"],
            "commit_button": "確定",
        },
        {
            "id": "line_division",
            "label": "分割数の設定",
            "columns": ["線分", "分割数", "分割幅", "選択状態"],
            "default_rows": len(_ensure_list(pre.get("line_divisions", []))),
            "commit_button": "確定",
        },
        {
            "id": "boundary_curve",
            "label": "境界条件値編集",
            "columns": ["時刻", "水頭", "降雨", "流量", "単位"],
            "commit_button": "確定",
        },
        {
            "id": "post_value_check",
            "label": "数値確認",
            "columns": ["Step", "Time", "ID", "水頭", "間隙水圧", "飽和度", "流速", "動水勾配"],
            "copy_format": "TSV",
            "default_node_selection": _ensure_list(post.get("history_nodes", [])),
        },
        {
            "id": "print_item_setting",
            "label": "印刷項目設定",
            "columns": ["項目", "出力", "節点番号", "要素番号", "時刻範囲"],
            "post_apply": _ensure_list(report.get("post_apply", report.get("print_profile", {}).get("post_apply", []))) if isinstance(report.get("print_profile", {}), Mapping) else [],
            "commit_button": "プレビュー",
        },
    ]


def _button_enablement(model_ready: bool, has_boundary: bool, run_ready: bool, transient: bool) -> list[dict[str, Any]]:
    return [
        _enable_rule("block_decision", model_ready, "モデル線またはCAD/ラスタ由来の閉合候補がある"),
        _enable_rule("element_property", model_ready, "ブロック化または要素集合がある"),
        _enable_rule("mesh_split", model_ready, "メッシュモードと分割条件が入力済みである"),
        _enable_rule("boundary_condition", model_ready, "メッシュ分割結果がある"),
        _enable_rule("run", run_ready, "材料・メッシュ・境界条件がそろっている"),
        _enable_rule("transient_playback", run_ready and transient, "非定常解析結果がある"),
        _enable_rule("report_preview", run_ready, "解析結果または入力データがある"),
        _enable_rule("video_export", run_ready and transient, "非定常Postフレームが作成済みである"),
        _enable_rule("copy_selected_values", run_ready, "Postの節点/要素表が作成済みである"),
    ]


def _selection_palette(seepage: Mapping[str, Any]) -> dict[str, str]:
    raw = _mapping(seepage, "ui", "vgflow_ui").get("selection_palette", {})
    defaults = {
        "canvas_background": "#ffffff",
        "mesh_line": "#334155",
        "selected_line": "#dc2626",
        "selected_block": "#fde68a",
        "material_fill": "#bfdbfe",
        "known_head_boundary": "#2563eb",
        "rainfall_boundary": "#16a34a",
        "flux_boundary": "#c2410c",
        "seepage_face": "#7c3aed",
        "post_probe": "#f59e0b",
        "warning": "#ef4444",
    }
    if isinstance(raw, Mapping):
        defaults.update({str(key): str(value) for key, value in raw.items()})
    return defaults


def _screen_transitions(transient: bool) -> list[dict[str, Any]]:
    transitions = [
        _transition("new_project", "basic_conditions", "mesh_mode/problem_type/analysis_modeを設定"),
        _transition("basic_conditions", "model_creation", "CAD/手入力/ラスタ校正で形状線を作成"),
        _transition("model_creation", "model_decision", "ブロック化とハッチング確認"),
        _transition("model_decision", "element_definition", "浸透材料と不飽和特性を設定"),
        _transition("element_definition", "mesh_definition", "線分の分割数または分割幅を設定"),
        _transition("mesh_definition", "mesh_confirmation", "メッシュ分割結果を確認"),
        _transition("mesh_confirmation", "boundary_conditions", "水頭既知/降雨/流量/浸出面を設定"),
    ]
    if transient:
        transitions.append(_transition("boundary_conditions", "initial_wetting_surface", "初期浸潤面を設定"))
        transitions.append(_transition("initial_wetting_surface", "run", "非定常解析条件を確認"))
    else:
        transitions.append(_transition("boundary_conditions", "run", "定常解析条件を確認"))
    transitions.extend(
        [
            _transition("run", "post", "コンター/流線/ベクトル/数値確認を表示"),
            _transition("post", "report", "Pre/Post印刷項目を選択してプレビュー"),
            _transition("report", "save", "公開代替プロジェクトと帳票を保存"),
        ]
    )
    return transitions


def _post_playback(post: Mapping[str, Any], artifacts: Mapping[str, str], transient: bool) -> dict[str, Any]:
    return {
        "enabled": transient,
        "controls": [
            _menu_item("first_frame", "先頭", "最初の時刻へ移動"),
            _menu_item("previous_frame", "戻る", "1フレーム戻る"),
            _menu_item("play_pause", "再生/停止", "時刻歴を再生または停止"),
            _menu_item("next_frame", "進む", "1フレーム進む"),
            _menu_item("last_frame", "最終", "最後の時刻へ移動"),
            _menu_item("speed", "速度", "再生間隔を選択する"),
        ],
        "frame_sources": {
            "contours": artifacts.get("post_contours", ""),
            "flowlines": artifacts.get("flowlines", ""),
            "vectors": artifacts.get("flow_vectors", ""),
            "animation_manifest": artifacts.get("post_animation_manifest", ""),
        },
        "video_export": {
            "format": str(post.get("video_format", "avi")),
            "mode": "public_substitute_direct_avi_and_external_encoder_profile",
            "direct_public_avi_binary": bool(artifacts.get("post_animation_avi")),
            "direct_commercial_avi_binary": False,
            "recommended_command": str(post.get("video_command", "ffmpeg -r 6 -i frame_%04d.png -c:v mjpeg vgflow_post_animation.avi")),
        },
    }


def _report_operations(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    sections = _ensure_list(report.get("sections", [])) or ["model", "mesh", "analysis", "materials", "boundaries", "post_outputs", "time_history"]
    return [
        {"id": "open_pre_print_items", "label": "Pre部印刷項目設定", "enabled_rule": "設計データがある", "sections": ["model", "mesh", "materials", "boundaries"]},
        {"id": "open_post_print_items", "label": "Post部印刷項目設定", "enabled_rule": "解析結果がある", "sections": sections},
        {"id": "preview", "label": "プレビュー", "enabled_rule": "少なくとも1つの帳票項目が有効", "sections": sections},
        {"id": "export_html_pdf", "label": "HTML/PDF出力", "enabled_rule": "プレビュー可能", "sections": sections},
    ]


def _write_profile_csv(path: Path, profile: Mapping[str, Any]) -> None:
    fields = ["section", "group", "id", "label", "enabled", "enabled_rule", "action", "source"]
    rows = _flatten_profile_rows(profile)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _flatten_profile_rows(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in profile.get("toolbar_groups", []):
        for command in group.get("commands", []):
            rows.append({"section": "toolbar", "group": group.get("id", ""), **command, "source": "public_pdf_toolbar_and_geofeas_like_fallback"})
    for menu in profile.get("context_menus", []):
        for item in menu.get("items", []):
            rows.append({"section": "context_menu", "group": menu.get("id", ""), **item, "source": "public_pdf_selection_flow_and_geofeas_like_fallback"})
    for modal in profile.get("modal_tables", []):
        rows.append({"section": "modal_table", "group": modal.get("id", ""), "id": modal.get("id", ""), "label": modal.get("label", ""), "enabled": True, "enabled_rule": "opened_by_related_command", "action": ",".join(str(col) for col in modal.get("columns", [])), "source": "public_pdf_tables_and_geofeas_like_fallback"})
    for rule in profile.get("button_enablement", []):
        rows.append({"section": "button_enablement", "group": "state", **rule, "source": "public_substitute_state_machine"})
    for transition in profile.get("screen_transitions", []):
        rows.append({"section": "screen_transition", "group": transition.get("from", ""), "id": transition.get("to", ""), "label": f"{transition.get('from', '')} -> {transition.get('to', '')}", "enabled": True, "action": transition.get("action", ""), "source": "public_pdf_flowchart"})
    return rows


def _profile_html(profile: Mapping[str, Any]) -> str:
    toolbar_rows = [[group["id"], group["label"], ", ".join(command["label"] for command in group["commands"])] for group in profile["toolbar_groups"]]
    menu_rows = [[menu["id"], menu["target"], ", ".join(item["label"] for item in menu["items"])] for menu in profile["context_menus"]]
    enable_rows = [[row["id"], row["label"], row["enabled"], row["enabled_rule"]] for row in profile["button_enablement"]]
    transition_rows = [[row["from"], row["to"], row["action"]] for row in profile["screen_transitions"]]
    palette_rows = [[key, value] for key, value in profile["selection_palette"].items()]
    return (
        "<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\">"
        "<title>VGFlow 2D UI Profile</title>"
        f"<style>{report_css()}</style></head><body>"
        "<h1>VGFlow 2D UI Profile Substitute</h1>"
        "<p>商用画面のピクセル完全一致ではなく、公開操作情報とGeoFEAS風操作で定義した本ツール内UI状態プロファイルです。</p>"
        f"<h2>状態</h2><pre>{html_escape(json.dumps(profile['state'], ensure_ascii=False, indent=2))}</pre>"
        f"<h2>ツールバー</h2>{table(['group','label','commands'], toolbar_rows)}"
        f"<h2>右クリック/選択メニュー</h2>{table(['id','target','items'], menu_rows)}"
        f"<h2>ボタン有効化</h2>{table(['id','label','enabled','rule'], enable_rows)}"
        f"<h2>画面遷移</h2>{table(['from','to','action'], transition_rows)}"
        f"<h2>選択色</h2>{table(['role','color'], palette_rows)}"
        "</body></html>"
    )


def _mapping(mapping: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _command(command_id: str, label: str, enabled: bool, action: str) -> dict[str, Any]:
    return {"id": command_id, "label": label, "enabled": bool(enabled), "action": action}


def _menu_item(item_id: str, label: str, action: str) -> dict[str, str]:
    return {"id": item_id, "label": label, "action": action}


def _enable_rule(command_id: str, enabled: bool, rule: str) -> dict[str, Any]:
    return {"id": command_id, "label": command_id, "enabled": bool(enabled), "enabled_rule": rule, "action": "enable_command" if enabled else "disable_command"}


def _transition(source: str, target: str, action: str) -> dict[str, str]:
    return {"from": source, "to": target, "action": action}


__all__ = ["vgflow_ui_profile", "write_vgflow_ui_profile_outputs"]
