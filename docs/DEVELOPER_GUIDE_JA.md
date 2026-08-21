# GeoFEM 開発者ガイド

更新日: 2026-05-23

## 主要モジュールの責務

- `fem2d_types.py`: 2D FEM のデータ型、要素/ステージ種別、例外。
- `fem2d_mesh.py`: YAML/JSON 入力から `Mesh2D` を構築する。
- `fem2d_materials.py`: 材料パラメータを正規化し、構成則へ渡す。
- `fem2d_elements.py`: 要素剛性、B 行列、Numba 高速化カーネル。
- `fem2d_solver.py`: ステージ解析、非線形解析、動的解析、圧密、SRM。
- `sparse_assembly.py`: 要素ブロックをNumPy配列チャンクとして集約する疎行列組立ヘルパー。
- `fem2d_io.py`: CSV/VTK/HTML/JSON 出力と `summary.json`。
- `mesh_quality.py`: メッシュ品質評価と修復候補。
- `material_models.py`: 材料モデルカタログと入力フォーム用スキーマ。
- `analysis_log.py`: 解析イベントログ。
- `performance_monitor.py`: 実行時間、反復数、行列規模、性能回帰。
- `html_report_utils.py`: 帳票HTMLのエスケープ、表、リンク、共通CSS。
- `pdf_writer.py`: 外部依存なしのテキストPDF生成。標準帳票と計算書で共有する。
- `standard_report.py`: 1つの標準データから HTML/PDF/CSV 帳票を生成する。
- `api_contracts.py`: 主要境界の契約と検証。
- `messages.py`: 診断、帳票、起動確認の文言カタログ。
- `geofeas_seepage.py`: GeoFEAS/UC-1/VGFlow 風の浸透・水圧CSV正規化と往復比較。
- `gui/result_table_routes.py`: GUI 結果表の種別から成果物CSVへのルーティング。
- `maintainability_audit.py`: 巨大ファイルと責務混在候補の監査。
- `gui/main_window.py`: PySide6 GUI。解析ロジックは直接持たず、アプリ層 API を呼び出す。

## 追加実装の手順

1. 入力キーを増やす場合は `input_diagnostics.py` と `docs/API_CONTRACTS_JA.md` の該当契約を確認する。
2. ソルバーの結果に新しい項目を追加する場合は `StageResult2D` または `SolveResult2D` の責務範囲に収まるか確認する。
3. Post/帳票の出力を増やす場合は `summary.json` と `result_view_index.json` から到達できるようにする。
4. 表示文言を増やす場合は `messages.py` にキーを追加し、解析ロジックへ文字列を埋め込まない。
5. 性能に影響する変更は標準ベンチを更新し、必要なら baseline を更新する。

## テスト構成

- `tests/test_fem2d.py`: 2D FEM コア、構成則、ステージ解析、出力。
- `tests/test_import_and_mesh.py`: CAD/メッシュ/GUI モデルチェック。
- `tests/test_geofem_completion_backlog.py`: 完成度向上項目、診断、起動確認、API 契約、帳票、性能。
- `tests/test_geofeas_*.py`: GeoFEAS 公開情報ベースの代替実装・監査。

## 回帰コマンド

```powershell
python -W error::ResourceWarning -m unittest discover -s tests -p "test_*.py" -v
python tools\run_geofeas_benchmarks.py --out runs\geofeas_public_compat
python -m geofem_app.cli benchmarks --out runs\geofem_standard_benchmarks --update-baseline
python -m geofem_app.cli doctor --out runs\startup_check
python -m geofem_app.cli maintainability-audit --out runs\maintainability_audit
```

## 性能測定

標準ベンチは、ケースごとの経過時間、反復数、最大行列非ゼロ数、推定メモリに加え、疎行列builder fallbackと重複scatterの使用回数を出力します。診断カウンタはベンチ区間だけ有効で、通常解析には計測ロックを持ち込みません。

```powershell
python -m geofem_app.cli benchmarks --out runs\perf_current --baseline runs\geofem_standard_benchmarks\standard_benchmark_performance_baseline.json
```

時間、行列構造、メモリ、fallbackの許容値はそれぞれ `--max-slowdown`、`--max-structure-growth`、`--max-memory-growth`、`--max-fallback-increase` で指定できます。理由が説明できない性能低下は baseline に取り込まないでください。
