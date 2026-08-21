# GeoFEM 内部 API 契約

更新日: 2026-05-23

GeoFEM は責務境界を次の契約として扱います。契約は `geofem_app.api_contracts` で機械的に取得・検証できます。

## 契約の生成

```powershell
python -m geofem_app.cli api-contracts --out runs\api_contracts
```

生成物:

- `api_contracts.json`
- `api_contracts.csv`
- `api_contracts.md`

## 境界

- `input_config`: CLI/GUI/YAML loader から診断、メッシュ、ソルバー、帳票へ渡す入力設定。
- `mesh2d`: メッシュ生成からソルバー、品質診断、Post、帳票へ渡す節点・要素構造。
- `material_table`: 材料設定からソルバー、帳票、材料カタログへ渡す材料表。
- `stage_result2d`: ソルバーから Post、解析ログ、性能測定、帳票へ渡すステージ結果。
- `solve_result2d`: 解析実行全体の結果を出力、GUI、ベンチへ渡す統合結果。
- `analysis_artifact_bundle`: `summary.json`、結果インデックス、標準帳票、性能情報の成果物束。

## 契約テスト

契約を壊す変更は、次のテストで検出します。

```powershell
python -m unittest tests.test_geofem_completion_backlog.GeoFEMCompletionBacklogTests.test_api_contracts_validate_public_boundaries_and_write_docs -v
```

追加実装時は、境界をまたぐデータのキー名、型、必須項目を変える前に `api_contracts.py` の契約バージョンを更新します。
