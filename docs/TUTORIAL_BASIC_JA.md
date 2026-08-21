# GeoFEM 基本チュートリアル

更新日: 2026-05-23

このチュートリアルは、初見ユーザーが最小モデル作成から結果確認までを一通り完了するための短い手順です。

## 1. サンプル入力を作成する

```powershell
cd C:\Users\link_\Downloads\WORK\GeoFEM
python -m geofem_app.cli sample runs\tutorial\input.yaml --kind quad4
```

作成される入力は、矩形メッシュ、弾性材料、左端固定、下端鉛直拘束、上端荷重を持つ 2D 平面ひずみモデルです。

## 2. 入力を診断する

```powershell
python -m geofem_app.cli diagnose runs\tutorial\input.yaml --out runs\tutorial\diagnostics
```

`input_diagnostics.html` を開き、エラーがないことを確認します。警告がある場合は `path` と `suggestion` に従って YAML を修正します。

## 3. 解析を実行する

```powershell
python -m geofem_app.cli solve runs\tutorial\input.yaml --out runs\tutorial\solve
```

主要出力:

- `summary.json`: 実行結果の入口
- `Stage-1\displacements.csv`: 節点変位
- `Stage-1\element_stress.csv`: 要素応力
- `result_view_index.html`: 結果確認用インデックス
- `standard_report.html`: 標準帳票

## 4. メッシュ品質を確認する

```powershell
python -m geofem_app.cli mesh-quality runs\tutorial\input.yaml --out runs\tutorial\mesh_quality
```

`mesh_quality.html` と `mesh_quality_repairs.csv` で、低品質要素と修復候補を確認します。

## 5. 材料を確認する

```powershell
python -m geofem_app.cli materials runs\tutorial\input.yaml --out runs\tutorial\materials
```

`material_inventory.csv` は入力モデルで使った材料、`material_model_catalog.html` は利用できる材料モデルの一覧です。

## 6. CAD 取込を行う場合

DXF/SXF/GF1 取込は `geofem_app.cad_import` 系の公開アダプタを使います。取込後は必ず次を確認します。

- 取込レイヤと線分が意図した境界に対応している。
- 閉合、重複、微小ギャップの診断がエラーになっていない。
- メッシュ生成後に `mesh-quality` を実行し、要素品質を確認する。

## 7. GUI で同じ流れを行う

```powershell
run_gui.bat
```

GUI では、メッシュ、材料、境界/荷重、解析、結果のタブを順に進めます。結果タブでは `結果インデックス`、`メッシュ品質`、`解析ログ`、`性能`、`標準帳票` を確認します。
