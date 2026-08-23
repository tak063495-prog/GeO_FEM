# GeoFEM 2D

[English](README.md) | [日本語](README.ja.md)

GeoFEMは、PySide6 GUIとコマンドライン実行に対応した2次元有限要素解析ソフトウェアです。平面ひずみの地盤解析を中心に、弾塑性材料、強度低減法（SRM）、圧密・u-p連成、大変形、軸対称、動的解析を扱います。

## 主な機能

- TRI3、TRI6、QUAD4、QUAD8要素
- FULL、SRI、B-bar積分
- Drucker-Prager、Mohr-Coulombなどの弾塑性経路
- adaptive bracket、厳格なfactor許容差、診断ログ、先読み、キャンセル、worker制御を備えたSRM/FOS探索
- 圧密、u-p、軸対称、動的、Riks、MPC/Lagrange、interface解析
- CSV、VTK、JSONおよび遅延生成HTML/PDF結果
- 大規模メッシュ向けGUIプレビューキャッシュ、結果サマリ、読み取り専用結果参照、表示品質制御

## 重要な適用範囲と精度上の注意

本プロジェクトは商用GeoFEAS/VGFlowとの機能同等性や、論文掲載値との完全一致を保証しません。SRMでは、正則化または構成則fallbackが使われた場合に `material_fallback_evidence` として信頼度が限定されることがあります。`converged=true` だけでFOSの妥当性が証明されるわけではありません。

実務・研究で確認する場合は、FOSだけでなく、安定/失敗bracket、factor許容差、fallback回数、収束診断、メッシュ品質、使用YAML、commit、実行環境と `analysis_log` を保存してください。

## 起動

CLI:

```powershell
python -m geofem_app.cli solve examples\plane_strain_quad4_bbar.yaml --out runs\sample_quad4
```

GUI:

```powershell
run_gui.bat
```

SRMのCase1～4は `examples/` にあります。長時間解析中に保証される操作は協調的な中断です。帳票とresult viewは必要時に遅延生成されます。

## 検証

```powershell
python -W error::ResourceWarning -m unittest discover -s tests -p "test_*.py" -v
```

Case1～4の完全実行時間と結果は、ハードウェア、NumPy/SciPy/Numba、worker設定に依存します。FOS比較では、同じYAML、commit、worker方針、実行環境を記録してください。

## ドキュメント

- [開発者ガイド](docs/DEVELOPER_GUIDE_JA.md)
- [起動・インストール](docs/INSTALL_STARTUP_JA.md)
- [API契約](docs/API_CONTRACTS_JA.md)
- [性能レビュー](docs/PERFORMANCE_REVIEW_JA.md)
- [Case2/4 Mohr-Coulomb Numba検証](docs/CASE2_CASE4_MC_NUMBA_VALIDATION_JA.md)
- [開発履歴・失敗分類](development_history_failures_20260821/DEVELOPMENT_FAILURE_HISTORY_JA.md)
- [成果物マニフェスト](docs/DELIVERABLE_MANIFEST_20260821.md)
- [変更履歴](CHANGELOG.md)

## ライセンス

ソースコードは [MIT License](LICENSE) で提供します。依存ライブラリ、GUIライブラリ、Numba/LLVM関連、外部DWG変換ツールにはそれぞれのライセンスと通知が適用されます。ネイティブDWGデコーダーを同梱するものではありません。

## 開発への参加

変更を提案する場合は、[CONTRIBUTING.md](CONTRIBUTING.md)を確認してください。脆弱性は公開Issueに秘密情報を貼らず、[SECURITY.md](SECURITY.md)の手順に従って報告してください。
