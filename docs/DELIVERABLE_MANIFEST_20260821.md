# GeoFEM 現行成果物マニフェスト

作成日: 2026-08-21

このマニフェストは初回Gitコミットに含める現行成果物と、別途配布する失敗分類Zipの内容を固定するための記録です。

## Gitコミットに含めるもの

- `geofem_app/`: 2D FEMソルバー、SRM、GUI、入出力、性能計測
- `tests/`: FEM、SRM、GUI、入力診断、帳票、性能、配布関連の回帰テスト
- `docs/`: API契約、開発者ガイド、性能レビュー、GUI/起動/利用ガイド
- `examples/`: plane strain、TRI6、SRM Case1〜4を含む例題YAML
- `templates/`: GUI/解析テンプレート
- ルートのCLI/GUI起動スクリプト、Case実行BAT、依存関係定義、README
- `input/` にある現在の入力YAML
- `development_history_failures_20260821/`: 失敗分類報告書、台帳、時系列、証拠資料
- `GeoFEM_development_history_failures_20260821.zip`: 上記失敗分類アーカイブ。`.gitignore`の `*.zip` を尊重し、明示的に追加する

## Zipの内容

`GeoFEM_development_history_failures_20260821.zip` には以下を収録しています。

- `DEVELOPMENT_FAILURE_HISTORY_JA.md`: 開発時系列、失敗12項目、原因、影響、対策、残存リスク、今後の知見
- `failure_inventory.json`: 失敗分類の機械可読台帳
- `failure_timeline.csv`: 時系列データ
- `evidence_manifest.csv`: 証拠資料の出典と用途
- Case1〜4のPhase6/部分実行 `summary.json`
- Case2/4 Mohr-Coulomb Numba検証のsummary/log
- Case1 failure score/early stop、Case2 fast-failのanalysis log
- `SRM_PARALLEL_LOOKAHEAD_TODO_JA.md`、性能レビュー、開発者ガイド、未達バックログ
- GUI watchdog、依存関係セットアップ、起動、監査ログ

## 除外する生成物

再現性を持たない、またはサイズが大きい生成物はGit初回コミットから除外します。

- `.venv/`、`.build_exe_venv/`
- `build/`、`dist/`
- `runs/`、`outputs/`、`autosave/`
- Pythonキャッシュ、IDE設定、展開済み一時ファイル

必要な実行結果は、run directoryを個別にZip化し、使用したYAML、環境、Git commit ID、summary、analysis_logを同梱する運用にします。

## 受入記録

- 失敗分類: 12項目、6系統
- Zip内証拠資料: 23件
- Zip内エントリ: 29件
- Zip SHA-256: `2EB0B0E7D9EAD80F9862B6B901F71B14AB8AFC272693D46644CF09805B4F8B36`
- Gitリモート: 未設定。現時点では外部pushは実施していない
