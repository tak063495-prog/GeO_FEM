# GeoFEM プロジェクトフォルダー構成

このフォルダーは、アプリ本体、テスト、サンプル、実行結果、計画文書を分けて管理します。

## ルート直下

- `geofem_app/`: アプリケーション本体。
- `tests/`: 回帰テスト。
- `examples/`: 解析サンプルYAML。
- `templates/`: テンプレートと組織プロファイル。
- `tools/`: ベンチ、監査、補助スクリプト。
- `docs/`: 利用者向け・開発者向け文書。
- `runs/`: 解析・ベンチ実行結果。
- `input/`, `mesh/`, `results/`, `reports/`, `logs/`: GUI/手動運用の作業用フォルダー。
- `autosave/`: GUI自動保存。古い履歴は `autosave/archive/` に退避する。
- `run_gui.bat`, `build_gui_exe.bat`, `geofem_gui.py`: 起動・ビルド用の入口。
- `requirements.txt`: Python依存関係。
- `README_GeoFEM_MVP.md`: プロジェクト概要。
- `GeoFEM_completion_backlog.md`, `unresolved_items.md`: 既存テスト互換を含むバックログ入口。

## docs 配下

- `docs/requirements/`: 要件定義や仕様メモ。
- `docs/*_JA.md`: 利用・起動・開発・エンコーディング等の文書。

## 整理ルール

1. ソース、テスト、起動入口はルートまたは既存パッケージ構造から動かさない。
2. 新しい未達・改善バックログは `unresolved_items.md` に集約する。
3. 要件定義メモや詳細な仕様メモは `docs/requirements/` に置く。
4. 実行結果は `runs/` または入力YAML近傍のrunフォルダーに置く。
5. Python/pytestキャッシュは再生成可能なので保持しない。
6. 古いautosaveは削除せず `autosave/archive/` に退避する。
