# GeoFEM UTF-8・文字化け防止方針

## 目的

GeoFEM は日本語のGUI文言、診断、帳票、CSV、Markdown、JSONを扱うため、商用級FEMとして文字化けを起こさない入出力経路を標準化する。特に Windows PowerShell、外部CAD/CSV、帳票HTML/PDF、解析ログの往復で、利用者が内容を確認できる状態を維持する。

## 基本方針

- ソースコード、Markdown、YAML、JSON、CSV、HTML、ログは UTF-8 を標準文字コードとする。
- Python のテキスト入出力は `encoding="utf-8"` を明示する。
- CSV 書込みでは `newline=""` を併用し、Windows 上の余分な空行を防ぐ。
- JSON は `ensure_ascii=False` を基本とし、日本語を読みやすいまま保存する。
- HTML 帳票は `<meta charset="utf-8">` を含める。
- CLI 起動時は標準出力と標準エラーを UTF-8/replace に再設定し、PowerShell 表示での例外的な文字化けを抑制する。

## 監査コマンド

以下のコマンドで、対象ディレクトリ配下の主要テキストファイルについて UTF-8 decode、BOM、代表的な文字化け断片を監査する。

```powershell
python -m geofem_app.cli encoding-audit --root . --out runs/encoding_audit
```

警告も失敗扱いにしたい場合は次を使う。

```powershell
python -m geofem_app.cli encoding-audit --root . --out runs/encoding_audit --fail-on-warning
```

出力物は `encoding_audit.json`、`encoding_audit.csv`、`encoding_audit.html` である。商用配布前や大きな帳票・GUI文言変更後に実行する。

## 失敗時の修正指針

- `utf8_decode` が失敗した場合は、ファイルを UTF-8 で保存し直す。
- `mojibake_marker` が出た場合は、該当文字列が意図した日本語ではなく文字化け断片でないか確認する。
- `utf8_bom` は実行継続可能な警告だが、基本は BOM なし UTF-8 へ統一する。
- 外部ツールから受け取るファイルは、読込時に明示的なエンコーディング選択または `errors="replace"` を使い、診断ログに変換結果を残す。

## 回帰の考え方

文字化けは見た目の問題に見えて、診断、帳票、監査証跡、ユーザーサポートの信頼性を直接下げる。したがって、文字コード方針は単なる文書ではなく、`encoding-audit` と単体テストで継続的に検出する。
