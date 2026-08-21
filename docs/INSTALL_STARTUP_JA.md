# GeoFEM 配布と起動手順

更新日: 2026-05-30

この手順は、新しい Windows 環境で GeoFEM を起動し、最小サンプル解析を完了できることを確認するためのものです。

## 1. 前提

- Python 3.12 以降を使用します。再現性のため `requirements-lock.txt` の固定版を使用します。
- 作業フォルダは `C:\Users\link_\Downloads\WORK\GeoFEM` を想定します。
- 必須ライブラリは `requirements.txt` にまとめています。
- GUI を使う場合は `PySide6`、実行ファイル化を検証する場合は `PyInstaller` も必要です。

## 2. 初回セットアップ

```powershell
cd C:\Users\link_\Downloads\WORK\GeoFEM
setup_dependencies.bat
```

`run_gui.bat` は初回起動時に `.venv` がない場合、自動で `setup_dependencies.bat` を呼び出します。
配布Zipにはライブラリ本体は含めず、`requirements.txt` からローカル `.venv` に導入します。
インターネットなしで配布する場合は、同階層に `wheelhouse/` を置くとそこからインストールします。

## 3. 起動確認

依存ライブラリ、配布補助ファイル、サンプル解析、主要出力をまとめて確認します。

```powershell
python -m geofem_app.cli doctor --out runs\startup_check
```

確認結果は次に出力されます。

- `runs\startup_check\startup_check.json`
- `runs\startup_check\startup_check.csv`
- `runs\startup_check\startup_check.html`
- `runs\startup_check\sample_run\summary.json`

`passed=true` であれば、CLI の初回起動とサンプル解析は完了です。

## 4. GUI 起動

```powershell
run_gui.bat
```

GUI 依存関係も必須扱いで確認する場合は次を実行します。

```powershell
python -m geofem_app.cli doctor --include-gui --out runs\startup_check_gui
```

## 5. GUI 実行ファイル化の検証

`PyInstaller` を導入した環境で次を実行します。

```powershell
build_gui_exe.bat
```

検証項目:

- `dist\GeoFEM-GUI.exe` が生成される。
- EXE を起動して GUI が表示される。
- サンプル入力を開き、解析実行、結果表、標準帳票を確認できる。
- `runs` 配下に `summary.json`、`result_view_index.json`、`standard_report.html` が生成される。

## 6. 回帰

配布前には次を実行します。

```powershell
python -W error::ResourceWarning -m unittest discover -s tests -p "test_*.py" -v
python -m geofem_app.cli benchmarks --out runs\geofem_standard_benchmarks --update-baseline
```
