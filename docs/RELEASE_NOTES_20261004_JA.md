# 2026-10-04 レビュー修正版（未公開・ドラフトPR）

基準: master `2b6a96eded5a593293aa9d9ec230a7289f51fa77`。実行PC: Triela / Windows x64。
main/master、タグ、GitHub Releaseは変更しません。既存checkoutを使用せず独立cloneで作業しました。

## 修正と根拠

- 遅延CSV再読込のnode_id/element_idを文字列として保存。`1`/`001`/`n1`で非ゼロ変位・水圧・反力を往復検証。
  未知・重複・欠落節点の変位/水圧表はFEM2DErrorで診断。構造回転等の追加DOF再構築は今回対象外です。
- TRI6生成境界セットに中間節点を追加。全4辺の幾何学的節点一致とux/uy/p拘束を検証。
- 明示した無効水位対象の全域fallbackを撤去。対象省略だけ全域を既定とし、edge/edgesも選択辺の節点に限定。
- cached LU分解を例外変換のtry内に移動。mockと32DOF実特異系でFEM2DErrorを確認、待機解除機構を維持。
- Mohr-Coulomb共有硬化κnew=κ+Σγに対して、整合行列をN C Mᵀ+H11ᵀへ修正。
  Python/cache/Numbaと主応力接線を同じ導関数に整合。独立2面KKTと公開応力の差分で確認。
  H=100の主試行応力[-100,0,0]で修正応力[-88.92220613,-16.61669081,-16.61669081]、κ=0.0221555877453。
- rankine_capの接線は公開のpre-cap→MC→post-cap更新を直接差分。専用高速接線の異なる処理順を回避。
  legacy_bounded等へこの制限は広げません。rankine_capは正しさ優先のため接線計算が遅くなる場合があります。

## 検証

Python 3.12.14、NumPy 2.4.6、SciPy 1.17.1、Numba 0.65.1、PySide6 6.11.1（既存lockと一致）。
修正前: 初期7テストで6候補が失敗/例外として再現。修正後: 新規11テストPASS。
最終の全回帰は639件実行、637件PASS、2件NOT_RUN（下記の外部dist不足）。
`python -W error::ResourceWarning -m unittest discover -s tests -p "test_*.py" -v`、233.705秒。
PyInstaller 6.20.0でCLI/GUIのbuildを検証。Numba cache locatorがPYZ内の相対ファイル名を
解決できず起動不能となる問題を再現し、geofem_appを実在するPythonソースとして同梱するhookで修正。
起動診断に必要な配布補助3ファイルも同梱します。
ビルド時PATH先頭にSystem32を置き、別ツールの同名ICU DLL混入を防ぎます。
最終CLI/GUIとも `doctor --include-gui` PASS（エラー0）、サンプル解析PASS。
QtWidgetsと日本語フォントの診断もOK。警告1件は実行時に不要なPyInstaller未同梱だけです。
手動GUI操作・署名付き配布公開はNOT_RUN。配布exeはGitHub Releaseとして公開していません。
GitHub上のチェック状態はドラフトPRで確認できます。

NOT_RUN: ソースに含まれない外部dist/配布パッケージを要する既存2試験。
テストはパッケージ不存在時だけ明示skipし、期待値やパッケージが存在する場合の検証は維持します。
今回のrepoには既存GitHub Actions workflowがありません。外部CIの設定は確認できていません。

## 未採用・対象外

元の線形方程式の後退誤差による新判定、全SRMケースの再認証、追加DOF報告の完全復元は対象外。
Čermák/Sysala/Valdmanの固定B・接線変化点更新は既存cache/batchと重なるため、大規模な再設計は見送り。
[論文](https://arxiv.org/abs/1805.04155)。inexact Newton、fastmath、許容差緩和は採用しません。

## 性能調査（既定方式の変更なし）

QUAD4弾性の12×12/40×40/80×80格子、底面ux/uy拘束、1 thread、
5回のfactor/ordering cacheをclearしたcold計測と同じ数値LUのwarm-hitを比較。
coldの計測はPython cProfile下です。Numbaのassembly初回コンパイルはsolver時間から除外し、
profileへ分けて記録。メモリはL/U/置換配列の推定量でありピークRSSではありません。
全結果は同じ参照解の∞誤差<1e-7、元方程式相対残差<1e-10、warm-hitは配列厳密一致、全case成功。
基準は不具合修正後であり、旧バグ挙動を性能基準にはしません。

|格子|順序|cold中央値s|LU-hit中央値s|L+U nnz|LU推定bytes|
|---:|---|---:|---:|---:|---:|
|12|rcm_natural|0.002143|0.000408|18094|222128|
|12|colamd|0.001049|0.000262|15882|195584|
|12|mmd_at_plus_a|0.001011|0.000256|12930|160160|
|40|rcm_natural|0.031597|0.003833|674922|8151552|
|40|colamd|0.014760|0.002446|394706|4788960|
|40|mmd_at_plus_a|0.009474|0.002278|280287|3415932|
|80|rcm_natural|0.309551|0.021603|5426682|65327552|
|80|colamd|0.120088|0.013571|3123233|37686164|
|80|mmd_at_plus_a|0.055630|0.011215|1688354|20467616|

80×80でMMD_AT_PLUS_AはRCM+NATURALよりcold約5.56倍、LU推定領域約31%でした。
ただし弾性格子しか測定していないため、非対称塑性接線・u-p・SRMの収束/bracketへの影響を保証せず、
既定のorderingは変更しません。既存の明示設定 `cache_symbolic: false`、`permc_spec: MMD_AT_PLUS_A`
でこの経路を選択できます。数値LUを構造だけで再利用する変更や追加のglobal cacheは採用しません。
profileでは数値分解が主な時間を占めました。大規模な固定B再構成やinexact Newtonは今回見送りします。
再現: `python tools/benchmark_review_orderings.py --output runs/review-orderings.json`。
生データとprofile: [REVIEW_ORDERING_20261004.json](REVIEW_ORDERING_20261004.json)。
