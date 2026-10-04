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
修正前: 初期7テストで6候補が失敗/例外として再現。修正後: 新規10テストPASS。
全回帰、配布build、性能計測、remote/CIの最終結果は後続コミットで追記します。

NOT_RUN: ソースに含まれない外部dist/配布パッケージを要する既存2試験。
テストはパッケージ不存在時だけ明示skipし、期待値やパッケージが存在する場合の検証は維持します。
今回のrepoには既存GitHub Actions workflowがありません。外部CIの設定は確認できていません。

## 未採用・対象外

元の線形方程式の後退誤差による新判定、全SRMケースの再認証、追加DOF報告の完全復元は対象外。
Čermák/Sysala/Valdmanの固定B・接線変化点更新は既存cache/batchと重なるため、大規模な再設計は見送り。
[論文](https://arxiv.org/abs/1805.04155)。inexact Newton、fastmath、許容差緩和は採用しません。
