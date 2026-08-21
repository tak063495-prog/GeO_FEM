# Changelog

## v0.1.0 - 2026-08-21

初回公開基準版。以下を同一コミットに固定しました。

- 2D FEM solver、GUI、入出力、性能計測、回帰テスト
- TRI3/TRI6/QUAD4/QUAD8、FULL/SRI/B-bar、塑性・大変形・圧密/u-p・軸対称・動的・Riks・MPC/Lagrange・SRM
- SRMのadaptive bracket、lookahead、worker自動設定、失敗trial診断、キャンセル、安全な先読み制御
- Mohr-CoulombのNumba、regularization、active-set、fallback診断
- GUIの結果サマリ、遅延帳票、プレビューcache、LOD、読み取り専用結果参照、ナビゲーション整理
- Case1〜4の入力例、テスト、性能・失敗分類資料
- 開発履歴と失敗分類Zip: `GeoFEM_development_history_failures_20260821.zip`

### Known limitations

- 2D plane-strainを中心とした実装で、3D/v26完全互換は対象外
- SRMで構成則fallbackまたは正則化が発生した結果は、`limited` confidenceや検証要求が付く場合がある
- 論文値との一致は保証せず、メッシュ、境界条件、荷重経路、failure定義、構成則を分けて検証する必要がある
- Case1〜4の実行時間はCPU、Numba warmup、I/O、worker設定で変わる
