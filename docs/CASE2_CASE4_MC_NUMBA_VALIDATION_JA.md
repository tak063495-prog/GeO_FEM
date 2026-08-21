# Case2 / Case4 Mohr-Coulomb Numba高速化検証

`run_case2-4_mc_numba_validation.bat`を起動すると、次の順に連続実行します。

1. Case2: `factor_tol=0.005`の厳格確認
2. Case4: 現行のspeed-guarded Auto SRM

入力条件は複製せず、次の既存Canonical YAMLを使用します。

- Case2: `dist/sustainability_2024_case2_strict_fos_005_20260725/sustainability_2024_case2_quad4_sri_strict_fos_005.yaml`
- Case4: `dist/sustainability_2024_case1-4_auto_srm_speed_guarded_20260612/sustainability_2024_case4_quad4_sri_auto_srm_speed_guarded.yaml`

結果は`runs/case2-4_mc_numba_validation_日時`へ保存されます。ルートの
`last_case2-4_mc_numba_validation.txt`には最新結果フォルダーが記録されます。

検証サマリーにはFOS、stable/failed factor、経過時間に加えて、次の件数を保存します。

- `mc_numba_to_python_fallback_count`
- `mc_numba_regularized_projection_count`
- `mc_regularized_projection_count`

高速化が有効なら、正則化が発生していても
`mc_numba_to_python_fallback_count`が大幅に減り、
`mc_numba_regularized_projection_count`へ処理件数が移ります。

中断は`stop_case2-4_mc_numba_validation.bat`を使用します。
