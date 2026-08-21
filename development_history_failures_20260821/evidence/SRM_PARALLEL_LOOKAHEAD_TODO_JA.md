## 2026-07-22 進捗

### 今回の実装済み
- [x] SRM lookaheadの実行方式へ `process` を追加し、重いtrialがPython GILで直列化されないようにした。
- [x] process worker内でmesh topologyとfactor step cacheを初期化し、trial実行に再利用するようにした。
- [x] process poolを利用できない条件では逐次実行へ安全にfallbackし、要求方式・実効方式・fallback理由をログへ残すようにした。
- [x] CLI/GUIエントリポイントへ `multiprocessing.freeze_support()` を追加し、PyInstaller版のspawn起動へ対応した。
- [x] `boundary_verification_strategy: deferred_final` を追加し、中間failed trialのcold再解析を最終failed端点まで遅延できるようにした。
- [x] deferred verificationでstableへ反転した場合に外側bracketへ戻す復旧処理と、復旧回数上限を追加した。
- [x] failure scoreが弱い暫定failedは即時cold検証し、弱い数値失敗をFOS上限へ採用しないようにした。
- [x] Case1-4 speed-guarded YAMLへprocess lookaheadとdeferred final verificationを反映した。
- [x] Mohr-Coulomb Python fallbackの材料・factor不変候補行列、特異判定、非特異LU分解をキャッシュした。
- [x] 非関連流れ・硬化なし・十分な摩擦角のfallbackで、候補順位付け後の上位候補だけを従来LU/SVDで厳密再評価する経路を追加した。
- [x] 低摩擦角または分類境界が疑わしい条件は従来の全候補走査へ戻し、FOS精度を優先するガードを追加した。
- [x] 摩擦角5/15/20/25/35/40度の判定点で、成功・失敗、応力、yield値、active-set、塑性乗数が参照経路と一致することを確認した。
- [x] SRM/Mohr-Coulomb部分回帰（63件）と全回帰（587件、186 subtests）を通過した。

### 次回まとめて行う実ケース検証
- [ ] Case1-4を同一マシン・単独実行条件で再解析し、従来版とFOS bracket、trial採用順、elapsedを比較する。
- [ ] deferred final verificationによるcold再解析削減数とstable反転復旧の有無をCase別に確認する。
- [ ] process lookaheadのworker数、CPU使用率、メモリピーク、投機未使用trial数をCase別に確認する。
- [ ] Case2/4でMohr-Coulomb fallback時間とfallback回数を比較し、摩擦角ガード後の実効短縮率を確認する。
- [ ] Case1/3の低摩擦角経路は全候補走査を維持しているため、FOS一致を確認したうえで別方式の高速化要否を判断する。

## 2026-06-13 進捗

### 実装済み
- [x] SRM解析開始時に論理CPU数、可能なら物理CPU数、空きメモリを検出する。
- [x] SRM trial検索へメッシュ規模（節点数、要素数、有効要素数、DOF数）を渡し、trial 1本あたりの概算メモリを見積もる。
- [x] `srm.parallel.policy: auto` / `max_workers: auto` で、実行context、CPU、trial数、メモリ上限からworker数を選ぶ。
- [x] worker数に応じた `threads_per_worker` と `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `NUMBA_NUM_THREADS` の推奨値を算出する。
- [x] 選択されたworker数、thread数、CPU/メモリ/mesh情報を `summary.json` に残す。
- [x] 同じ情報を `analysis_log.csv` / `analysis_log.json` のstage_completedおよびsrm_summaryイベントに残す。
- [x] 同じ情報を `run.log` の `[srm-parallel]` 行に残す。
- [x] 並列SRM trial実行中に `threadpoolctl` が利用可能な場合は `threads_per_worker` を実際に適用する。
- [x] `threadpoolctl` が無い環境では、prefetch window中だけ `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `NUMBA_NUM_THREADS` を一時設定し、終了後に復元する。
- [x] thread制御の適用可否、適用方式、未適用理由を `analysis_log` と `run.log` に残す。
- [x] thread環境変数fallbackの復元状況と `threadpoolctl` 利用可否を `analysis_log` と `run.log` に残す。
- [x] `srm.parallel.strategy: lookahead` をSRM並列設定として受け取り、summary/ログに残す。
- [x] adaptive bracketの粗探索で、`strategy: lookahead` かつ `preserve_decision_order: true` の場合にwindow先読み並列を実行する。
- [x] Auto下側探索projection候補を、採用順序を維持したままlookaheadで先読み並列化する。
- [x] adaptive bisectionで、workerが3以上ある場合に現在midと左右の次候補を投機prefetchする。
- [x] 先読みtrialは従来のfactor順でのみ採用し、未使用trialをFOS bracket判定へ混ぜない。
- [x] speculative trial数、採用数、未使用数、window評価数、未使用factorをsummaryに残す。
- [x] lower_projection内にprojection先読みの有効/本数/採用数/未使用数を残す。
- [x] bisection投機trialの有効/本数/採用数/未使用数/未使用factorをsummaryに残す。
- [x] speculative prefetchの呼び出し数、wall時間、trial合計時間、queue待ち、worker実行時間、推定短縮時間をsummaryに残す。
- [x] SRM lookahead/prefetchにcancel token/callbackを渡せるようにし、キャンセル要求済みなら未開始の投機trialをskip/cancelしてFOS判定へ混ぜない。
- [x] speculative cancellation requested、canceled/skipped trial数、キャンセル理由メモをsummary/analysis_log/run.logへ残す。
- [x] CLI `solve --cancel-file` とGUI停止ボタンからのcancel-file作成を追加し、サブプロセス実行でもSRM cancel tokenへ要求を伝えられるようにする。
- [x] GUI停止時はcancel-file作成後に短い猶予を置き、まだ解析プロセスが残る場合だけkillする。
- [x] 実行中SRM trialでも、微小変形increment、大変形adaptive step、通常/軸対称Newton iterationの安全な区切りでcancel-fileを検出して停止できるようにする。
- [x] 実行中trialキャンセルは `trial_status=solver_cancelled`、`failure_reason=cancelled`、`solver_cancel_checkpoint` としてSRM trialログ/analysis_logへ残す。
- [x] speculative trial数、採用数、未使用数、window評価数を `analysis_log` と `run.log` に残す。
- [x] bisection投機trialの有効/本数/採用数/未使用数を `analysis_log` と `run.log` に残す。
- [x] speculative prefetchのwall時間、trial合計時間、queue待ち、推定短縮時間を `analysis_log` と `run.log` に残す。
- [x] examples配下のCase1-4 `auto_srm` YAMLに `srm.parallel.strategy: lookahead` / `max_workers: auto` / `preserve_decision_order: true` を反映する。
- [x] examples配下のCase1-4 `accurate_fos_with_trial_logs` YAMLに同じlookahead並列設定を反映する。
- [x] dist配下のCase1-4 `fast_current_fos` YAMLに同じlookahead並列設定を反映する。
- [x] dist配下のCase1-4 `speed_guarded` YAMLに同じlookahead並列設定を反映する。
- [x] Case1-4のexamples/dist合計16 YAMLがYAMLとして正常にparseできることを確認する。
- [x] SRM並列設定とanalysis_log出力の単体回帰を追加する。
- [x] 失敗trialログの `last_accepted_load_factor` / `final_step_size` / `cutback_count` / `residual_reduction_ratio` / 塑性クラスタ情報を次trialの増分幅調整へ反映する。
- [x] `srm.adaptive_increment_control` または `srm.auto` 配下の設定で、増分幅調整の有効化・閾値・最大steps倍率を指定できるようにする。
- [x] 並列lookahead/prefetchでもfactor別solver overrideを渡せるようにし、採用順序を変えずに失敗ログ由来の増分調整を適用できるようにする。
- [x] 増分幅調整の根拠factor、理由、cutback比、目標初期step係数などを `analysis_log.csv` / `analysis_log.json` のSRM trialイベントへ残す。
- [x] 失敗ログ増分幅調整の単体回帰を追加する。
- [x] `srm.warm_start.enabled` / `displacement_only` / `max_factor_distance` を受け取り、直近stable trialの変位だけを次factorの初期値として渡せるようにする。
- [x] warm startはplane strain SRMで有効化し、大変形など未対応solverでは明示的に無効化してsummaryへ理由を残す。
- [x] 軸対称SRMにも変位のみwarm startを拡張し、Newton初期推定として安全に渡せるようにする。
- [x] warm startのsource factor、target factor、factor距離、変位サイズ、最大変位ノルムをSRM trialログへ残す。
- [x] warm startの単体回帰を追加する。
- [x] 軸対称SRM warm startの単体回帰を追加する。
- [x] `tests.test_fem2d` 全体回帰（274件）を実行し通過を確認する。

### 残タスク
- [ ] `srm.parallel.strategy: lookahead` 反映後のCase1-4実ケースで速度とFOS bracket維持を検証する。
- [ ] 粗探索の先読み並列化をAuto retry/suspect failureが多いケースで検証し、必要なら制限条件を追加する。
- [x] 下側探索projectionの並列化をCase1の実ケースで検証し、過剰な未使用trialが出る場合はprobe数/深さを調整する。
- [ ] bisectionの投機実行をCase1-4の実ケースで検証し、worker数3未満では無効の既定が妥当か確認する。
- [x] GUI中断からSRM cancel token/callbackへ要求を伝え、worker pool上の未開始投機trialを止める。
- [x] 実行中trialを増分/iterationの安全な区切りで停止する仕組みを検討・実装する。
- [ ] 変位のみwarm startをCase1-4実ケースで検証し、FOS bracketが変わらないこととNewton反復/elapsed短縮を確認する。
- [ ] 大変形SRMへの変位warm start拡張可否を、更新形状 `total_u` とNewton初期推定を混同しない形で別途検討する。
- [ ] 失敗ログに基づく増分幅調整をCase1-4実ケースで検証し、過度な保守化やFOS bracket変化が起きないことを確認する。
- [ ] stable/failed trialのさらなる軽量化を検証する。
- [ ] 線形ソルバ前処理cache強化を検証する。
- [ ] 粗探索ジャンプ候補は先読み評価に限定し、採用順序を変えない形で検証する。
- [ ] Case1-4でFOS bracket維持とelapsed短縮を比較する。

# SRM先読み並列化・自動worker設定 ToDo

## 目的

SRMのFOS精度を維持しつつ、環境に応じてworker数と数値ライブラリthread数を自動調整し、粗探索・下側projection・bisectionを先読み/投機実行して壁時計時間を短縮する。

## 基本方針

- 探索の採用順序は従来どおり維持する。
- 先に完了したtrialはキャッシュし、従来探索順で必要になった時だけ採用する。
- FOS判定、stable/failed bracket更新、suspect/confirmed判定は現行ロジックを維持する。
- `preserve_decision_order: true` を既定とする。
- GUI実行中は中断以外の操作を想定しないため、従来よりworkerを積極利用する。
- ただしGUIの中断応答性を維持するため、最低1 worker相当の余力を残す。
- worker数を増やす場合、BLAS/OpenMP/Numbaなどworker内thread数も同時に抑制し、過剰並列を避ける。

## 1. 環境検出

- [ ] 解析開始時に論理CPU数を取得する。
- [ ] 物理CPU数を可能な範囲で取得する。
- [ ] 空きメモリ量を取得する。
- [ ] 実行コンテキストを `gui` / `cli` / `batch` として判定する。
- [ ] メッシュ要素数、節点数、DOF数を取得する。
- [ ] SRM trial 1本あたりの概算メモリ量を推定する。
- [ ] 環境検出結果を `summary.json` に出力する。
- [ ] 環境検出結果を `analysis_log.csv` / `analysis_log.json` に出力する。
- [ ] 環境検出結果を `run.log` に出力する。

## 2. 自動worker数ポリシー

- [ ] `srm.parallel.policy: auto` を追加する。
- [ ] `srm.parallel.strategy: lookahead` を追加する。
- [ ] `srm.parallel.preserve_decision_order: true` を追加し、既定値にする。
- [ ] GUIでは `physical_cores - gui_reserve_workers` を上限候補にする。
- [ ] GUIの既定 `gui_reserve_workers` は `1` とする。
- [ ] CLI/バッチではGUIより積極的なworker数を選択する。
- [ ] 空きメモリと `memory_fraction` からworker上限を制限する。
- [ ] 小規模モデルでは並列化固定費が勝たないよう、worker数を自動で1に落とす閾値を設ける。
- [ ] ユーザー指定の `max_workers` がある場合は上限として尊重する。
- [ ] `max_workers: auto` を許容する。
- [ ] 選択されたworker数と選択理由をログに残す。

設定案:

```yaml
solver:
  execution:
    thread_policy: auto
    srm_parallel_policy: auto

srm:
  parallel:
    enabled: true
    policy: auto
    strategy: lookahead
    preserve_decision_order: true
    gui_reserve_workers: 1
    memory_fraction: 0.70
    max_workers: auto
```

## 3. 数値ライブラリthread数制御

- [ ] worker数とworker内thread数をセットで決定する。
- [ ] SRM trial並列時はworker内thread数を原則1から2へ抑制する。
- [ ] `OMP_NUM_THREADS` 相当の制御方法を整理する。
- [ ] `MKL_NUM_THREADS` 相当の制御方法を整理する。
- [ ] `OPENBLAS_NUM_THREADS` 相当の制御方法を整理する。
- [ ] `NUMBA_NUM_THREADS` 相当の制御方法を整理する。
- [ ] 実行中に安全に変更できない環境変数は、起動時/worker起動時に設定する。
- [ ] thread制御が効かなかった場合もログに明示する。
- [ ] 過剰並列の検出指標をログに残す。

## 4. SRM trial result cache

- [ ] factorごとのtrial結果キャッシュを明示的な構造に整理する。
- [ ] 実行中、完了済み、採用済み、未使用、キャンセル済みを区別する。
- [ ] 同一factorを二重起動しないようにする。
- [ ] retry/solver_override付きtrialは通常trialとキャッシュキーを分離する。
- [ ] 先読みtrialの結果にも既存の詳細診断ログを保持する。
- [ ] 未使用の投機trialは `speculative_unused` として記録する。

## 5. 粗探索の先読み並列化

- [ ] 上側粗探索で次の候補factorをlookahead起動する。
- [ ] 下側粗探索で次の候補factorをlookahead起動する。
- [ ] 採用順序は現在の逐次探索順に固定する。
- [ ] 失敗bracketが見つかった時点で未開始trialをキャンセルする。
- [ ] 実行中trialは安全停止できる場合だけキャンセルし、そうでなければ完了後に未使用扱いにする。
- [ ] `lookahead_depth` を自動設定する。
- [ ] `lookahead_depth` のユーザー上限を設定可能にする。
- [ ] worker数1では現行挙動と完全一致させる。

## 6. 下側探索projectionの並列化

- [ ] `last_accepted_load_factor` 由来のprojection候補を並列実行できるようにする。
- [ ] projection候補の採用順序は既存の `lower_projection_multipliers` 順に固定する。
- [ ] Case1のような探索経路に敏感なケースでは、`lower_projection_skip_coarse_scan_on_bracket` の既存設定を尊重する。
- [ ] projectionでbracketが得られた場合も、既存設定がfalseなら粗探索を継続する。
- [ ] projectionの未使用trialをログに残す。
- [ ] projection並列化によってFOS bracketが変わらないことをテストする。

## 7. bisectionの投機実行

- [ ] midpoint trialを従来どおり必須trialとして起動する。
- [ ] midpointの次に必要になりそうな左右候補を投機実行する。
- [ ] midpoint結果に応じて必要側だけ採用する。
- [ ] 不要側は `speculative_unused` としてログに残す。
- [ ] bracket更新は従来順序を厳守する。
- [ ] `factor_tol` 判定は現行ロジックを維持する。
- [ ] bisection投機実行を無効化できる設定を用意する。

## 8. キャンセル・GUI応答性

- [x] GUIの中断操作からworker poolへキャンセル要求を伝搬する。
- [x] cancel token/callbackが要求済みの場合、未開始trialを即skip/cancelする。
- [x] 実行中trialは増分/iterationの安全な区切りで停止できるようにする。
- [x] cancel token/callbackでskip/cancelされた投機trialをFOS判定へ使わない。
- [ ] キャンセルされたtrialを `speculative_canceled` としてログに残す。
- [ ] GUIでは解析中ログ表示を継続する。
- [ ] GUIでworkerを積極利用しても中断応答性が維持されることを確認する。

## 9. ログ・計測

- [ ] `summary.json` にSRM並列ポリシーを出力する。
- [ ] `analysis_log.csv` にSRM並列ポリシーを出力する。
- [ ] `run.log` にSRM並列ポリシーを出力する。
- [ ] 以下の値を記録する。
  - [ ] detected logical CPUs
  - [ ] detected physical CPUs
  - [ ] available memory MB
  - [ ] selected SRM workers
  - [ ] selected threads per worker
  - [ ] lookahead depth
  - [ ] speculative trial count
  - [ ] used speculative trial count
  - [ ] unused speculative trial count
  - [ ] canceled speculative trial count
  - [ ] worker wait time
  - [ ] trial queue wait time
  - [ ] estimated wall-clock saving

## 10. YAML設定更新

- [ ] Case1-4用YAMLに `srm.parallel.policy: auto` を追加する。
- [ ] Case1-4用YAMLに `strategy: lookahead` を追加する。
- [ ] Case1-4用YAMLに `preserve_decision_order: true` を追加する。
- [ ] GUI想定の `gui_reserve_workers: 1` を追加する。
- [ ] `memory_fraction: 0.70` を追加する。
- [ ] 既存のFOS精度設定を維持する。
- [ ] Case1の `lower_projection_skip_coarse_scan_on_bracket: false` は維持する。
- [ ] Case2-4の `lower_projection_skip_coarse_scan_on_bracket: true` は維持する。

## 11. 回帰テスト

- [ ] worker数1で現行SRM結果と完全一致することを確認する。
- [ ] `preserve_decision_order: true` で逐次探索と同じFOS bracketになることを確認する。
- [ ] 粗探索lookahead有効時でもstable/failed採用順序が変わらないことを確認する。
- [x] projection並列化有効時でもCase1のFOSが変わらないことを確認する。
- [ ] bisection投機実行有効時でもFOS bracketが変わらないことを確認する。
- [ ] Case1-4で既存FOSと同等のbracketが得られることを確認する。
- [ ] GUIキャンセル時にworkerが停止し、部分結果を採用しないことを確認する。
- [ ] worker数自動設定の単体テストを追加する。
- [ ] thread数自動設定の単体テストを追加する。
- [ ] メモリ制限時にworker数が下がることを確認する。

## 12. 性能検証

- [x] Case1で逐次版とlookahead版のFOS、trial数、elapsedを比較する。
- [ ] Case2でsuspect failureの扱いが変わらないことを確認する。
- [ ] Case3で早期停止+lookaheadの短縮率を確認する。
- [ ] Case4で上側粗探索の短縮率を確認する。
- [ ] GUI実行時の中断応答性を確認する。
- [x] CLI/バッチ実行時のworker増加効果を確認する。
- [x] CPU使用率とメモリ使用量を確認する。
- [ ] 過剰並列で遅くなる条件を記録し、既定値を調整する。

## 13. 変位のみwarm start

- [ ] SRM trial間で近傍factorの変位解を初期変位として使う仕組みを追加する。
- [ ] warm startで引き継ぐ対象は変位ベクトルのみに限定する。
- [ ] 塑性状態、内部変数、履歴変数は引き継がない。
- [ ] stable trialから近傍factorへのwarm startを優先する。
- [ ] failed trialの最終受理変位を使う場合は、`last_accepted_load_factor` と診断ログを併記する。
- [ ] warm start元factorと対象factorの差が大きい場合は無効化する。
- [ ] `srm.warm_start.enabled` を追加する。
- [ ] `srm.warm_start.displacement_only: true` を既定にする。
- [ ] `srm.warm_start.max_factor_distance` を追加する。
- [ ] warm start有効/無効でFOS bracketが変わらないことを確認する。
- [ ] Newton反復数、cutback数、elapsedの差をログに出す。

設定案:

```yaml
srm:
  warm_start:
    enabled: true
    displacement_only: true
    max_factor_distance: 0.05
    prefer_stable_source: true
```

## 14. 失敗ログに基づく増分幅調整

- [ ] 直前trialの `last_accepted_load_factor` を次trialの増分制御へ反映する。
- [ ] 直前trialの `final_step_size` を次trialの初期step候補へ反映する。
- [ ] 直前trialの `cutback_count / max_cutbacks` を次trialの保守度へ反映する。
- [ ] `residual_reduction_ratio` が悪い場合は、次trialの初期stepを小さくする。
- [ ] `plastic_cluster_spans_boundary=true` かつ塑性率が高い場合は、限界近傍として初期stepを抑える。
- [ ] 増分幅調整は探索factorの順序やbracket更新には影響させない。
- [ ] 調整した増分設定をtrialログへ出力する。
- [ ] 調整前後のcutback削減効果を計測する。
- [ ] Case2のsuspect failureのような曖昧な失敗では過度に保守化しない。
- [ ] worker並列時でも、採用済みログだけを次trialの制御に使う。

設定案:

```yaml
srm:
  adaptive_increment_control:
    enabled: true
    use_last_accepted_load_factor: true
    use_final_step_size: true
    min_initial_step_factor: 0.25
    max_initial_step_factor: 1.0
```

## 15. stable/failed trialのさらなる軽量化

- [ ] SRM trial中は帳票用integration point行を作らない方針を再確認する。
- [ ] stable trialでFOS判定に不要なpostprocessをさらに遅延する。
- [ ] failed trialで最終帳票用のelement row生成を完全に抑制する。
- [ ] failed trialでは診断に必要な配列指標のみ計算する。
- [ ] `plastic_ratio`、`cluster`、`last_accepted_load_factor`、`residual`、`cutback` は維持する。
- [ ] 最終採用factorのみ通常postprocessを実行する。
- [ ] `analysis_log` に必要な診断項目が欠落しないことを確認する。
- [ ] lightweight trialの削減量を `trial_timing` に出す。
- [ ] Case1-4でFOS bracketが変わらないことを確認する。

## 16. 線形ソルバ前処理キャッシュ強化

- [ ] SRM trial間でreduced matrix抽出情報の再利用状況を再確認する。
- [ ] symbolic ordering cacheの適用漏れを再確認する。
- [ ] Newton反復ごとの前処理構築コストを計測する。
- [ ] 疎行列パターンが同一のtrialでorderingを再利用する。
- [ ] iterative solver/preconditioner利用時の前処理再利用方針を整理する。
- [ ] 直接法と反復法でログ項目を分ける。
- [ ] 失敗trial早期停止時もsolver cacheが破棄されすぎないようにする。
- [ ] cache hit/missを `summary.json` と `analysis_log` に出す。
- [ ] worker並列時にcacheを安全に共有できるものと、worker内ローカルにすべきものを分ける。
- [ ] 共有不能な数値分解はworkerごとの再利用に限定する。

## 17. 粗探索ジャンプの慎重検証

- [ ] plastic_ratio増加曲線から上側粗探索の遠方候補を推定する。
- [ ] `last_accepted_load_factor` から明らかな上側失敗候補を推定する。
- [ ] 候補factorを飛ばして採用するのではなく、まず先読み評価に限定する。
- [ ] `preserve_decision_order: true` の場合、粗探索ジャンプは採用順序を変えない。
- [ ] 逐次探索と同じbracketになることを検証する。
- [ ] FOSが変わる場合は既定無効にする。
- [ ] Case4のような長い安定粗探索でのみ有効化候補とする。
- [ ] Case1の下側探索では既定無効にする。
- [ ] 有効化条件をログに明示する。

## 18. 追加高速化の回帰・採用判定

- [ ] warm start単独の効果をCase1-4で計測する。
- [ ] 増分幅調整単独の効果をCase1-4で計測する。
- [ ] trial軽量化単独の効果をCase1-4で計測する。
- [ ] solver cache強化単独の効果をCase1-4で計測する。
- [ ] 粗探索ジャンプ候補はFOS bracket一致を最優先で検証する。
- [ ] 各高速化を組み合わせた時のFOS bracketを確認する。
- [ ] 逐次基準のFOSとの差が `factor_tol` 以下か確認する。
- [ ] Case2のsuspect failureを不当にconfirmed化していないか確認する。
- [ ] Case4の論文値との差は探索高速化ではなくモデル差として切り分ける。

## 段階導入順

1. 環境検出、自動worker/thread設定、ログ出力を実装する。
2. worker数1で現行挙動と一致することを確認する。
3. 粗探索lookaheadを実装する。
4. 下側projection候補の並列化を実装する。
5. bisection投機実行を実装する。
6. GUIキャンセル連携を強化する。
7. 変位のみwarm startを実装し、FOS bracket一致を確認する。
8. 失敗ログに基づく増分幅調整を実装し、cutback削減効果を確認する。
9. stable/failed trial軽量化を強化する。
10. 線形ソルバ前処理キャッシュを強化する。
11. 粗探索ジャンプは先読み評価限定で検証する。
12. Case1-4で精度維持と速度改善を検証する。
13. 既定値を調整し、YAMLへ反映する。

## 完了条件

- worker数1で従来結果と完全一致する。
- `preserve_decision_order: true` の並列実行でCase1-4のFOS bracketが従来と同等になる。
- 先読み/投機trialの採用・未使用・キャンセルがログで追跡できる。
- warm start有効時も塑性状態をtrial間で引き継がない。
- 増分幅調整有効時も探索factor順序とbracket更新が変わらない。
- trial軽量化後もSRM診断ログに必要項目が残る。
- solver cache強化後も線形解の回帰が通る。
- GUI実行中の中断が効く。
- Case1-4の少なくとも一部で壁時計時間が短縮する。

## 2026-07-22 Case1実測とユーザー側一括検証

- Case1は `factor_of_safety=0.682511967421`、stable-failed bracket
  `0.682511967421 - 0.686367967237` を従来実行と同値で再現した。
- Case1の壁時計は `105102.722 s -> 40934.968 s` となり、61.1%短縮、
  2.57倍高速だった。
- trialは報告9件、実評価12件、投機11件中7件採用・4件未使用だった。
- process worker 3本のbisection窓はCPU時間がほぼ均等で、直列16,531秒相当の
  区間を約9,000秒で処理した。
- 最終failed境界のみcold verificationを1回実行し、途中failed 4件は遅延した。
- Case2-4の長時間実測はユーザー側で実行する。Case1-4を同一条件で一括実行する
  `run_case1-4_srm.bat` / `run_case1-4_srm.ps1`、再開bat、停止要求bat、
  CSV/JSON集計をYAMLフォルダーへ追加した。
- 残タスク: 一括実行でCase2-4のFOS bracket、suspect failure、wall elapsedを確認し、
  Case1-4比較を完了する。
