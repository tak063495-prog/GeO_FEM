# GeoFEM 開発履歴と失敗分類

作成日: 2026-08-21

## 1. 目的と証拠の範囲

本書は、GeoFEM 2D の開発過程で発生した失敗、不具合、精度上の不確定事項、性能上の逆効果を、今後の実装・レビュー・回帰試験で再利用できる知見として整理したものです。

Gitリポジトリは存在しますが、作業時点でコミット履歴は確認できず、主要ファイルは未追跡でした。そのため、以下の優先順位で履歴を再構成しています。

1. 既存の実行ログ、`summary.json`、`analysis_log.*`、監視ログ
2. ToDo、性能レビュー、検証文書、テストコード
3. 会話記録に残る実ケース結果と修正経緯

会話記録だけで補った内容は「推定」、実ケースの再実行が必要な内容は「未解決」と明示します。

## 2. 開発の大きな流れ

| 時期 | 主な内容 | 失敗・学び |
|---|---|---|
| 2026-05 | 大変形・微小変形・SRMのキャッシュ、配列化、疎行列高速化を拡大 | 局所最適化だけでは、SRM探索回数、fallback、帳票生成が支配的になる |
| 2026-05〜06 | Case1〜4のYAML作成、FOS探索、adaptive bracket、診断ログを追加 | 数値非収束を物理破壊と同一視するとFOSが低く出る。経路依存性も顕在化 |
| 2026-06 | 自動worker数、thread制御、lookahead、投機実行、cancel、安全停止を実装 | 並列化は採用順序と中断応答性を同時に管理しないと再現性を損なう |
| 2026-07 | Mohr-Coulomb正則化・active-set・Numba、solver全体の性能回帰 | Numba化後もPython fallbackや後処理が残る。速度と精度のガードが必要 |
| 2026-07 | GUIの凍結防止、帳票遅延生成、結果表モデル化、LOD描画、Phase 1〜6 | 解析時間以外に、イベントループ、結果表示、帳票が体感性能を決める |
| 2026-08 | Auto SRM・ログ判定・GUI動線の精錬 | 実ケースでのFOS精度確認と、別環境での配布再現性が残る重要課題 |

## 3. 失敗分類一覧

| ID | 分類 | 代表症状 | 状態 | 証拠レベル |
|---|---|---|---|---|
| HIS-01 | 開発履歴・追跡性 | コミット履歴がなく、変更理由と採用版を厳密に辿れない | 未解決 | 確定 |
| INP-01 | 入力・YAML | `boundary_conditions[2].set` が存在しない節点セットを参照 | 改善済み、入力検証は継続 | 確定 |
| NUM-01 | 構成則・数値精度 | Case2/4で論文値とFOSが離れ、regularization/fallbackの検証が必要 | 未解決 | 確定 |
| NUM-02 | SRM失敗判定 | 非収束、cutback上限、物理破壊、数値不安定の区別が難しい | 改善済みだが実ケース再確認要 | 確定 |
| NUM-03 | 探索経路依存 | adaptive bracket、先読み、早期打切りでFOSが変動 | 未解決 | 確定 |
| PERF-01 | 性能計測 | warmup、I/O、帳票、並列CPU競合がelapsedを揺らす | 部分改善 | 確定 |
| PERF-02 | 高速化fallback | Mohr-Coulomb Python fallback、postprocess、solver overheadが支配 | 部分改善 | 確定 |
| GUI-01 | GUI応答性 | 高メッシュCaseでwatchdog遅延、最大約18.5秒の報告。現存ログでは最大約9.6秒 | 改善済み、実機再確認要 | 確定 |
| GUI-02 | GUI動線・状態 | 工程間でボタン位置がずれる、結果表示が隠れる、空結果でも操作が残る | 改善済み | 確定 |
| PKG-01 | 配布・依存関係 | BAT起動失敗、Python依存、PyInstallerのDLL/hidden import警告 | 部分改善 | 確定 |
| OBS-01 | 観測可能性 | 初期は失敗trialの診断指標やFOS進行が不足 | 改善済み | 確定 |
| QA-01 | 回帰・実ケース検証 | 単体・標準ベンチは通るが、Case1〜4の完全な同一環境比較が後回し | 未解決 | 確定 |

## 4. 詳細な失敗と得られた知見

### 4.1 構成則・FOS精度の失敗（NUM-01）

Case2/4では、論文値に近いかどうかと、ソルバーが収束したかどうかが一致しませんでした。記録には `factor_of_safety_status=material_fallback_evidence`、`factor_of_safety_confidence=limited`、`material_fallback_verification_required=true` が残る実行があります。これは、FOS値を確定値として画面表示する前に、構成則fallback、正則化、メッシュ、境界条件、荷重経路を分離検証すべきことを示します。

代表的な記録では、Case2 strict実行のFOSは1.340625、elapsedは約1157秒でしたが、solver部分は約4秒、SRM trial overheadが約1153秒でした。単純な要素kernel高速化だけでは、探索・検証・ログ処理の設計問題を解消できません。

得られたルール:

- `収束した` と `FOSが妥当` は別の判定にする
- regularization/fallback発生時は、FOSに信頼度と再検証要求を必ず付ける
- 論文値との差は、まず材料・メッシュ・境界・荷重経路・failure定義を分解して比較する
- 構成則の高速経路は、参照経路との応力、yield、active-set、塑性乗数の一致試験を通してから実ケースへ適用する

### 4.2 SRM探索・非収束判定の失敗（NUM-02/NUM-03）

初期のSRMはfactorの総当たりで時間がかかり、非収束trialを即failureと扱うとFOSを低く見積もる危険がありました。Case1では自重解析と失敗trialの `last_accepted_load_factor`、塑性率、cluster、cutback、残差、line searchを用いた判定へ拡張しました。

その後、adaptive bracket、bisection、coarse-to-fine、lookahead、projection、warm start、early stop、deferred verification、checkpoint continuationを導入しましたが、探索経路を変える最適化は数値経路依存性を持ちます。したがって、先読みtrialは完了順ではなく従来factor順で採用し、未使用trialをFOS判定へ混ぜない設計が重要です。

今後の基準:

- worker=1の結果を再現性基準にする
- `preserve_decision_order=true` を既定にする
- 早期打切りは、単一指標でなく塑性率、連結cluster、cutback、残差減少率、変位増分、detJ、仕事比を総合評価する
- failed trialは、物理failure、数値非収束、未確定、キャンセルを別状態で保存する
- factor_tolを狭めた確認と高速Auto探索を別の運用モードにする

### 4.3 性能最適化の失敗（PERF-01/02）

性能改善では、factor間cache、StepCache、疎行列direct-fill、内力と接線の同時組立、塑性状態配列化、圧密/u-p/軸対称/dynamicへのcache移植、QUAD8/TRIバッチ化、Mohr-Coulomb Numba化を順次実装しました。

一方で、次の失敗パターンが確認されました。

- Numba warmupをelapsedへ含めると初回だけ遅く見える
- 帳票、HTML、PDF、result view生成がsolverより目立つ
- Python fallbackが残ると、kernel高速化の効果が隠れる
- 並列worker増加はCPU競合、メモリ、GUI応答性で逆効果になり得る
- 小規模ケースではcache構築費が再利用効果を上回る

得られたルール:

- cold、warm、solver、post、I/O、reportを分離計測する
- FOS bracket、trial採用順、反復数、残差、fallback件数、メモリをelapsedと同時に比較する
- 数値結果が変わる高速化は、性能改善ではなく別アルゴリズムとして精度ゲートを通す
- 小規模/大規模、GUI/CLI、worker=1/autoを別ベンチにする

### 4.4 GUI・結果表示の失敗（GUI-01/02）

高メッシュSRMでは、プレビューの再構築、メッシュ診断、結果CSV全量読込、QTableWidgetItem大量生成、個別GraphicsItem大量生成、帳票生成がイベントループを塞ぎました。工程遷移では中央枠の幅が変わり、戻る/次へボタンがずれる問題も発生しました。

改善では、プレビューdebounce・設定hash cache、診断再利用、CSVの必要部分だけの走査、QAbstractTableModel、LOD描画、結果サマリ中心表示、実行中のナビゲーション制御、読み取り専用結果参照、空結果時の操作非表示、左ナビ/中央/右ペインの責務整理を導入しました。

現存の `gui_freeze_watchdog.json` では最大約9.6秒の遅延が記録され、会話上のCase実行では約18.5秒の遅延も報告されました。GUI改善後の単体GUI回帰は会話記録上67件、モデルチェック系10件が通過していますが、実大規模CaseのGUI応答性は未解決の実機検証項目です。

### 4.5 入力・配布の失敗（INP-01/PKG-01）

入力では、存在しない節点セット参照が解析前に検出されました。これはソルバー内部で失敗させるより、入力診断で対象セルへ戻せることが重要です。Case YAMLが複数フォルダーに存在し、どれが最新か分かりにくい問題もあり、canonical YAMLと実行結果の出力先を一元化する必要がありました。

配布では、BAT実行時のPython/依存ライブラリ不足、PyInstaller版の` tbb12.dll `警告、`scipy.special._cdflib` hidden import警告が問題になりました。配布版は「起動する」だけでなく、別PCで依存診断、hidden import、DLL、Case YAML、出力先、ログ生成まで検査する必要があります。

## 5. 今後のソフト開発で参照する知見

1. 数値・探索・表示・配布を一つの成功判定にしない。各層に独立した受入条件を置く。
2. FOSは値だけでなく、stable/failed bracket、failure class、信頼度、fallback、探索設定を保存する。
3. 高速化の受入条件は、速度だけでなく、worker=1との結果一致、採用順序一致、残差・反復数の許容範囲を含める。
4. 失敗trialは削除しない。軽量な診断配列として、次の探索判断と後解析に使える形で残す。
5. 実行中ログは、解析終了後の帳票より先に設計する。少なくともfactor、trial状態、last accepted、cutback、plastic指標、残差、elapsedを出す。
6. GUIはsolver完了待ちの画面ではなく、中断可能な状態機械として設計する。
7. YAMLのcanonical版、実行時snapshot、summary、analysis_log、環境情報を同じrun directoryへ保存する。
8. Gitコミット、変更理由、ベンチ結果、精度ゲートを一つの変更単位で残す。今回のようにコミットがない状態は再現性を大きく下げる。
9. 論文値との差は、すぐにバグまたは材料定数の誤りと断定せず、failure定義、荷重経路、メッシュ、構成則、境界条件、収束基準を順に切り分ける。
10. 実ケースCase1〜4は重くても、リリース前に同一マシン・単独実行・worker=1・autoの最低4条件を記録する。

## 6. 残存リスクと推奨アクション

| 優先度 | アクション | 完了条件 |
|---|---|---|
| P0 | Git初期コミットと変更単位の運用を開始 | cleanなbaseline、変更理由、テスト結果がcommitから辿れる |
| P0 | Case1〜4の精度確認をworker=1で固定 | FOS bracket、failure class、fallback、factor_tolを比較できる |
| P1 | material fallback/regularizationの厳格確認経路を分離 | limitedではなく、検証済み/要確認を明示して報告 |
| P1 | Auto探索とstrict FOS confirmationの二段運用を固定 | 高速探索が厳格値の代替として誤表示されない |
| P1 | 別PC配布の起動・依存・DLL・Case実行を自動検査 | PyInstaller smoke testと最小Caseが通る |
| P2 | 大規模GUIの実機応答性を測定 | watchdog、メモリ、描画時間、停止応答時間の閾値を満たす |
| P2 | performance baselineを環境別に管理 | cold/warm、solver/I/O、worker=1/autoを比較可能 |

## 7. 主要な参照資料

- `SRM_PARALLEL_LOOKAHEAD_TODO_JA.md`: SRM先読み、worker、warm start、早期打切り、未検証項目
- `docs/PERFORMANCE_REVIEW_JA.md`: ソルバー高速化と性能計測方針
- `docs/CASE2_CASE4_MC_NUMBA_VALIDATION_JA.md`: Mohr-Coulomb Numba検証手順
- `unresolved_items.md`: 品質改善バックログの集約先
- `runs/phase6_20260720_verified/case1..4/summary.json`: Case1〜4の実行記録
- `runs/case1-4_speed_validation_20260722/case1/analysis_log.json`: SRM failure score、early stop、fallbackの代表記録
- `runs/case2-4_mc_numba_validation_20260725_220415/`: Case2/4のstrict・Numba検証記録
- `gui_freeze_watchdog.json`: GUIイベントループ遅延記録
- `.geofem_audit_log.jsonl`: 承認差戻し、再承認、autosaveの監査記録
