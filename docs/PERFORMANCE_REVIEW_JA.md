# GeoFEM 性能検討メモ

更新日: 2026-05-23

## 実装済み

- `fem2d_elements.py` の QUAD4 整合質量行列を Numba 化した。動的解析で使う `assemble_mass_matrix()` は QUAD4 の場合に専用カーネルを使う。
- 平面ひずみQUAD4の純弾性について、非線形解法で使う接線剛性と内力ベクトルの組立を Numba 化した。
- 平面ひずみQUAD4の J2/von Mises と Drucker-Prager について、ガウス点ごとの塑性ひずみと硬化変数を配列化し、弾塑性接線剛性と内力ベクトルを Numba 化した。テンションカットオフ、高度材料、追加状態変数ありのケースは既存の汎用経路へフォールバックする。
- 平面ひずみ応力の主応力計算とテンションカットオフを、3x3密行列固有値分解から平面ひずみ専用の解析式Numbaカーネルへ置き換えた。Post応力行、テンションカットオフ材料、数値接線の応力評価で使う。
- Mohr-Coulomb の弾性試行判定を平面ひずみ専用の解析主応力Numbaカーネルで早期判定し、弾性域では固有ベクトル計算とactive set候補探索を回避するようにした。塑性域は既存のreturn mapping経路を使う。
- Mohr-Coulomb の塑性return mappingについて、主応力空間のactive set候補生成、1から3元の小規模線形解法、補正応力と残差評価をNumbaカーネル化した。特異な候補で解けない場合は従来のPython/NumPy経路へフォールバックする。
- Mohr-Coulomb の整合接線について、主応力空間ヤコビアン、固有値接近時の極限係数、4成分摂動列のスペクトル微分をNumbaカーネル化した。特異なactive setでは従来どおり数値接線へフォールバックする。
- 平面ひずみQUAD4のMohr-Coulombについて、ガウス点ごとの塑性ひずみと硬化変数を配列化し、接線剛性と内力ベクトルの要素組立をNumba化した。内力ベクトルは接線計算を伴わない専用カーネルに分け、特異なactive setでは既存の汎用経路へフォールバックする。
- テンションカットオフ付き弾性材料の数値接線を、4成分摂動をまとめて評価するNumbaカーネルへ置き換えた。
- J2/von Mises と Drucker-Prager にテンションカットオフを併用する場合の数値接線を、塑性リターン後のカットオフ応力を4成分摂動でまとめて評価するNumbaカーネルへ置き換えた。
- Mohr-Coulomb とテンションカットオフを併用する場合の数値接線を、平面ひずみ専用の固有値分解、active set return mapping、カットオフ応力評価を含むNumba一括差分カーネルへ置き換えた。特異なactive setでは従来経路へフォールバックする。
- Mohr-Coulomb とテンションカットオフを併用する場合の整合接線について、active set固定時のMohr-Coulomb整合接線と主応力カットオフ写像ヤコビアンをNumbaで合成する経路を追加した。カットオフしきい値上や重根近傍では既存のNumba数値差分へフォールバックする。
- 軸対称QUAD4の圧力質量行列、圧力拡散行列、Biot連成行列を Numba 化した。軸対称U-P解析と軸対称Riks-U-P解析で使う。
- 軸対称QUAD4の弾性接線剛性と弾性内力ベクトルを Numba 化した。
- 軸対称QUAD4の J2/von Mises と Drucker-Prager について、ガウス点ごとの塑性ひずみと硬化変数を配列化し、弾塑性接線剛性と内力ベクトルを Numba 化した。Mohr-Coulomb、テンションカットオフ、高度材料、追加状態変数ありのケースは既存の汎用経路へフォールバックする。
- `sparse_assembly.py` を追加し、グローバル剛性、質量、Biot連成、圧力行列、非線形接線行列の組立で、Pythonリストへスカラーを逐次追加する処理を NumPy 配列チャンク集約に置き換えた。
- `mesh_generation.py` の境界節点抽出で、節点群と境界線分群の距離判定をNumba一括判定へ置き換えた。多数節点の `point_on_boundary()` 反復を避ける。
- `mesh_generation.py` の要素領域分類で、要素重心群の包含判定をNumba一括判定へ置き換えた。多領域メッシュで `first_region_index()` のPython反復を減らす。
- `mesh_generation.py` の自己交差判定とポリゴン重複判定をNumbaカーネル化し、曲線境界の修復判定や複数穴/複数領域チェックの交差ループを軽量化した。
- 平面ひずみQUAD4の純弾性Post応力行について、ガウス点座標、ひずみ、応力、主応力、p/qをNumbaで配列生成し、Python側は辞書行への変換に限定した。
- 平面ひずみQUAD4のB-bar純弾性Post応力行について、体積ひずみ平均、補正ひずみ、応力、主応力、p/qをNumbaで配列生成する専用経路を追加した。
- 平面ひずみQUAD4の J2/von Mises と Drucker-Prager のPost応力行について、履歴配列を使って応力、塑性ひずみ、硬化変数、主応力、p/qをNumbaで配列生成し、追加状態変数ありのケースは既存の汎用経路へフォールバックする。
- 平面ひずみQUAD4のB-bar J2/von Mises と Drucker-Prager のPost応力行について、B-bar補正ひずみ、応力、塑性ひずみ、硬化変数、主応力、p/qをNumbaで配列生成する専用経路を追加した。
- 平面ひずみQUAD4のMohr-Coulomb Post応力行について、FULL/B-barともに主応力フレーム、active set候補探索、補正応力、塑性ひずみ、硬化変数、主応力、p/qをNumbaで配列生成する専用経路を追加した。特異候補で失敗した場合は汎用Postへフォールバックする。
- 高度材料の履歴更新について、`state_vars` を固定長配列へ正規化し、モデルIDとパラメータ配列で `gamma_eq`、液状化 `ru`、剛性低下率、有効剛性を計算するNumbaカーネルへ切り出した。公開API境界では従来どおり辞書へ戻す。
- 強度連成を伴わない高度材料の平面ひずみQUAD4 Post応力行について、FULL/B-barともに履歴配列、剛性低下、有効D行列応力、主応力、p/qをNumbaで配列生成する専用経路を追加した。辞書行への変換はPython側に限定した。
- 強度連成を伴う高度材料の平面ひずみQUAD4 Post応力行について、UW粘土、PZ砂/粘土、液状化モデルがDrucker-Pragerまたはvon Misesへ連成する経路を、FULL/B-barともに履歴配列、有効D行列、塑性更新、塑性ひずみ、硬化変数、主応力、p/qをNumbaで配列生成する専用経路へ置き換えた。
- 強度連成を伴う高度材料の平面ひずみQUAD4非線形要素組立について、UW粘土、PZ砂/粘土、液状化モデルがDrucker-Pragerまたはvon Misesへ連成する経路を、履歴配列、有効D行列、基底塑性応力、従来同等の数値差分接線、内力ベクトルまでNumbaで一括生成する専用経路へ置き換えた。
- テンションカットオフを併用する高度材料の平面ひずみQUAD4 Post応力行について、強度連成なしの高度材料とJ2/Drucker-Prager/von Mises強度連成高度材料のFULL/B-bar経路を、カットオフ後応力、塑性フラグ、超過量、主応力、p/qまでNumbaで配列生成する専用経路へ置き換えた。
- テンションカットオフを併用する高度材料の平面ひずみQUAD4非線形要素組立について、J2/Drucker-Prager/von Mises強度連成高度材料の応力評価、数値差分接線、内力ベクトルをNumba経路へ統合した。
- `strength_model: mohr_coulomb` または `yield_surface: mc` を指定した高度材料の平面ひずみQUAD4非線形要素組立について、履歴配列、有効D行列、Mohr-Coulomb active set応力、数値差分接線、内力ベクトルをNumba経路へ統合した。
- `mesh_generation.py` の境界投影で、節点群から境界線分/円境界への最近点探索をNumba一括投影へ置き換えた。
- `analytic_boolean.py` の曲線交差・重複・近接診断で、全曲線ペアの高コスト比較を境界ボックスのソート・アンド・プルーン候補に限定した。端点ギャップと交点クラスタはグリッド候補で近傍点だけを比較する。
- `mesh_generation.py` の複数穴・複数領域ポリゴン重複チェックを境界ボックス候補に限定し、Numbaの線分交差ループにも線分境界ボックスの早期スキップを追加した。
- `mesh_generation.py` に近接点・重複点検出用の共通APIを追加し、NumPyのセルソートとNumbaの隣接セル走査で候補点だけを比較するようにした。GUIの近接点チェックもこの経路を使う。
- 疎行列ソルバーの反復法に `solver.linear.preconditioner: jacobi` を追加し、CG/GMRES/BiCGSTABへ対角前処理を渡せるようにした。直接法キャッシュと平衡化設定は従来どおり併用できる。
- 既存の QUAD4 要素剛性、圧力行列、Biot行列の Numba カーネルは維持した。
- 平面ひずみQUAD8の形状関数、ヤコビアン、B行列、FULL 3x3積分点、SRI用2x2積分点を固定長Numbaカーネルへ切り出した。
- 平面ひずみQUAD8の純弾性について、FULL/SRI/B-Barの要素剛性、非線形解法用の接線剛性、内力ベクトルをNumba化した。SRIは偏差3x3と体積2x2、B-Barは体積B行列平均を使う。
- 平面ひずみQUAD8の整合質量行列をNumba化し、動的解析で使う `assemble_mass_matrix()` から専用カーネルを使うようにした。
- 平面ひずみQUAD8の圧力質量行列、圧力拡散行列、Biot連成行列をNumba化し、U-P解析の組立でQUAD4と同様に専用カーネルを使うようにした。
- 平面ひずみQUAD8のBiot連成行列について、SRIでは体積側2x2積分、B-Barでは体積B行列平均を使う積分モード別Numba経路へ分けた。U-P解析の組立でも要素のFULL/SRI/B-Bar指定を反映する。
- 平面ひずみQUAD8の純弾性Post応力行について、FULL/SRI/B-Barのガウス点座標、ひずみ、B-Bar補正ひずみ、応力、主応力、p/qをNumbaで配列生成する専用経路を追加した。Python側は既存の辞書行変換に限定した。
- 平面ひずみQUAD8のテンションカットオフ付き弾性材料について、FULL/SRI/B-Barの数値接線剛性、内力ベクトル、Post応力行をNumba化した。Postは塑性フラグ、超過量、主応力、p/qまで配列生成し、ソルバー組立と値確認CSVで同じ高速経路を使う。
- 平面ひずみQUAD8のJ2/von MisesとDrucker-Pragerについて、FULL/SRI/B-Barの塑性ひずみ・硬化変数を固定長配列化し、弾塑性接線剛性、内力ベクトル、Post応力行をNumba化した。SRIは9点+4点の履歴を使い、Postは既存仕様どおり9点行を出力する。追加状態変数と高度材料は既存の汎用経路へフォールバックする。
- 平面ひずみQUAD8のテンションカットオフ付きJ2/von MisesとDrucker-Pragerについて、FULL/SRI/B-Barの数値接線剛性、内力ベクトル、Post応力行をNumba化した。塑性ひずみと硬化変数は既存のQUAD8状態配列を使い、カットオフ後の塑性ひずみ更新も汎用経路と同等にした。
- 平面ひずみQUAD8のMohr-Coulombについて、FULL/SRI/B-Barの塑性ひずみ・硬化変数を固定長配列化し、既存のactive set return mappingと整合接線Numba部品をQUAD8用B行列カーネルから呼び出す経路を追加した。内力、接線、Post応力行をNumbaで生成し、特異active setでは既存の汎用経路へフォールバックする。
- 平面ひずみQUAD8のテンションカットオフ付きMohr-Coulombについて、FULL/SRI/B-Barの塑性ひずみ・硬化変数を固定長配列化し、カットオフ後応力、整合接線、内力ベクトル、Post応力行をNumba経路へ統合した。特異active setや接線合成不能時は既存の汎用経路へフォールバックする。
- 平面ひずみQUAD8の強度連成を伴わない高度材料について、FULL/SRI/B-Barの履歴配列、剛性低下、有効E、Post応力行をNumba化した。テンションカットオフ付きPostも同じ9点状態配列を使い、塑性フラグ、超過量、主応力、p/qまで配列生成する。
- 平面ひずみQUAD8のJ2/Drucker-Prager/von Mises強度連成高度材料について、FULL/SRI/B-Barの履歴配列、液状化ru、有効D行列、塑性更新、Post応力行をNumba化した。テンションカットオフ付きPostも同じ経路でカットオフ後応力、塑性ひずみ、硬化変数、主応力、p/qを配列生成する。
- 平面ひずみQUAD8のJ2/Drucker-Prager/von Mises強度連成高度材料について、FULL/SRI/B-Barの非線形要素組立をNumba化した。SRIは9点の偏差側と4点の体積側の履歴配列を使い、テンションカットオフ付き数値接線と内力ベクトルも同じ専用経路で生成する。
- 平面ひずみQUAD8のMohr-Coulomb強度連成高度材料について、FULL/SRI/B-Barの履歴配列、剛性低下、有効D行列、active set return mapping、数値接線、内力ベクトル、Post応力行をNumba化した。特異active setや浮動小数例外では既存の汎用経路へフォールバックする。
- QUAD8高速化の回帰として、Mohr-Coulomb強度連成高度材料のPost応力行と非線形要素組立同等性テストを追加した。
- 軸対称QUAD8の弾性剛性と弾性内力ベクトルを、FULL/SRI/B-BarすべてでNumba化した。SRIの2x2体積積分とB-Barの体積B平均は、半径重み付き定式化を使う。
- 軸対称QUAD8の圧力質量行列、圧力拡散行列、Biot連成行列をNumba化した。軸対称U-P解析と軸対称Riks-U-P解析の組立でQUAD4と同様に専用カーネルを使う。
- 軸対称QUAD8のJ2/von MisesとDrucker-Pragerについて、FULL/SRI/B-Barの塑性ひずみ・硬化変数を固定長配列化し、半径重み付きの弾塑性接線剛性、内力ベクトル、Post応力行をNumba化した。SRIは9点+4点の履歴、B-Barは半径重み付き体積ひずみ平均を使う。
- 軸対称の2節点/3節点辺荷重について、半径重み付き線積分をNumba化した。3節点辺は3点Gauss積分で評価する。
- QUAD8専用ベンチとして、同一メッシュ密度のQUAD4/QUAD8比較、Numba初回コンパイルを除いたwarm実行、FULL/SRI/B-Bar別、平面ひずみ/軸対称別、材料モデル別の要素数スケーリングを `standard_benchmark_performance.json` の `quad8_scaling` に記録するようにした。

## 計測方針

- 変更後は `python -m geofem_app.cli benchmarks --out runs\geofem_standard_benchmarks_perf --update-baseline` で標準ベンチを更新する。
- 個別改善は、Numbaの初回コンパイル時間と2回目以降の定常時間を分けて測る。
- 性能回帰は `standard_benchmark_performance.json` の `elapsed_seconds`、`total_solver_iterations`、`max_matrix_nnz` を比較する。
