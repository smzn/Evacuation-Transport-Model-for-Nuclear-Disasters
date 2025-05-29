This repository contains the source code used in the research paper "Development of an Evacuation Transport Model for Residents in Need of Assistance in Evacuation during Nuclear Disasters".

Two main optimization approaches are implemented in this repository:
1. Exact optimization using the Gurobi solver
2. Heuristic optimization using a Genetic Algorithm (GA)

■ Optimization using Gurobi

[Target Directory]
20250511_elevationXXm_gurubi_client12m/
(* Replace XX with the elevation threshold in meters. For example, elevation6m means roads below 6m are excluded.)

[Steps]

1. Run the script `CVRP_Geography_v7.py` in the Geography folder.

This script filters out roads below the specified elevation, creating a road network and travel-time matrix suitable for evacuation planning.

Example: To include only areas at or above 6 meters
(Python code)

```python
elev = 6
geo.get_filtered_road_network(output_file=f"omaezaki_≤{elev}melev.graphml", elev=elev, nrate=0.5)
print('Network recreated', time.time() - start)
```

2. Run the script `CVRP_gurobi_3d_v2.py` in the Optimization folder.

This script uses the preprocessed road network and travel-time matrix to optimize the evacuation plan using the Gurobi solver.

■ Optimization using Genetic Algorithm (GA)

[Target Directory]
20250417_elevation0m_npop500_ngen30000/
(* GA is applied only for the case with no elevation restriction: elevation = 0m)

[Steps]

1. Run `CVRP_Geography_v7.py` in the Geography folder (same as Gurobi procedure).

2. Run `CVRP_Calculation_3d_v2.py` in the Optimization folder.

This script applies a Genetic Algorithm to iteratively search for optimized evacuation routes.

■ Notes

- Geographic data is based on OpenStreetMap and elevation data sources.
- The optimization model is formulated as a Capacitated Vehicle Routing Problem (CVRP).
- To reproduce the results, a Python environment and Gurobi installation are required.
　Gurobi requires a valid license to run.

---


このリポジトリは、論文「Development of an Evacuation Transport Model for Residents in Need of Assistance in Evacuation during Nuclear Disaster*」において用いた避難支援者搬送計画モデルのソースコード一式です。
本研究では、災害時における要支援者の避難支援を目的とし、制限高度を考慮した地理情報をもとに、搬送ルートを最適化する数理モデルを構築しています。

本コードでは、次の2種類の最適化手法を用いて実験を行っています。
1. 数理最適化ソルバー「Gurobi」による厳密解法
2. 遺伝的アルゴリズム（GA）による近似解法

■ Gurobi による最適化手法

【対象ディレクトリ】
20250511_elevationXXm_gurubi_client12m/
※ XX には制限高度（メートル単位）が入ります（例：elevation6m）

【手順】

1. Geography フォルダのスクリプト `CVRP_Geography_v7.py` を実行

このスクリプトでは、指定された高度よりも低い領域を地図から除外し、
現実的な避難ルート用ネットワークおよび時間マトリクスを作成します。

例：6m以上の地域のみ対象とする設定
（以下、Pythonコード）

```python
elev = 6
geo.get_filtered_road_network(output_file=f"omaezaki_≤{elev}melev.graphml", elev=elev, nrate=0.5)
print('ネットワーク再作成', time.time() - start)
```

2. Optimization フォルダ内の `CVRP_gurobi_3d_v2.py` を実行

このスクリプトでは、上記で作成された地図および時間マトリクスを用いて、
Gurobi により搬送計画（CVRP）の最適化を行います。

■ GA（遺伝的アルゴリズム）による最適化手法

【対象ディレクトリ】
20250417_elevation0m_npop500_ngen30000/
※ GAは elevation=0m のケースのみ対応しています。

【手順】

1. Geography フォルダの `CVRP_Geography_v7.py` を実行（上記と同様）

2. Optimization フォルダの `CVRP_Calculation_3d_v2.py` を実行

このスクリプトでは、GAにより進化的に最適なルートを探索します。

■ 備考

・地理情報の取得・処理には OpenStreetMap のオープンデータを使用しています。
・最適化問題は Capacitated Vehicle Routing Problem（CVRP）として定式化されています。
・研究の再現には Python 環境および Gurobi の導入が必要です。
　Gurobi の実行にはライセンスが必要です。
