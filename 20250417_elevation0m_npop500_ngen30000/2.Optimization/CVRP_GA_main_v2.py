from CVRP_Calculation_3d_v2 import CVRP_Calculation_3d
import time
import pandas as pd
import numpy as np

# 時間計測開始
start_time = time.time()

# 遺伝アルゴリズムのパラメータ設定
population_size = 500
crossover_rate = 0.8
mutation_rate = 0.1
generations = 30000
penalty = 1000
theta = 10

# CSVファイルを取り込み、条件に合う行を抽出して特定の列を取得
omaezaki_nodes_csv = "../1.Geography/omaezaki_nodes.csv"

try:
    # CSVファイルを読み込む
    nodes_data = pd.read_csv(omaezaki_nodes_csv)

    # "type" が "shelter" の行を抽出
    #shelter_nodes = nodes_data[nodes_data["type"] == "shelter"]

    # 必要な列だけを取得
    nodes = nodes_data[["id", "type", "x", "y", "z", "demand"]].to_dict(orient="records")

    # 結果を表示
    print(nodes)

except FileNotFoundError:
    print(f"ファイル '{omaezaki_nodes_csv}' が見つかりませんでした。パスを確認してください。")

# 対称行列（移動時間行列）の読み込み
symmetric_matrix = pd.read_csv("../1.Geography/omaezaki_symmetric_travel_time_matrix.csv", index_col=0).values

# 車両情報の読み込み
vehicles = pd.read_csv("../1.Geography/omaezaki_vehicle_info.csv").to_dict(orient="records")


# CVRP_Calculation のインスタンス生成
calculation = CVRP_Calculation_3d(
    nodes=nodes,
    vehicles=vehicles,
    cost_matrix=symmetric_matrix,
    population_size=population_size,
    crossover_rate=crossover_rate,
    mutation_rate=mutation_rate,
    generations=generations,
    penalty=penalty,
    theta= theta
)

# 初期集団を生成
population = calculation.generate_initial_population(population_size)
# 遺伝アルゴリズムの実行
best_individual, best_fitness = calculation.run_genetic_algorithm(generations)

calculation.save_vehicle_statistics(best_individual)
calculation.visualize_routes(best_individual, nodes)
calculation.visualize_routes_3d(best_individual, nodes)
#calculation.plot_elevation_changes(best_individual, nodes, cost_matrix)

# 計算時間の表示
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n計算時間: {elapsed_time:.2f} 秒")

# 計算時間をログファイルに記録
log_file='./result/log.txt'
with open(log_file, mode='a', encoding='utf-8') as log:
    log.write(f"\n計算時間: {elapsed_time:.2f} 秒\n")
