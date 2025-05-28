from gurobipy import Model, GRB, quicksum
import pandas as pd
import numpy as np
import time
import networkx as nx
from collections import defaultdict
import csv
from datetime import datetime
import os
import folium
import matplotlib.pyplot as plt

class CVRP_Gurobi_Model:
    def __init__(self, nodes, vehicles, cost_matrix, theta):
        self.V = [n['id'] for n in nodes if n['type'] == 'client']
        self.H = [n['id'] for n in nodes if n['type'] == 'shelter']
        self.D = [n['id'] for n in nodes if n['type'] == 'city_hall']
        self.N = self.D + self.V + self.H
        self.M = [v['id'] for v in vehicles]

        self.d = {n['id']: n['demand'] for n in nodes if n['type'] == 'client'}
        self.Q = {v['id']: v['capacity'] for v in vehicles}
        self.c = cost_matrix
        self.theta = theta

        self.model = Model("CVRP_Gurobi")
        self.nodes = nodes
        self.vehicles = vehicles

        self.x = None
        self.y = None
        self.best_routes = None

    def build_model(self):
        model = self.model
        N = self.N
        M = self.M
        V = self.V
        H = self.H
        D = self.D
        c = self.c

        self.x = model.addVars(N, N, M, vtype=GRB.BINARY, name="x")
        self.y = model.addVars(M, vtype=GRB.CONTINUOUS, name="y")
        self.u = model.addVars(V, M, vtype=GRB.CONTINUOUS, name="u")
        
        # 新たな変数：各車両の最大移動時間/距離を表す変数
        self.max_travel_time = model.addVar(vtype=GRB.CONTINUOUS, name="max_travel_time")
        
        # 各車両の移動時間/距離を計算する変数
        self.travel_time = model.addVars(M, vtype=GRB.CONTINUOUS, name="travel_time")
        
        # 各車両の移動時間/距離を計算
        for m in M:
            model.addConstr(
                self.travel_time[m] == quicksum(c[i][j] * self.x[i, j, m] for i in N for j in N if i != j),
                name=f"travel_time_calc_{m}"
            )
        
        # 最大移動時間/距離の制約
        for m in M:
            model.addConstr(
                self.max_travel_time >= self.travel_time[m],
                name=f"max_travel_time_{m}"
            )
        
        # ウェイトパラメータ
        alpha = 1.0  # 総移動コストの重み
        gamma = 100.0  # 最大移動時間/距離の重み
        
        # 目的関数
        model.setObjective(
            alpha * quicksum(c[i][j] * self.x[i, j, m] for i in N for j in N if i != j for m in M) +
            self.theta * quicksum(self.y[m] for m in M) +
            gamma * self.max_travel_time,
            GRB.MINIMIZE
        )

        # 以下、元々の制約と同じ...
        for v in V:
            model.addConstr(
                quicksum(self.x[i, v, m] for i in N if i != v for m in M) == 1,
                name=f"visit_once_{v}"
            )

        for i in N:
            for m in M:
                model.addConstr(self.x[i, i, m] == 0, name=f"no_self_loop_{i}_{m}")

        for v in V:
            for m in M:
                model.addConstr(
                    self.u[v, m] <= self.Q[m],
                    name=f"capacity_limit_{v}_{m}"
                )

        for i in V:
            for j in V:
                if i == j:
                    continue
                for m in M:
                    model.addConstr(
                        self.u[j, m] >= self.u[i, m] + self.d[j] - self.Q[m] * (1 - self.x[i, j, m]),
                        name=f"mtz_flow_{i}_{j}_{m}"
                    )

        for m in self.M:
            self.model.addConstr(
                self.y[m] == quicksum(self.x[i, h, m] for i in self.N for h in self.H if i != h),
                name=f"shelter_visits_{m}"
            )

        for v in V:
            model.addConstr(
                quicksum(self.x[v, h, m] for h in H for m in M if v != h) == 1,
                name=f"must_go_to_shelter_{v}"
            )

    def solve_model(self):
        self.model.optimize()

        # 訪問済み要支援者の確認
        visited_clients = set()
        for v in self.V:
            visited = False
            for i in self.N:
                for m in self.M:
                    if i != v and self.x[i, v, m].X > 0.5:
                        visited = True
                        visited_clients.add(v)
                        break
                if visited:
                    break

        unvisited_clients = set(self.V) - visited_clients
        print(f"\n=== 要支援者の搬送状況 ===")
        print(f"訪問済み要支援者数: {len(visited_clients)} / {len(self.V)}")
        if unvisited_clients:
            print(f"未訪問の要支援者: {sorted(unvisited_clients)}")
        else:
            print("すべての要支援者が搬送されています。")

        # 避難所割り当ての計算（制約から直接計算する方法）
        shelter_assignments = defaultdict(list)
        
        for h in self.H:
            clients_to_shelter = []
            for m in self.M:
                for v in self.V:
                    # 要支援者から避難所への直接アーク
                    if v != h and self.x[v, h, m].X > 0.5:
                        clients_to_shelter.append(v)
            
            shelter_assignments[h] = clients_to_shelter
        
        print("\n=== 避難所ごとの要支援者搬送一覧 ===")
        for h in self.H:
            assigned = shelter_assignments[h]
            print(f"Shelter {h}: {len(assigned)} 人 -> {sorted(assigned)}")
        
        # 避難所から車両への割り当て
        vehicle_to_shelters = defaultdict(list)
        for m in self.M:
            for h in self.H:
                clients_to_shelter = []
                for v in self.V:
                    if v != h and self.x[v, h, m].X > 0.5:
                        clients_to_shelter.append(v)
                
                if clients_to_shelter:
                    vehicle_to_shelters[m].append((h, clients_to_shelter))
        
        # 各車両のルート再構築の部分で、以下のコードを追加
        print("\n=== 各車両のルート（改良版） ===")

        for m in self.M:
            if not vehicle_to_shelters[m]:
                print(f"車両 {m}: 使用されていません")
                continue
            
            print(f"\n車両 {m} のルート:")
            
            # この車両に関連する避難所と要支援者の数をカウント
            total_shelters = len(vehicle_to_shelters[m])
            total_clients = sum(len(clients) for _, clients in vehicle_to_shelters[m])
            
            # 車両の総搬送時間/距離を計算
            total_travel_time = 0
            last_node = self.D[0]  # 市役所からスタート
            
            # ルートの構築と同時に総移動時間/距離を計算
            print(f"  搬送する要支援者数: {total_clients}")
            print(f"  利用する避難所数: {total_shelters}")
            
            # 市役所からスタート
            route_nodes = [self.D[0]]
            
            # 各避難所とそれに割り当てられた要支援者
            for idx, (shelter, clients) in enumerate(vehicle_to_shelters[m]):
                # 要支援者を追加
                route_nodes.extend(clients)
                # 避難所を追加
                route_nodes.append(shelter)
            
            # 最後に市役所に戻る
            route_nodes.append(self.D[0])
            
            # ルートに沿って総移動時間/距離を計算
            for i in range(1, len(route_nodes)):
                from_node = route_nodes[i-1]
                to_node = route_nodes[i]
                if from_node != to_node:  # 同じノードの場合はスキップ
                    total_travel_time += self.c[from_node][to_node]
            
            print(f"  総搬送時間/距離: {total_travel_time:.2f}")
            
            # ルートを表示
            print(f"  市役所(0)", end="")
            
            # 各避難所とそれに割り当てられた要支援者
            for idx, (shelter, clients) in enumerate(vehicle_to_shelters[m]):
                # 要支援者グループ
                clients_str = " → ".join([f"要支援者({c})" for c in clients])
                print(f" → {clients_str}", end="")
                
                # 避難所
                print(f" → 避難所({shelter})", end="")
            
            # 最後に市役所に戻る
            print(" → 市役所(0)")

        # 使用された車両数のカウント
        used_vehicles = sum(1 for m in self.M if vehicle_to_shelters[m])
        print(f"\n使用された車両数: {used_vehicles} / {len(self.M)}")

        self.save_shelter_assignments(vehicle_to_shelters)
        self.save_vehicle_routes(vehicle_to_shelters)
        self.save_evacuation_timeline(vehicle_to_shelters)
        self.visualize_routes_on_map(vehicle_to_shelters)
        self.save_vehicle_statistics(vehicle_to_shelters)


    def save_shelter_assignments(self, vehicle_to_shelters, output_path="shelter_assignments.csv"):
        rows = []
        for m in self.M:
            for shelter, clients in vehicle_to_shelters[m]:
                for order, client in enumerate(clients, start=1):
                    rows.append({
                        "Vehicle": m,
                        "Shelter": shelter,
                        "Client": client,
                        "Order_in_Vehicle": order
                    })
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"\n避難所搬送一覧を保存しました: {output_path}")

    def save_vehicle_routes(self, vehicle_to_shelters, output_path="vehicle_routes.csv"):
        rows = []
        for m in self.M:
            if not vehicle_to_shelters[m]:
                continue
            route = []
            route.append("Depot(0)")  # 市役所スタート
            for shelter, clients in vehicle_to_shelters[m]:
                for client in clients:
                    route.append(f"Client({client})")
                route.append(f"Shelter({shelter})")
            route.append("Depot(0)")  # 市役所帰還

            route_str = " -> ".join(route)
            rows.append({
                "Vehicle": m,
                "Route": route_str
            })
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"\n車両ルート一覧を保存しました: {output_path}")

    def report_objective_details(self):
        # 1. 総距離コスト
        total_distance_cost = sum(
            self.c[i][j] * self.x[i, j, m].X
            for i in self.N for j in self.N if i != j for m in self.M
        )
        
        # 2. 最大移動距離（変数 max_travel_time から取得）
        max_travel_distance = self.max_travel_time.X

        # 3. 搬送回数（避難所訪問回数の合計）
        total_visits = sum(self.y[m].X for m in self.M)

        # 4. 目的関数値
        objective_value = self.model.ObjVal

        # 出力
        print("\n=== 目的関数の内訳 ===")
        print(f"総距離コスト           : {total_distance_cost:.2f}")
        print(f"最大移動距離（1台あたり）: {max_travel_distance:.2f}")
        print(f"搬送回数（避難所訪問回数）: {total_visits:.0f}")
        print(f"目的関数値（合計）       : {objective_value:.2f}")

    def save_summary_report(self, elapsed_time):
        # 現在時刻
        calc_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 目的関数の内訳
        total_distance_cost = sum(
            self.c[i][j] * self.x[i, j, m].X
            for i in self.N for j in self.N if i != j for m in self.M
        )
        max_travel_distance = self.max_travel_time.X
        total_visits = sum(self.y[m].X for m in self.M)
        objective_value = self.model.ObjVal

        # メタ情報
        num_shelters = len(self.H)
        num_clients = len(self.V)
        num_vehicles = len(self.M)

        # 保存データ
        summary_data = {
            "計算日時": calc_time,
            "計算時間（秒）": round(elapsed_time, 2),
            "避難所数": num_shelters,
            "要支援者数": num_clients,
            "車両数": num_vehicles,
            "総距離コスト": round(total_distance_cost, 2),
            "最大移動距離（1台あたり）": round(max_travel_distance, 2),
            "搬送回数（避難所訪問回数）": int(total_visits),
            "目的関数値（合計）": round(objective_value, 2)
        }

        # CSVファイル名
        csv_file = "cvrp_summary_report.csv"

        # ヘッダーがなければ作成
        try:
            with open(csv_file, 'x', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=summary_data.keys())
                writer.writeheader()
        except FileExistsError:
            pass  # すでにファイルがある場合はスキップ

        # データ追記
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_data.keys())
            writer.writerow(summary_data)

        print(f"\n[INFO] 計算結果サマリーを '{csv_file}' に保存しました。")

    def save_evacuation_timeline(self, vehicle_to_shelters, output_path="evacuation_timeline.csv"):
        """
        車両ごとの搬送記録を、搬送開始時刻・搬送完了時刻を含めて出力する。
        """
        rows = []

        for m in self.M:
            if not vehicle_to_shelters[m]:
                continue

            current_time = 0.0
            transport_count = 1
            last_node = self.D[0]  # 市役所スタート

            for shelter, clients in vehicle_to_shelters[m]:
                start_times = dict()

                # 各要支援者宅への移動と搬送開始時刻の記録
                for client in clients:
                    travel_to_client = self.c[last_node][client]
                    current_time += travel_to_client
                    start_times[client] = round(current_time, 2)
                    last_node = client

                # 避難所まで移動して搬送完了時刻を決定
                travel_to_shelter = self.c[last_node][shelter]
                current_time += travel_to_shelter
                end_time = round(current_time, 2)

                # 全員分記録
                for client in clients:
                    rows.append({
                        "搬送開始時刻": start_times[client],
                        "搬送完了時刻": end_time,
                        "要支援者ID": client,
                        "避難所ID": shelter,
                        "車両ID": m,
                        "搬送回数": transport_count
                    })

                # 次の移動は避難所が起点
                last_node = shelter

                # 市役所に戻る時間も加算（次の搬送に備える）
                current_time += self.c[last_node][self.D[0]]
                last_node = self.D[0]

                transport_count += 1

        # DataFrame化 & ソート
        df = pd.DataFrame(rows)
        df = df.sort_values(by=["搬送完了時刻", "車両ID"])
        df.to_csv(output_path, index=False)
        print(f"\n[INFO] 搬送時系列データを '{output_path}' に保存しました。")


    '''
    def save_evacuation_timeline(self, vehicle_to_shelters, output_path="evacuation_timeline.csv"):
        rows = []
        for m in self.M:
            if not vehicle_to_shelters[m]:
                continue

            current_time = 0.0
            transport_count = 1  # 搬送回数（最初は1）

            last_node = self.D[0]  # 市役所スタート

            for shelter, clients in vehicle_to_shelters[m]:
                for client in clients:
                    # 前ノードから要支援者宅への移動
                    current_time += self.c[last_node][client]
                    
                    # 要支援者宅から避難所への移動
                    current_time += self.c[client][shelter]

                    # データ記録
                    rows.append({
                        "時刻": round(current_time, 2),
                        "要支援者ID": client,
                        "避難所ID": shelter,
                        "車両ID": m,
                        "搬送回数": transport_count
                    })

                    last_node = shelter  # 次のスタート地点は避難所

                # 避難所から市役所に戻る
                current_time += self.c[last_node][self.D[0]]
                last_node = self.D[0]  # 市役所に戻る

                # 次の搬送準備
                transport_count += 1

        # DataFrame化して時系列順に並べ替え
        df = pd.DataFrame(rows)
        df = df.sort_values(by="時刻")
        df.to_csv(output_path, index=False)
        print(f"\n[INFO] 避難時系列データを '{output_path}' に保存しました。")
    '''

    def visualize_routes_on_map(self, vehicle_to_shelters, output_dir='result/vehicle_maps'):
        """
        全車両のルートマップと、各車両ごとの個別ルートマップを地図上に可視化する。
        :param vehicle_to_shelters: 各車両ごとの避難所と要支援者の割り当て情報
        :param output_dir: 出力先フォルダ
        """
        os.makedirs(output_dir, exist_ok=True)

        # 市役所の位置を中心にマップ作成
        city_hall = next(node for node in self.nodes if node["type"] == "city_hall")
        node_positions = {node["id"]: (node["y"], node["x"]) for node in self.nodes}

        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                'lightblue', 'lightgreen', 'cadetblue', 'darkpurple']

        ### ① 全車両まとめたマップ
        all_map = folium.Map(location=[city_hall["y"], city_hall["x"]], zoom_start=13)

        for idx, vehicle_id in enumerate(self.M):
            if not vehicle_to_shelters[vehicle_id]:
                continue

            route_nodes = [self.D[0]]
            for shelter, clients in vehicle_to_shelters[vehicle_id]:
                route_nodes.extend(clients)
                route_nodes.append(shelter)
            route_nodes.append(self.D[0])

            route_coords = [node_positions[n] for n in route_nodes]

            folium.PolyLine(
                locations=route_coords, 
                color=colors[idx % len(colors)], 
                weight=4, 
                opacity=0.7,
                tooltip=f'Vehicle {vehicle_id}'
            ).add_to(all_map)

        # ノードマーカー（全体マップに共通表示）
        for node in self.nodes:
            coord = (node["y"], node["x"])
            if node["type"] == "city_hall":
                folium.Marker(coord, icon=folium.Icon(color='blue', icon='building'), tooltip="City Hall").add_to(all_map)
            elif node["type"] == "shelter":
                folium.Marker(coord, icon=folium.Icon(color='green', icon='home'), tooltip=f"Shelter {node['id']}").add_to(all_map)
            elif node["type"] == "client":
                folium.CircleMarker(coord, radius=3, color='orange', fill=True, tooltip=f"Client {node['id']}").add_to(all_map)

        all_map.save(f"{output_dir}/all_vehicle_routes_map.html")
        print(f"[INFO] 全車両のルートマップを保存しました: {output_dir}/all_vehicle_routes_map.html")

        ### ② 各車両ごとのマップ
        for idx, vehicle_id in enumerate(self.M):
            if not vehicle_to_shelters[vehicle_id]:
                continue

            v_map = folium.Map(location=[city_hall["y"], city_hall["x"]], zoom_start=13)

            route_nodes = [self.D[0]]
            for shelter, clients in vehicle_to_shelters[vehicle_id]:
                route_nodes.extend(clients)
                route_nodes.append(shelter)
            route_nodes.append(self.D[0])

            route_coords = [node_positions[n] for n in route_nodes]

            folium.PolyLine(
                locations=route_coords, 
                color=colors[idx % len(colors)], 
                weight=5, 
                opacity=0.9,
                tooltip=f'Vehicle {vehicle_id}'
            ).add_to(v_map)

            # ノードマーカー（個別マップにも表示）
            for node in self.nodes:
                coord = (node["y"], node["x"])
                if node["type"] == "city_hall":
                    folium.Marker(coord, icon=folium.Icon(color='blue', icon='building'), tooltip="City Hall").add_to(v_map)
                elif node["type"] == "shelter":
                    folium.Marker(coord, icon=folium.Icon(color='green', icon='home'), tooltip=f"Shelter {node['id']}").add_to(v_map)
                elif node["type"] == "client":
                    folium.CircleMarker(coord, radius=3, color='orange', fill=True, tooltip=f"Client {node['id']}").add_to(v_map)

            v_map.save(f"{output_dir}/vehicle_{vehicle_id}_route_map.html")
            print(f"[INFO] 車両 {vehicle_id} のルートマップを保存しました: {output_dir}/vehicle_{vehicle_id}_route_map.html")

    def save_vehicle_statistics(self, vehicle_to_shelters, output_dir='result'):
        """
        各車両の移動距離、搬送人数、避難所訪問回数を集計し、CSVとグラフで保存する。
        :param vehicle_to_shelters: 各車両ごとの搬送情報
        :param output_dir: 保存先ディレクトリ
        """
        os.makedirs(output_dir, exist_ok=True)

        vehicle_stats = []

        for m in self.M:
            if not vehicle_to_shelters[m]:
                continue

            total_distance = 0
            total_clients = 0
            shelter_visits = 0

            route_nodes = [self.D[0]]  # 市役所スタート
            for shelter, clients in vehicle_to_shelters[m]:
                route_nodes.extend(clients)
                route_nodes.append(shelter)
                total_clients += len(clients)
                shelter_visits += 1
            route_nodes.append(self.D[0])  # 市役所帰還

            # 総移動距離計算
            for i in range(len(route_nodes) - 1):
                from_node = route_nodes[i]
                to_node = route_nodes[i + 1]
                total_distance += self.c[from_node][to_node]

            vehicle_stats.append({
                "Vehicle": m,
                "Total Distance": round(total_distance, 2),
                "Total Clients": total_clients,
                "Shelter Visits": shelter_visits
            })

        # CSV保存
        df_stats = pd.DataFrame(vehicle_stats)
        csv_path = os.path.join(output_dir, 'vehicle_statistics.csv')
        df_stats.to_csv(csv_path, index=False)
        print(f"\n[INFO] 車両統計データを保存しました: {csv_path}")

        # 各統計の棒グラフを作成
        self._plot_vehicle_stat(df_stats, 'Total Distance', os.path.join(output_dir, 'vehicle_costs.png'))
        self._plot_vehicle_stat(df_stats, 'Total Clients', os.path.join(output_dir, 'vehicle_loads.png'))
        self._plot_vehicle_stat(df_stats, 'Shelter Visits', os.path.join(output_dir, 'vehicle_shelter_visits.png'))

    def _plot_vehicle_stat(self, df_stats, column_name, output_path):
        """
        車両統計データの棒グラフを作成して保存する補助関数。
        :param df_stats: 車両統計のDataFrame
        :param column_name: プロット対象の列名
        :param output_path: 画像保存パス
        """
        plt.figure(figsize=(10, 6))
        plt.bar(df_stats['Vehicle'].astype(str), df_stats[column_name])
        plt.xlabel('Vehicle ID')
        plt.ylabel(column_name)
        plt.title(f'Vehicle-wise {column_name}')
        plt.grid(axis='y')
        plt.savefig(output_path)
        plt.close()
        print(f"[INFO] {column_name} のグラフを保存しました: {output_path}")


# ---------------------------------------
# ↓ 実行スクリプト
# ---------------------------------------

if __name__ == "__main__":
    start_time = time.time()

    # ファイル読み込み
    nodes_data = pd.read_csv("../1.Geography/omaezaki_nodes_class0.csv")
    nodes = nodes_data[["id", "type", "x", "y", "z", "demand"]].to_dict(orient="records")
    symmetric_matrix = pd.read_csv("../1.Geography/omaezaki_symmetric_travel_time_matrix.csv", index_col=0).values
    vehicles = pd.read_csv("../1.Geography/omaezaki_vehicle_info.csv").to_dict(orient="records")

    # パラメータ設定
    theta = 0

    # クラスインスタンス生成
    cvrp_model = CVRP_Gurobi_Model(nodes=nodes, vehicles=vehicles, cost_matrix=symmetric_matrix, theta=theta)

    # モデル構築と解決
    cvrp_model.build_model()
    cvrp_model.solve_model()

    # 計算結果サマリー保存
    elapsed_time = time.time() - start_time
    cvrp_model.save_summary_report(elapsed_time)

    print(f"\n計算時間: {elapsed_time:.2f} 秒")