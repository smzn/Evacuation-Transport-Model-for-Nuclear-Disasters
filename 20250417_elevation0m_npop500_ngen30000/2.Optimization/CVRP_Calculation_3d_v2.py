import random
import numpy as np
import csv
import matplotlib.pyplot as plt
import copy
import os
from mpl_toolkits.mplot3d import Axes3D

class CVRP_Calculation_3d:
    def __init__(self, nodes, vehicles, cost_matrix, population_size, crossover_rate, mutation_rate, generations, penalty, theta):
        """
        CVRP計算クラスの初期化。
        :param nodes: ノード情報（辞書リスト）
        :param vehicles: 車両情報（辞書リスト）
        :param cost_matrix: 移動コスト行列（numpy.array）
        :param population_size: 集団サイズ
        :param crossover_rate: 交叉率
        :param mutation_rate: 突然変異率
        :param generations: 世代数
        :param penalty: ペナルティ係数
        """
        # ノード情報から属性を設定
        self.V = [node["id"] for node in nodes if node["type"] == "client"]  # クライアントノード
        self.H = [node["id"] for node in nodes if node["type"] == "shelter"]  # 避難所ノード
        self.d = {node["id"]: node["demand"] for node in nodes if node["type"] == "client"}  # 各クライアントの需要

        # 車両情報から属性を設定
        self.M = [vehicle["id"] for vehicle in vehicles]  # 車両のIDリスト
        self.Q = {vehicle["id"]: vehicle["capacity"] for vehicle in vehicles}  # 各車両の最大積載量

        self.y = {m: 0 for m in self.M}  # 搬送回数（避難所訪問回数）を車両ごとに初期化
        self.theta = theta  # 追加

        # コスト行列
        self.c = cost_matrix

        # 辺集合を生成
        self.E = [(i, j) for i in range(len(nodes)) for j in range(len(nodes)) if i != j]

        # 遺伝アルゴリズムのパラメータ
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.penalty = penalty

        # 待ち時間と搬送時間の記録用
        self.waiting_times = {client: 0 for client in self.V}  # 要支援者IDごとの待ち時間
        self.transport_times = {client: 0 for client in self.V}  # 要支援者IDごとの搬送時間

    def evaluate_individual(self, individual):
        """
        目的関数: 移動コスト + 搬送回数ペナルティ（theta × y_m）
        :param individual: 個体（各車両のルート）
        :param theta: 搬送回数ペナルティ係数（θ）
        :return: 総コスト（目的関数値）
        """
        if not self.check_constraints(individual):
            return float('inf')  # 制約違反には大きな値を返す

        total_cost = 0
        self.y = {m: 0 for m in self.M}

        for vehicle_index, vehicle_route in enumerate(individual):
            if not vehicle_route:
                continue

            m = self.M[vehicle_index]
            load = 0
            route_cost = 0
            previous_node = 0

            for node in vehicle_route:
                if node == 0:
                    previous_node = node
                    continue

                if node in self.V:
                    load += self.d[node]  # ← キャパ確認は check_constraints で済んでいる
                elif node in self.H:
                    self.y[m] += 1
                    load = 0

                route_cost += self.c[previous_node][node]
                previous_node = node

            route_cost += self.c[previous_node][0]
            total_cost += route_cost

        penalty_by_y = sum(self.theta * y_m for y_m in self.y.values())

        print(f"[DEBUG] total_cost = {total_cost}, penalty_by_y = {penalty_by_y}, fitness = {total_cost + penalty_by_y}")

        return total_cost + penalty_by_y

    '''
    def evaluate_individual(self, individual):
        """
        目的関数: 制約確認 + 総移動コスト + ペナルティ
        :param individual: 個体 (遺伝子表現: 各車両のルートのリスト)
        :return: 総コスト (目的関数値) または大きなペナルティ値
        """
        # 制約確認
        if not self.check_constraints(individual):
            return float('inf')  # 制約違反の場合、大きなペナルティ値を返す

        # 制約を満たしている場合、コスト計算
        total_cost = 0
        vehicle_penalties = 0

        for vehicle_route in individual:
            if not vehicle_route:
                continue

            load = 0
            route_cost = 0
            previous_node = 0  # デポ (市役所) から出発

            for node in vehicle_route:
                if node == 0:  # デポは積載量に影響を与えない
                    previous_node = node  # デポを次の計算の基点に設定
                    continue

                if node in self.V:  # 要支援者ノードの場合のみ処理
                    load += self.d[node]  # 要支援者の需要を加算
                    if load > max(self.Q.values()):  # 最大積載量を超えた場合
                        vehicle_penalties += self.penalty

                # 移動コストを加算
                route_cost += self.c[previous_node][node]
                previous_node = node

            # デポに戻るコストを加算
            route_cost += self.c[previous_node][0]

            total_cost += route_cost

        return total_cost + vehicle_penalties
    '''
    
    def generate_initial_population(self, population_size=1):
        """
        初期集団を生成し、制約を確認する。
        要支援者全体のルートを生成し、それを車両台数で分割。
        キャパを超える前に避難所を挿入してルートを作成する。
        :return: 制約を満たす初期集団 (リスト)
        """
        population = []
        for _ in range(population_size):
            while True:
                # ステップ 1: 要支援者全体のルートをシャッフルして生成
                shuffled_clients = random.sample(self.V, len(self.V))
                #print(f"  Shuffled clients: {shuffled_clients}")

                # ステップ 2: 台数分に均等分割
                num_vehicles = len(self.Q)
                split_size = len(shuffled_clients) // num_vehicles
                split_routes = [
                    shuffled_clients[i * split_size:(i + 1) * split_size]
                    for i in range(num_vehicles)
                ]
                # 余りを最後の車両に追加
                remainder = len(shuffled_clients) % num_vehicles
                if remainder > 0:
                    split_routes[-1].extend(shuffled_clients[-remainder:])

                #print(f"  Split routes (before capacity adjustment): {split_routes}")

                # ステップ 3: 各ルートにキャパを考慮して避難所を挿入
                individual = []
                for route_index, route in enumerate(split_routes):
                    current_vehicle_route = [0]  # デポで開始
                    current_load = 0  # 現在の積載量

                    for client in route:
                        client_demand = self.d[client]
                        if current_load + client_demand <= max(self.Q.values()):  # キャパシティ確認
                            # キャパ内なら要支援者を追加
                            current_vehicle_route.append(client)
                            current_load += client_demand
                        else:
                            # キャパ超過前に避難所を追加
                            last_client = current_vehicle_route[-1]
                            #print(f"Debug: Adding shelter. last_client: {last_client}, current_load: {current_load}, client_demand: {client_demand}")
                            #print(f"Debug: Route before shelter addition: {current_vehicle_route}")
                            # 避難所選択時のデバッグ情報
                            #print(f"self.H (候補の避難所リスト): {self.H}")
                            #print(f"self.c.shape (行列のサイズ): {self.c.shape}")

                            try:
                                nearest_shelter = min(self.H, key=lambda shelter: self.c[last_client][shelter])
                            except IndexError as e:
                                print(f"IndexError occurred for last_client {last_client} and shelter selection. Error: {e}")
                                print(f"Last client: {last_client}, self.H: {self.H}, c.shape: {self.c.shape}")
                                raise  # エラーを再度スローして終了
                            #nearest_shelter = min(self.H, key=lambda shelter: self.c[last_client][shelter])
                            current_vehicle_route.append(nearest_shelter)
                            #print(f"      Added shelter {nearest_shelter} to route, resetting load.")
                            current_load = client_demand  # キャパリセット
                            current_vehicle_route.append(client)

                    # 最後に避難所を追加してデポに戻る
                    if current_vehicle_route[-1] not in self.H:
                        last_client = current_vehicle_route[-1]
                        nearest_shelter = min(self.H, key=lambda shelter: self.c[last_client][shelter])
                        current_vehicle_route.append(nearest_shelter)
                    current_vehicle_route.append(0)  # デポに戻る
                    individual.append(current_vehicle_route)
                    #print(f"    Final route for vehicle {route_index + 1}: {current_vehicle_route}")

                # 制約確認: 制約を満たす場合のみ集団に追加
                if self.check_constraints(individual):
                    population.append(individual)
                    #print(f"  Added individual to population.")
                    break  # 有効な個体を生成した場合、次へ
                else:
                    print("  Constraint check failed. Retrying...")

        #print(f"Initial population size: {len(population)}")
        return population
    
    
    def check_constraints(self, individual):
        """
        制約を確認する。
        :param individual: 車両ルート（リスト）
        :return: True（制約を満たす場合）または False（制約違反の場合）
        """
        visited_clients = set()  # 訪問済みの要支援者
        #print("  Checking constraints...")

        for route_index, route in enumerate(individual):
            current_load = 0  # 現在の積載量

            for node in route:
                if node == 0:  # デポは無視
                    continue
                if node in self.V:  # 要支援者ノードの場合
                    if node in visited_clients:
                        print(f"    Constraint failed: client {node} visited multiple times in route {route_index + 1}.")
                        return False
                    visited_clients.add(node)
                    current_load += self.d[node]
                    if current_load > max(self.Q.values()):  # キャパ超過
                        print(f"    Constraint failed: capacity exceeded in route {route_index + 1}. Load: {current_load}")
                        return False
                elif node in self.H:  # 避難所ノードの場合
                    #print(f"    Reached shelter {node}, resetting load.")
                    current_load = 0  # キャパリセット
                else:
                    print(f"    Unknown node {node} in route {route_index + 1}.")
                    return False

        # すべての要支援者が訪問されているか確認
        if visited_clients != set(self.V):
            missing_clients = set(self.V) - visited_clients
            print(f"    Constraint failed: missing clients {missing_clients}.")
            return False

        #print("  Constraints satisfied.")
        return True

    def select_parents(self, population, fitness_values, k=3):
        """
        トーナメント選択で親個体を選択。
        """
        selected = random.sample(range(len(population)), k)
        best = min(selected, key=lambda idx: fitness_values[idx])
        return population[best]

    def crossover(self, parent1, parent2):
        """
        交叉を行い、後処理で制約を満たすよう調整。
        """
        child1 = []
        child2 = []

        for route1, route2 in zip(parent1, parent2):
            # 要支援者ノードのみ抽出して部分順序交叉を実施
            clients1 = [node for node in route1 if node in self.V]
            clients2 = [node for node in route2 if node in self.V]

            # PMX交叉
            route_length = len(clients1)
            new_clients1 = [-1] * route_length
            new_clients2 = [-1] * route_length

            start, end = sorted(random.sample(range(route_length), 2))
            new_clients1[start:end] = clients1[start:end]
            new_clients2[start:end] = clients2[start:end]

            pos1 = end
            for i in range(len(clients2)):
                idx = (end + i) % len(clients2)
                if clients2[idx] not in new_clients1:
                    new_clients1[pos1 % len(new_clients1)] = clients2[idx]
                    pos1 += 1

            pos2 = end
            for i in range(len(clients1)):
                idx = (end + i) % len(clients1)
                if clients1[idx] not in new_clients2:
                    new_clients2[pos2 % len(new_clients2)] = clients1[idx]
                    pos2 += 1

            # 後処理で制約を満たすよう調整
            fixed_clients1 = self._fix_routes(new_clients1)
            fixed_clients2 = self._fix_routes(new_clients2)

            # 避難所を追加してルートを生成
            child1.append(self._add_shelters_to_route(fixed_clients1))
            child2.append(self._add_shelters_to_route(fixed_clients2))

        # 全体の後処理（重複削除、未訪問ノード追加、避難所再配置）
        fixed_child1 = self._validate_and_fix_routes(child1)
        fixed_child2 = self._validate_and_fix_routes(child2)

        return fixed_child1, fixed_child2

    def _fix_routes(self, clients):
        """
        重複する訪問を削除し、未訪問拠点を追加。
        :param clients: 要支援者ノードのリスト
        :return: 修正済みの要支援者ノードのリスト
        """
        # 訪問されたクライアントを一意にする
        unique_clients = []
        seen = set()
        for client in clients:
            if client not in seen:
                unique_clients.append(client)
                seen.add(client)

        # 未訪問のクライアントを探して追加
        missing_clients = set(self.V) - set(unique_clients)
        for missing_client in missing_clients:
            insert_pos = random.randint(0, len(unique_clients))  # ランダムな位置に挿入
            unique_clients.insert(insert_pos, missing_client)

        return unique_clients

    def _validate_and_fix_routes(self, individual):
        """
        個体全体を修正し、制約を満たすよう調整。
        :param individual: 個体（ルートリスト）
        :return: 修正済み個体
        """
        # 全ルートを1つのリストにまとめて検証
        all_clients = []
        for route in individual:
            all_clients.extend([node for node in route if node in self.V])

        # 重複削除と未訪問ノードの検出
        fixed_clients = self._fix_routes(all_clients)

        # 車両ごとに再分割
        num_vehicles = len(individual)
        split_size = len(fixed_clients) // num_vehicles
        split_routes = [
            fixed_clients[i * split_size:(i + 1) * split_size]
            for i in range(num_vehicles)
        ]

        # 最後の車両に余りを追加
        remainder = len(fixed_clients) % num_vehicles
        if remainder > 0:
            split_routes[-1].extend(fixed_clients[-remainder:])

        # 避難所を再配置
        fixed_individual = [self._add_shelters_to_route(route) for route in split_routes]
        return fixed_individual

    def _add_shelters_to_route(self, clients):
        """
        要支援者のリストにキャパを考慮して避難所を追加。
        """
        route = [0]  # デポから開始
        current_load = 0

        for client in clients:
            client_demand = self.d[client]

            if current_load + client_demand <= max(self.Q.values()):
                # キャパ内なら要支援者を追加
                route.append(client)
                current_load += client_demand
            else:
                # キャパ超過直前に最も近い避難所を追加
                last_client = route[-1]
                nearest_shelter = min(self.H, key=lambda shelter: self.c[last_client][shelter])
                route.append(nearest_shelter)
                route.append(client)
                current_load = client_demand  # キャパリセット

        # 最後に避難所を追加してデポに戻る
        if route[-1] not in self.H:
            last_client = route[-1]
            nearest_shelter = min(self.H, key=lambda shelter: self.c[last_client][shelter])
            route.append(nearest_shelter)
        route.append(0)  # デポに戻る

        return route

    def mutate(self, individual, mutation_rate=0.1):
        """
        要支援者だけを対象に突然変異を実施し、後から避難所を追加。
        :param individual: 個体 (リスト: 各車両のルート)
        :param mutation_rate: 突然変異率
        """
        for route_index, route in enumerate(individual):
            # 要支援者ノードを抽出
            clients = [node for node in route if node in self.V]

            # 短いリストでは突然変異をスキップ
            if len(clients) <= 1:
                continue

            # 突然変異を実行するかの判定
            if random.random() < mutation_rate:
                # ランダムな部分区間を逆順にする
                i, j = sorted(random.sample(range(len(clients)), 2))
                clients[i:j] = reversed(clients[i:j])

            # キャパシティを考慮して避難所を再配置
            individual[route_index] = self._add_shelters_to_route(clients)

    def create_next_generation(self, population):
        """
        次世代を構築する。
        :param population: 現世代の集団 (リスト)
        :return: 次世代の集団 (リスト)
        """
        # 現世代の適応度を計算
        fitness_values = [self.evaluate_individual(ind) for ind in population]

        # エリート個体を保存（最良個体）
        best_individual = copy.deepcopy(population[np.argmin(fitness_values)])
        best_fitness = min(fitness_values)

        # 次世代の初期化（エリート保存）
        next_generation = [best_individual]
        #print(f"エリート個体（適応度: {best_fitness}）が次世代に追加されました。")

        while len(next_generation) < self.population_size:
            # 親個体を選択
            parent1 = self.select_parents(population, fitness_values)
            parent2 = self.select_parents(population, fitness_values)

            # 交叉の実施
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            # 突然変異の適用
            self.mutate(child1, self.mutation_rate)
            self.mutate(child2, self.mutation_rate)

            # 子個体を次世代に追加（制約を満たす場合）
            if self.check_constraints(child1):
                next_generation.append(child1)
            else:
                print("Constraint failed for child1. Generating new individual.")
                next_generation.append(self.generate_initial_population()[0])  # 新規作成

            if len(next_generation) < self.population_size and self.check_constraints(child2):
                next_generation.append(child2)
            elif len(next_generation) < self.population_size:
                print("Constraint failed for child2. Generating new individual.")
                next_generation.append(self.generate_initial_population()[0])  # 新規作成

        # 次世代を返す（集団サイズを調整）
        return next_generation[:self.population_size]

    def run_genetic_algorithm(self, max_generations, output_csv='./result/genetic_results.csv',
                            best_individual_csv='./result/best_individual.csv',
                            log_file='./result/log.txt'):
        """
        遺伝アルゴリズムを実行し、結果を保存。
        :param max_generations: 世代数
        :param output_csv: 結果を保存するCSVファイル
        :param best_individual_csv: 最良個体のルートを保存するCSVファイル
        :param log_file: ログ出力ファイル
        :param theta: 搬送回数ペナルティ係数（目的関数中の重み）
        :return: 最良個体とその適合度
        """
        population = self.generate_initial_population(self.population_size)

        results = []
        best_overall_individual = None
        best_overall_fitness = float('inf')

        for generation in range(max_generations):
            fitness_values = [self.evaluate_individual(ind) for ind in population]

            # 現世代の最良個体を取得
            best_index = np.argmin(fitness_values)
            best_individual = population[best_index]
            best_fitness = fitness_values[best_index]
            mean_fitness = np.mean(fitness_values)
            std_fitness = np.std(fitness_values)

            # self.y を更新（評価を1回で済ませる）
            self.evaluate_individual(best_individual)
            total_y_m = sum(self.y.values())
            transport_cost = best_fitness - self.theta * total_y_m

            # 最良個体を更新
            if best_fitness < best_overall_fitness:
                best_overall_fitness = best_fitness
                best_overall_individual = best_individual

            # ログ出力
            with open(log_file, mode='a', encoding='utf-8') as log:
                log.write(
                    f"世代 {generation + 1}: "
                    f"最良適合度 = {best_fitness:.2f}, "
                    f"平均適合度 = {mean_fitness:.2f}, "
                    f"標準偏差 = {std_fitness:.2f}, "
                    f"搬送コスト = {transport_cost:.2f}, "
                    f"搬送回数 = {total_y_m}\n"
                )

            # 結果をリストに保存
            results.append([generation + 1, best_fitness, mean_fitness, std_fitness, transport_cost, total_y_m])

            # 次世代を生成
            population = self.create_next_generation(population)

        # CSV出力
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Generation', 'Best Fitness', 'Mean Fitness', 'Std Dev', 'Transport Cost', 'Total y_m'])
            writer.writerows(results)

        # 最良個体のルート保存
        with open(best_individual_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Vehicle', 'Route', 'Fitness'])
            for i, route in enumerate(best_overall_individual):
                writer.writerow([f"Vehicle {i + 1}", ' -> '.join(map(str, route)), ""])
            writer.writerow([])
            writer.writerow(['Best Fitness', best_overall_fitness])

        # 最終結果ログ出力
        with open(log_file, mode='a', encoding='utf-8') as log:
            log.write(f"\n全世代を通しての最良適合度: {best_overall_fitness}\n")
            log.write("最良個体:\n")
            for route in best_overall_individual:
                log.write(f"  ルート: {route}\n")

        self.plot_results(results, output_csv.replace('.csv', '.png'))
        self.calculate_times(best_overall_individual)
        self.plot_histograms()

        return best_overall_individual, best_overall_fitness

    '''
    def run_genetic_algorithm(self, max_generations, output_csv='./result/genetic_results.csv', best_individual_csv='./result/best_individual.csv', log_file='./result/log.txt'):
        """
        遺伝アルゴリズムを実行し、結果を保存。
        :param max_generations: 世代数
        :param output_csv: 結果を保存するCSVファイルの名前
        :return: 全世代を通じての最良個体とその適合度
        """
        # 初期集団生成
        population = self.generate_initial_population(self.population_size)

        # CSVデータの準備
        results = []
        best_overall_individual = None
        best_overall_fitness = float('inf')

        for generation in range(max_generations):
            # 適応度の計算
            fitness_values = [self.evaluate_individual(ind) for ind in population]

            # 現世代のデータ
            best_fitness = min(fitness_values)
            mean_fitness = np.mean(fitness_values)
            std_fitness = np.std(fitness_values)

            # 最良個体の保存
            best_individual = population[np.argmin(fitness_values)]
            if best_fitness < best_overall_fitness:
                best_overall_fitness = best_fitness
                best_overall_individual = best_individual

            # 結果を記録
            results.append([generation + 1, best_fitness, mean_fitness, std_fitness])

            # 次世代を作成
            population = self.create_next_generation(population)

            # 現世代の進捗をログファイルに記録
            with open(log_file, mode='a', encoding='utf-8') as log:
                log.write(f"世代 {generation + 1}: 最良適合度 = {best_fitness}, 平均適合度 = {mean_fitness}, 標準偏差 = {std_fitness}\n")
            # 現世代の進捗を表示
            #print(f"世代 {generation + 1}: 最良適合度 = {best_fitness}, 平均適合度 = {mean_fitness}, 標準偏差 = {std_fitness}")

        # 結果をCSVに保存
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Generation', 'Best Fitness', 'Mean Fitness', 'Std Dev'])
            writer.writerows(results)

        # 最良個体をCSVに保存
        with open(best_individual_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Vehicle', 'Route', 'Fitness'])
            for i, route in enumerate(best_overall_individual):
                writer.writerow([f"Vehicle {i + 1}", ' -> '.join(map(str, route)), ""])
            writer.writerow([])
            writer.writerow(['Best Fitness', best_overall_fitness])

        # 全世代を通しての結果をログに記録
        with open(log_file, mode='a', encoding='utf-8') as log:
            log.write(f"\n全世代を通しての最良適合度: {best_overall_fitness}\n")
            log.write("最良個体:\n")
            for route in best_overall_individual:
                log.write(f"  ルート: {route}\n")
            
        # グラフの作成
        self.plot_results(results, output_csv.replace('.csv', '.png'))

        # 最良個体が確定した後、待ち時間と搬送時間を計算
        self.calculate_times(best_overall_individual)

        # ヒストグラムを作成
        self.plot_histograms()

        return best_overall_individual, best_overall_fitness
    '''
        
    def plot_results(self, results, output_image='./result/genetic_results.png'):
        """
        遺伝アルゴリズムの結果をプロット。
        :param results: 世代ごとの結果データ (リスト)
        :param output_image: グラフを保存するファイル名
        """
        generations = [row[0] for row in results]
        best_fitness = [row[1] for row in results]
        mean_fitness = [row[2] for row in results]

        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, label="Best Fitness", marker='o')
        plt.plot(generations, mean_fitness, label="Mean Fitness", marker='x')
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Genetic Algorithm Progress")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_image)
        #plt.show()
        print(f"結果のグラフを保存しました: {output_image}")

    def save_vehicle_statistics(self, best_individual, output_dir='./result/'):
        """
        車両ごとの統計（移動距離、搬送人数、搬送回数）を計算し、グラフとCSVで保存。
        :param best_individual: 最良個体
        :param output_dir: 出力先のディレクトリ
        """
        os.makedirs(output_dir, exist_ok=True)

        vehicle_costs = []  # 各車両の移動距離
        vehicle_loads = []  # 各車両の搬送人数
        vehicle_shelter_visits = []  # 各車両の避難所訪問回数

        for route in best_individual:
            route_cost = 0
            total_load = 0
            shelter_visits = 0
            previous_node = 0  # デポから出発

            for node in route:
                if node in self.H:
                    shelter_visits += 1
                if node in self.V:
                    total_load += self.d[node]

                route_cost += self.c[previous_node][node]
                previous_node = node

            vehicle_costs.append(route_cost)
            vehicle_loads.append(total_load)
            vehicle_shelter_visits.append(shelter_visits)

        # CSVに保存
        csv_file = os.path.join(output_dir, 'vehicle_statistics.csv')
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Vehicle', 'Cost (Distance)', 'Load (People)', 'Shelter Visits'])
            for i, (cost, load, visits) in enumerate(zip(vehicle_costs, vehicle_loads, vehicle_shelter_visits), start=1):
                writer.writerow([f"Vehicle {i}", cost, load, visits])

        print(f"車両統計データをCSVに保存しました: {csv_file}")

        # グラフ作成
        self.plot_vehicle_statistics(vehicle_costs, 'Cost (Distance)', os.path.join(output_dir, 'vehicle_costs.png'))
        self.plot_vehicle_statistics(vehicle_loads, 'Load (People)', os.path.join(output_dir, 'vehicle_loads.png'))
        self.plot_vehicle_statistics(vehicle_shelter_visits, 'Shelter Visits', os.path.join(output_dir, 'vehicle_shelter_visits.png'))

    def plot_vehicle_statistics(self, data, ylabel, output_image):
        """
        統計データの棒グラフを作成して保存。
        :param data: 統計データ（リスト）
        :param ylabel: Y軸のラベル
        :param output_image: 保存先の画像ファイルパス
        """
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(data) + 1), data)
        plt.xlabel('Vehicle')
        plt.ylabel(ylabel)
        plt.title(f'Vehicle {ylabel}')
        plt.grid(axis='y')
        plt.xticks(range(1, len(data) + 1), [f"V{i}" for i in range(1, len(data) + 1)])
        plt.savefig(output_image)
        plt.close()
        print(f"グラフを保存しました: {output_image}")

    def visualize_routes(self, best_individual, nodes, output_dir='./result/'):
        """
        車両ごとの搬送ルートを可視化し、画像として保存。
        :param best_individual: 最良個体
        :param nodes: ノード情報（辞書リスト）
        :param output_dir: 出力先のディレクトリ
        """
        os.makedirs(output_dir, exist_ok=True)

        # ノード座標を取得
        node_positions = {node["id"]: (node["x"], node["y"]) for node in nodes}

        # ノードの描画設定（色と形）
        def plot_node(node, x, y):
            if node["type"] == "city_hall":
                plt.scatter(x, y, color="blue", marker="s", s=100)
            elif node["type"] == "shelter":
                plt.scatter(x, y, color="green", marker="^", s=100)
            elif node["type"] == "client":
                cmap = plt.cm.viridis
                norm = plt.Normalize(1, max(n["demand"] for n in nodes if n["type"] == "client"))
                color = cmap(norm(node["demand"]))
                plt.scatter(x, y, color=color, marker="o", s=100)

        # 矢印描画設定
        arrow_params = {
            "head_width": 0.5,  # 矢印の先端の幅
            "head_length": 0.5,  # 矢印の先端の長さ
            "width": 0.002,  # 矢印の線の太さ
            "length_includes_head": True,
            "alpha": 0.7,
        }

        # 各車両のルートを個別に可視化
        for vehicle_id, route in enumerate(best_individual, start=1):
            plt.figure(figsize=(8, 8))

            # ノードと矢印を描画
            for i, node_id in enumerate(route[:-1]):
                next_node_id = route[i + 1]
                x, y = node_positions[node_id]
                next_x, next_y = node_positions[next_node_id]
                
                # ノードのプロット
                node = next(n for n in nodes if n["id"] == node_id)
                plot_node(node, x, y)

                # 矢印でルートを表示
                plt.arrow(x, y, next_x - x, next_y - y, color="black", **arrow_params)

            # 最後のノードをプロット
            last_node = next(n for n in nodes if n["id"] == route[-1])
            last_x, last_y = node_positions[route[-1]]
            plot_node(last_node, last_x, last_y)

            plt.title(f"Vehicle {vehicle_id} Route")
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'vehicle_{vehicle_id}_route.png'))
            plt.close()
            print(f"車両 {vehicle_id} のルートを保存しました: {output_dir}/vehicle_{vehicle_id}_route.png")

        # 全車両のルートを1つの図にまとめて可視化
        plt.figure(figsize=(10, 10))
        color_map = plt.cm.get_cmap('tab10', len(best_individual))  # 車両数に応じて色を割り当て

        legend_handles = []  # 凡例用

        for vehicle_id, route in enumerate(best_individual, start=1):
            vehicle_color = color_map(vehicle_id - 1)  # 車両ごとに色を割り当て

            # ノードと矢印を描画
            for i, node_id in enumerate(route[:-1]):
                next_node_id = route[i + 1]
                x, y = node_positions[node_id]
                next_x, next_y = node_positions[next_node_id]
                
                # 矢印でルートを表示
                arrow_params["color"] = vehicle_color  # 矢印の色を更新
                plt.arrow(x, y, next_x - x, next_y - y, **arrow_params)

            # ノードのプロット
            for node_id in route:
                node = next(n for n in nodes if n["id"] == node_id)
                x, y = node_positions[node_id]
                plot_node(node, x, y)

            # 凡例に車両の色を追加
            legend_handles.append(plt.Line2D([0], [0], color=vehicle_color, lw=2, label=f"Vehicle {vehicle_id}"))

        # 凡例を追加（車両ごとの色のみ表示）
        plt.legend(handles=legend_handles, title="Vehicles", loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=10)
        plt.title("All Vehicle Routes")
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'all_vehicle_routes.png'))
        plt.close()
        print(f"全車両のルートを保存しました: {output_dir}/all_vehicle_routes.png")

    def calculate_times(self, best_individual):
        """
        各要支援者の車両到着までの待ち時間と搬送時間を計算。
        :param best_individual: 最良個体
        """
        # 初期化
        self.waiting_times = {client: None for client in self.V}
        self.transport_times = {client: 0 for client in self.V}

        for route in best_individual:
            current_time = 0
            for i in range(len(route) - 1):
                current_node = route[i]
                next_node = route[i + 1]
                travel_time = self.c[current_node][next_node]

                if next_node in self.V:  # 次のノードがクライアントの場合
                    # 車両が到着するまでの時間を待ち時間に記録
                    if self.waiting_times[next_node] is None:
                        self.waiting_times[next_node] = current_time
                    # 搬送時間を記録
                    self.transport_times[next_node] += travel_time

                current_time += travel_time

    def plot_histograms(self, output_dir='./result/'):
        """
        待ち時間、搬送時間、合計時間のヒストグラムを作成し保存。
        :param output_dir: 出力先ディレクトリ
        """
        os.makedirs(output_dir, exist_ok=True)

        waiting_times = [time for time in self.waiting_times.values() if time is not None]
        transport_times = [time for time in self.transport_times.values()]
        total_times = [waiting + transport for waiting, transport in zip(waiting_times, transport_times)]

        plt.figure(figsize=(10, 6))
        plt.hist(waiting_times, bins=20, alpha=0.7, label='Waiting Times')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title('Histogram of Waiting Times')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'waiting_times_histogram.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(transport_times, bins=20, alpha=0.7, label='Transport Times', color='orange')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title('Histogram of Transport Times')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'transport_times_histogram.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(total_times, bins=20, alpha=0.7, label='Total Times', color='green')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title('Histogram of Total Times (Waiting + Transport)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'total_times_histogram.png'))
        plt.close()

    def visualize_routes_3d(self, best_individual, nodes, output_dir='./result/'):
        """
        車両ごとの搬送ルートを可視化し、3D画像として保存。
        :param best_individual: 最良個体
        :param nodes: ノード情報（辞書リスト）
        :param output_dir: 出力先のディレクトリ
        """
        os.makedirs(output_dir, exist_ok=True)

        # ノード座標を取得
        node_positions = {node["id"]: (node["x"], node["y"], node["z"]) for node in nodes}

        # ノードの描画設定（色と形）
        def plot_node(ax, node, x, y, z):
            if node["type"] == "city_hall":
                ax.scatter(x, y, z, color="blue", marker="s", s=100, label="City Hall")
            elif node["type"] == "shelter":
                ax.scatter(x, y, z, color="green", marker="^", s=100, label="Shelter")
            elif node["type"] == "client":
                cmap = plt.cm.viridis
                norm = plt.Normalize(1, max(n["demand"] for n in nodes if n["type"] == "client"))
                color = cmap(norm(node["demand"]))
                ax.scatter(x, y, z, color=color, marker="o", s=100, label="Client")

        # 各車両のルートを個別に可視化
        for vehicle_id, route in enumerate(best_individual, start=1):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')

            # ノードと矢印を描画
            for i, node_id in enumerate(route[:-1]):
                next_node_id = route[i + 1]
                x, y, z = node_positions[node_id]
                next_x, next_y, next_z = node_positions[next_node_id]

                # ノードのプロット
                node = next(n for n in nodes if n["id"] == node_id)
                plot_node(ax, node, x, y, z)

                # 矢印でルートを表示
                ax.plot([x, next_x], [y, next_y], [z, next_z], color="black", alpha=0.7)

            # 最後のノードをプロット
            last_node = next(n for n in nodes if n["id"] == route[-1])
            last_x, last_y, last_z = node_positions[route[-1]]
            plot_node(ax, last_node, last_x, last_y, last_z)

            ax.set_title(f"Vehicle {vehicle_id} Route")
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_zlabel('Z Coordinate')
            ax.grid(True)
            plt.savefig(os.path.join(output_dir, f'vehicle_{vehicle_id}_route_3d.png'))
            plt.close()
            print(f"車両 {vehicle_id} のルートを保存しました: {output_dir}/vehicle_{vehicle_id}_route_3d.png")

    '''
    def plot_elevation_changes(best_individual, nodes, cost_matrix, output_file='./result/elevation_changes.png'):
        """
        各車両のルートにおける登った高さと降った高さを積み上げグラフとして保存。
        :param best_individual: 最良個体 (車両ごとのルート)
        :param nodes: ノード情報（辞書リスト）
        :param cost_matrix: コスト行列（numpy.array, 3次元の距離データを含む）
        :param output_file: グラフを保存するファイル名
        """
        
        print("デバッグ: best_individual の最初の要素:", best_individual[0] if best_individual else "空")
        
        # ノードの z 座標を取得し、float型に変換
        node_z_positions = {}
        
        # best_individualがリストのリストになっていることを確認
        if not isinstance(best_individual, list) or not all(isinstance(route, list) for route in best_individual):
            raise TypeError("best_individualはリストのリストである必要があります")
        
        # nodesから直接z座標を取得
        for node in nodes:
            if not isinstance(node, dict):
                print(f"警告: 不正なノード形式です: {node}")
                continue
                
            if 'id' not in node or 'z' not in node:
                print(f"警告: 必要なキーが見つかりません: {node}")
                continue
                
            try:
                node_z_positions[node['id']] = float(node['z'])
            except (ValueError, TypeError) as e:
                print(f"警告: ノード {node.get('id', 'unknown')} の z 座標変換に失敗: {e}")
                continue

        # 登りと下りの高さを計算
        vehicle_ascents = []
        vehicle_descents = []

        for route in best_individual:
            ascent = 0
            descent = 0
            
            if len(route) < 2:  # ルートが短すぎる場合はスキップ
                vehicle_ascents.append(0)
                vehicle_descents.append(0)
                continue

            for i in range(len(route) - 1):
                current_node = route[i]
                next_node = route[i + 1]

                try:
                    current_z = node_z_positions[current_node]
                    next_z = node_z_positions[next_node]

                    height_change = next_z - current_z
                    if height_change > 0:
                        ascent += height_change
                    elif height_change < 0:
                        descent += abs(height_change)
                except KeyError as e:
                    print(f"警告: ノード {current_node} または {next_node} の高度データが見つかりません")
                    continue

            vehicle_ascents.append(ascent)
            vehicle_descents.append(descent)

        if not vehicle_ascents or not vehicle_descents:
            raise ValueError("有効な高度変化データが計算できませんでした")

        # グラフの作成
        vehicle_indices = list(range(1, len(best_individual) + 1))

        plt.figure(figsize=(10, 6))
        plt.bar(vehicle_indices, vehicle_ascents, label="Ascent", color="orange")
        plt.bar(vehicle_indices, vehicle_descents, bottom=vehicle_ascents, label="Descent", color="blue")
        plt.xlabel("Vehicle Number")
        plt.ylabel("Elevation Change")
        plt.title("Elevation Changes by Vehicle")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.savefig(output_file)
        plt.close()
        print(f"積み上げグラフを保存しました: {output_file}")
    '''
