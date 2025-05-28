import geopandas as gpd
import folium
import pandas as pd
import random
import requests
import osmnx as ox
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from shapely.geometry import Point
import csv
import numpy as np
from shapely.geometry import LineString
from geopy.distance import geodesic
import time
import os
import xml.etree.ElementTree as ET
import scipy.ndimage
import matplotlib.colors as mcolors
import statistics

class CVRP_Geography:
    def __init__(self, file_path, layer_name="town"):
        """
        初期化メソッド
        :param file_path: TopoJSONファイルのパス
        :param layer_name: 読み込むレイヤー名（デフォルトは "town"）
        """
        self.file_path = file_path
        self.layer_name = layer_name
        self.gdf = None
        self.center_lat = None
        self.center_lon = None

    def load_data(self):
        """ TopoJSONファイルを読み込む """
        try:
            self.gdf = gpd.read_file(self.file_path, layer=self.layer_name)

            # CRS（座標系）の設定（WGS84）
            if self.gdf.crs is None:
                self.gdf.set_crs(epsg=4326, inplace=True)

            # 地図の中心座標を取得
            self.center_lat = self.gdf.geometry.centroid.y.mean()
            self.center_lon = self.gdf.geometry.centroid.x.mean()

            print(f"成功: データの読み込みが完了しました。({self.layer_name})")
        except Exception as e:
            print(f"エラー: データの読み込みに失敗しました - {e}")

    def save_csv(self, output_csv_path):
        """
        CSVファイルとして保存
        :param output_csv_path: 保存するCSVファイルのパス
        """
        try:
            columns_to_keep = ["PREF_NAME", "CITY_NAME", "S_NAME", "AREA", "JINKO", "SETAI", "X_CODE", "Y_CODE", "SUPPORT_NEEDS"]
            if all(col in self.gdf.columns for col in columns_to_keep):
                self.gdf[columns_to_keep].to_csv(output_csv_path, index=False, encoding='utf-8-sig')
                print(f"CSVファイルが {output_csv_path} に保存されました。")
            else:
                print("必要なカラムがデータ内に見つかりませんでした。")
        except Exception as e:
            print(f"エラー: CSVファイルの保存に失敗しました - {e}")

    def generate_map(self, output_html_path):
        """
        Foliumを使用して地図を生成し、HTMLファイルとして保存
        :param output_html_path: 保存するHTMLファイルのパス
        """
        if self.gdf is None:
            print("エラー: データをロードしてください。")
            return

        try:
            m = folium.Map(location=[self.center_lat, self.center_lon], zoom_start=13, tiles="cartodbpositron")

            # 各地区にマーカーを設置
            for _, row in self.gdf.iterrows():
                s_name = row["S_NAME"]      # 町名
                area = row["AREA"]          # 面積
                jinko = row["JINKO"]         # 人口
                x_code = row["X_CODE"]       # 経度
                y_code = row["Y_CODE"]       # 緯度
                support_needs = row["SUPPORT_NEEDS"] # 要支援者人数

                # ポップアップに表示する内容
                popup_text = f"""
                <b>町名:</b> {s_name}<br>
                <b>面積:</b> {area:.2f} m²<br>
                <b>人口:</b> {jinko} 人 <br>
                <b>要支援者:</b> {support_needs} 人
                """

                # マーカーの追加
                folium.Marker(
                    location=[y_code, x_code],
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=folium.Icon(color="blue", icon="info-sign")
                ).add_to(m)

            # GeoJSONレイヤーの追加
            folium.GeoJson(
                self.gdf.to_json(),
                style_function=lambda x: {'color': 'red', 'weight': 2, 'fillOpacity': 0.3}
            ).add_to(m)

            # HTMLファイルとして保存
            m.save(output_html_path)
            print(f"地図が {output_html_path} に保存されました。")
        except Exception as e:
            print(f"エラー: 地図の生成に失敗しました - {e}")
    
    def assign_support_needs(self, total_support_needs):
        """
        総要支援者人数を地区の人口比率に基づいて割り振る
        :param total_support_needs: 全体の要支援者の総数
        """
        if "JINKO" not in self.gdf.columns:
            print("エラー: 人口（JINKO）データが存在しません。")
            return

        # 人口の合計を取得
        total_population = self.gdf["JINKO"].sum()

        # 各地区に要支援者を人口比で割り当て
        self.gdf["SUPPORT_NEEDS"] = (self.gdf["JINKO"] / total_population * total_support_needs).round().astype(int)

        print("要支援者人数の割り当てが完了しました。")

    def load_shelters(self, csv_file, city_office_info=None):
        """
        避難所データのCSVファイルを読み込み、データフレームとして保存
        :param csv_file: 避難所情報を含むCSVファイルのパス
        """
        try:
            self.shelters_df = pd.read_csv(csv_file, encoding='shift_jis')

            # "一時避難所" 以外の避難所をフィルタリング
            self.shelters_df = self.shelters_df[self.shelters_df['備考'] != '一次避難所']
            # 市役所の情報が提供された場合に追加
            if city_office_info:
                city_office_df = pd.DataFrame([city_office_info])
                self.shelters_df = pd.concat([self.shelters_df, city_office_df], ignore_index=True)

            print(f"避難所データが正常にロードされました。対象避難所数: {len(self.shelters_df)}")
        except Exception as e:
            print(f"エラー: 避難所データの読み込みに失敗しました - {e}")

    def plot_shelters(self, output_html_path):
        """
        一時避難所以外の避難所を地図上に表示し、名称と想定収容人数を表示する
        :param output_html_path: 保存するHTMLファイルのパス
        """
        if self.shelters_df is None or self.shelters_df.empty:
            print("エラー: 避難所データがロードされていません。")
            return

        try:
            # 欠損値を適切な値に置き換え（収容人数が不明の場合は0にする）
            self.shelters_df["想定収容人数"] = self.shelters_df["想定収容人数"].fillna(0).astype(int)
            m = folium.Map(location=[self.center_lat, self.center_lon], zoom_start=13, tiles="cartodbpositron")

            for _, row in self.shelters_df.iterrows():
                name = row["名称"]
                capacity = row["想定収容人数"]
                lat = row["緯度"]
                lon = row["経度"]
                category = row["備考"]

                # 市役所は赤色アイコン、他の避難所は緑色アイコン
                if category == "市役所":
                    icon_color = "red"
                    icon_type = "info-sign"
                else:
                    icon_color = "green"
                    icon_type = "home"

                popup_text = f"<b>避難所:</b> {name}<br><b>想定収容人数:</b> {int(capacity)} 人<br><b>備考:</b> {category}"

                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=folium.Icon(color=icon_color, icon=icon_type)
                ).add_to(m)

            m.save(output_html_path)
            print(f"避難所の地図が {output_html_path} に保存されました。")
        except Exception as e:
            print(f"エラー: 避難所の地図の生成に失敗しました - {e}")

    def get_gsi_elevation(self, lat, lon):
        """ 国土地理院APIを利用して標高を取得し、無効な値を処理 """
        url = f"https://cyberjapandata2.gsi.go.jp/general/dem/scripts/getelevation.php?lon={lon}&lat={lat}&outtype=JSON"
        try:
            response = requests.get(url)
            response.raise_for_status()  # HTTPリクエストのエラーチェック
            data = response.json()

            # 取得したレスポンスを表示（デバッグ用）
            print(f"取得データ（{lat}, {lon}）: {data}")

            if "elevation" in data:
                elevation = data["elevation"]

                # 標高データが '-----' などの無効な値かどうかをチェック
                if elevation == "-----" or elevation is None:
                    print(f"警告: 標高データが無効 ({lat}, {lon})")
                    return None  # または return 0 に変更可

                return round(float(elevation), 2)

            else:
                print(f"警告: 標高データが存在しない ({lat}, {lon})")
                return None

        except requests.exceptions.RequestException as e:
            print(f"エラー: 標高データの取得に失敗しました ({lat}, {lon}) - {e}")
            return None
        except ValueError as ve:
            print(f"エラー: 標高データの変換に失敗しました（{lat}, {lon}）- {ve}")
            return None

    def assign_random_support_needs(self, output_csv_path, map_output_html):
        """ 各地区の要支援者をランダムに割り当て、位置情報と標高データを設定して保存 """
        if self.gdf is None:
            print("エラー: データをロードしてください。")
            return

        assigned_data = []

        # 1. 市役所データを先に追加
        id_counter = 0  # idのカウンタを0から開始
        for _, row in self.shelters_df.iterrows():
            if row['備考'] == '市役所':
                entry_type = 'city_hall'
                elevation = self.get_gsi_elevation(row['緯度'], row['経度'])
                assigned_data.append({
                    'id': id_counter,  # idを設定
                    'type': entry_type,
                    'x': row['経度'],
                    'y': row['緯度'],
                    'z': elevation,  # elevationをzに変更
                    'demand': 0,
                    'priority': '-',
                    'name': row['名称'],
                    'capacity': row.get('想定収容人数', 0),
                    'remarks': row.get('備考', '')
                })
                id_counter += 1  # idをインクリメント

        # 2. 一般の避難所データを追加
        for _, row in self.shelters_df.iterrows():
            if row['備考'] != '市役所':
                entry_type = 'shelter'
                elevation = self.get_gsi_elevation(row['緯度'], row['経度'])
                assigned_data.append({
                    'id': id_counter,  # idを設定
                    'type': entry_type,
                    'x': row['経度'],
                    'y': row['緯度'],
                    'z': elevation,  # elevationをzに変更
                    'demand': 0,
                    'priority': '-',
                    'name': row['名称'],
                    'capacity': row.get('想定収容人数', 0),
                    'remarks': row.get('備考', '')
                })
                id_counter += 1  # idをインクリメント

        # 3. 要配慮者データを追加（修正版）
        for _, row in self.gdf.iterrows():
            support_needs = row['SUPPORT_NEEDS']
            polygon = row['geometry']  # シェープ情報
            for i in range(support_needs):
                while True:
                    # ポリゴン内にランダムポイントを生成
                    minx, miny, maxx, maxy = polygon.bounds
                    lon = random.uniform(minx, maxx)
                    lat = random.uniform(miny, maxy)
                    random_point = Point(lon, lat)

                    # ポイントがポリゴン内にあるかを確認
                    if not polygon.contains(random_point):
                        continue

                    # 標高データを取得して有効性を確認
                    elevation = self.get_gsi_elevation(lat, lon)
                    if elevation is not None:  # 標高が有効であればループ終了
                        break

                # 有効な座標と標高データで追加
                assigned_data.append({
                    'id': id_counter,
                    'type': 'client',
                    'x': lon,
                    'y': lat,
                    'z': elevation,
                    'demand': random.choice([1, 2]),
                    'priority': random.randint(1, 5),
                    'name': f"{row['S_NAME']}_{i+1}",
                    'capacity': 0,
                    'remarks': ''
                })
                id_counter += 1  # idをインクリメント

        # データをDataFrameに変換してCSV保存
        df_assigned = pd.DataFrame(assigned_data)
        df_assigned.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"要支援者データと避難所情報が {output_csv_path} に保存されました。")

        # 地図の作成
        m = folium.Map(location=[self.center_lat, self.center_lon], zoom_start=13, tiles="cartodbpositron")
        for entry in assigned_data:
            if entry['type'] == 'client':
                popup_text = f"<b>名前:</b> {entry['name']}<br><b>人数:</b> {entry['demand']} 人<br><b>優先度:</b> {entry['priority']}<br><b>標高:</b> {entry['z']} m"
                color, icon = "red", "user"
            else:
                popup_text = f"<b>避難所:</b> {entry['name']}<br><b>想定収容人数:</b> {entry['capacity']} 人<br><b>標高:</b> {entry['z']} m<br><b>備考:</b> {entry['remarks']}"
                color, icon = ("blue", "info-sign") if entry['type'] == 'city_hall' else ("green", "home")

            folium.Marker(
                location=[entry['y'], entry['x']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color=color, icon=icon)
            ).add_to(m)

        m.save(map_output_html)
        print(f"地図が {map_output_html} に保存されました。")
            
            
    def plot_colored_roads(self, graphml_file, output_filepath):
        try:
            G = ox.load_graphml(filepath=graphml_file)
            print("ネットワークデータを読み込みました。")

            road_colors = {
                "trunk": "red",
                "primary": "blue",
                "secondary": "green",
                "tertiary": "orange",
            }

            fig, ax = plt.subplots(figsize=(12, 12))

            edge_groups = {}  # color: list of (xs, ys)
            for u, v, k, data in G.edges(keys=True, data=True):
                highway_type = data.get("highway")
                if isinstance(highway_type, list):
                    highway_type = highway_type[0]
                highway_type = highway_type or "other"
                color = road_colors.get(highway_type, "gray")

                if "geometry" in data:
                    xs, ys = data["geometry"].xy
                else:
                    # fallback to straight line if no geometry
                    x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
                    x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
                    xs, ys = [x1, x2], [y1, y2]

                edge_groups.setdefault(color, []).append((xs, ys))

            # 描画
            for color, lines in edge_groups.items():
                for xs, ys in lines:
                    ax.plot(xs, ys, color=color, linewidth=2)

            # 凡例
            legend_labels = {
                "trunk": "Trunk",
                "primary": "Primary",
                "secondary": "Secondary",
                "tertiary": "Tertiary",
                "gray": "Other"
            }
            used_colors = set(edge_groups.keys())
            handles = [plt.Line2D([0], [0], color=c, lw=2, label=legend_labels.get(k, "Other"))
                    for k, c in road_colors.items() if c in used_colors]
            if "gray" in used_colors:
                handles.append(plt.Line2D([0], [0], color="gray", lw=2, label="Other"))

            ax.legend(handles=handles, title="Road Types", loc="upper left")
            plt.title("Color-coded Road Types in Omaezaki City", fontsize=16)
            plt.axis("off")
            plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
            print(f"地図を保存しました: {output_filepath}")

        except Exception as e:
            print(f"エラー: 道路地図の生成に失敗しました - {e}")

    def calculate_travel_times(self, graphml_file, nodes_csv, output_csv, output_matrix_csv, penalty=100000):
        """
        全てのノード間の移動時間を計算し、結果をCSVと行列形式で保存
        :param graphml_file: GraphMLファイルのパス
        :param nodes_csv: ノード情報を含むCSVファイルのパス
        :param output_csv: 結果を保存するCSVファイルのパス
        :param output_matrix_csv: 行列形式の結果を保存するCSVファイルのパス
        """
        try:
            # グラフを読み込み
            print("ネットワークデータを読み込んでいます...")
            G = ox.load_graphml(filepath=graphml_file)
            print("ネットワークデータを読み込みました。")

            # エッジに移動時間（weight）を追加
            for u, v, k, data in G.edges(data=True, keys=True):
                # エッジの長さと制限速度を取得
                length = data.get("length", 1)  # 距離 (m)
                speed = data.get("maxspeed", 30)  # 制限速度 (km/h)

                # maxspeedがリストの場合、最初の値を使用
                if isinstance(speed, list):
                    speed = speed[0]

                # 制限速度がない場合はデフォルト値を使用
                try:
                    speed = float(speed)
                except (TypeError, ValueError):
                    speed = 30  # デフォルトの制限速度 (km/h)

                # 移動時間（秒）を計算してエッジに追加
                travel_time = length / (speed * 1000 / 3600)  # 秒単位の時間
                data["weight"] = travel_time

            # ノードデータを読み込み
            print("ノードデータを読み込んでいます...")
            nodes_df = pd.read_csv(nodes_csv)
            print("ノードデータを読み込みました。")

            # 結果を格納するリスト
            travel_times = []

            # 全てのノード間の組み合わせを取得
            node_pairs = combinations(nodes_df["id"], 2)

            # ノードIDと座標、名前をマッピング
            id_to_coords = nodes_df.set_index("id")[["x", "y", "name"]].to_dict(orient="index")

            # 全ての組み合わせで移動時間を計算
            for source_id, target_id in node_pairs:
                try:
                    source_coords = id_to_coords[source_id]
                    target_coords = id_to_coords[target_id]

                    # 起点と終点のノードを取得
                    source_node = ox.distance.nearest_nodes(G, X=source_coords["x"], Y=source_coords["y"])
                    target_node = ox.distance.nearest_nodes(G, X=target_coords["x"], Y=target_coords["y"])

                    # 最短経路を計算
                    route = nx.shortest_path(G, source_node, target_node, weight="weight")
                    travel_time = nx.shortest_path_length(G, source_node, target_node, weight="weight")

                    # 結果をリストに保存
                    travel_times.append({
                        "source_id": source_id,
                        "target_id": target_id,
                        "travel_time": travel_time,
                    })

                    # 計算状況をターミナルに出力
                    print(f"計算中: 拠点 {source_id} -> 拠点 {target_id} | 移動時間: {travel_time:.2f} 秒")

                except nx.NetworkXNoPath:
                    print(f"ルートが見つかりませんでした: {source_id} -> {target_id}")
                    travel_times.append({
                        "source_id": source_id,
                        "target_id": target_id,
                        "travel_time": penalty,#ルートが見つからない場合は、移動時間を大きくする
                    })

            # 結果をデータフレームに変換
            travel_times_df = pd.DataFrame(travel_times)
            all_nodes = list(range(len(nodes_df)))
            # 行列形式に変換して保存
            travel_times_df = travel_times_df.set_index(["source_id", "target_id"]).reindex(
                pd.MultiIndex.from_product([all_nodes, all_nodes], names=["source_id", "target_id"]),
                fill_value=0
            ).reset_index()

            # 結果をCSVに保存
            travel_times_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"移動時間データを {output_csv} に保存しました。")

            travel_time_matrix = travel_times_df.pivot(index="source_id", columns="target_id", values="travel_time").fillna(0)
            print(f"Travel time matrix created with shape: {travel_time_matrix.shape}")
            travel_time_matrix.to_csv(output_matrix_csv, index=True, encoding="utf-8-sig")
            print(f"行列形式の移動時間データを {output_matrix_csv} に保存しました。")

        except Exception as e:
            print(f"エラー: {e}")
            
            
    def calculate_travel_times2(self, graphml_file, nodes_csv, output_csv, output_matrix_csv, penalty=100000):
        """
        全てのノード間の移動時間を計算し、結果をCSVと行列形式で保存 + ルートを返す
        """
        try:
            G = ox.load_graphml(filepath=graphml_file)
            print("ネットワークデータを読み込みました。")

            for u, v, k, data in G.edges(data=True, keys=True):
                length = data.get("length", 1)
                speed = data.get("maxspeed", 30)
                if isinstance(speed, list):
                    speed = speed[0]
                try:
                    speed = float(speed)
                except (TypeError, ValueError):
                    speed = 30
                travel_time = length / (speed * 1000 / 3600)
                data["weight"] = travel_time

            nodes_df = pd.read_csv(nodes_csv)
            travel_times = []
            routes = []
            node_pairs = combinations(nodes_df["id"], 2)
            id_to_coords = nodes_df.set_index("id")[["x", "y", "name"]].to_dict(orient="index")

            for source_id, target_id in node_pairs:
                try:
                    source_coords = id_to_coords[source_id]
                    target_coords = id_to_coords[target_id]
                    source_node = ox.distance.nearest_nodes(G, X=source_coords["x"], Y=source_coords["y"])
                    target_node = ox.distance.nearest_nodes(G, X=target_coords["x"], Y=target_coords["y"])

                    route = nx.shortest_path(G, source_node, target_node, weight="weight")
                    travel_time = nx.shortest_path_length(G, source_node, target_node, weight="weight")

                    travel_times.append({
                        "source_id": source_id,
                        "target_id": target_id,
                        "travel_time": travel_time,
                    })
                    routes.append({
                        "source_id": source_id,
                        "target_id": target_id,
                        "route": route
                    })
                    print(f"計算中: {source_id} -> {target_id} | {travel_time:.2f} 秒")
                except nx.NetworkXNoPath:
                    print(f"ルートなし: {source_id} -> {target_id}")
                    travel_times.append({
                        "source_id": source_id,
                        "target_id": target_id,
                        "travel_time": penalty
                    })

            travel_times_df = pd.DataFrame(travel_times)
            all_nodes = list(range(len(nodes_df)))
            travel_times_df = travel_times_df.set_index(["source_id", "target_id"]).reindex(
                pd.MultiIndex.from_product([all_nodes, all_nodes], names=["source_id", "target_id"]),
                fill_value=0
            ).reset_index()

            travel_times_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"移動時間データを {output_csv} に保存しました。")

            travel_time_matrix = travel_times_df.pivot(index="source_id", columns="target_id", values="travel_time").fillna(0)
            travel_time_matrix.to_csv(output_matrix_csv, index=True, encoding="utf-8-sig")
            print(f"行列形式の移動時間データを {output_matrix_csv} に保存しました。")

            return routes  # ← ルート情報を返す
        except Exception as e:
            print(f"エラー: {e}")
            return []


    def set_vehicle_info(self, num_vehicles, vehicle_capacity, vehicle_file="omaezaki_vehicles.csv"):
        """
        車両情報を設定するメソッド
        :param num_vehicles: 車両の台数
        :param vehicle_capacity: 各車両の容量
        :param vehicle_file: 車両情報を保存するCSVファイル名
        """

        # 車両の生成
        vehicles = [{"id": i, "capacity": vehicle_capacity} for i in range(num_vehicles)]
        df_vehicles = pd.DataFrame(vehicles)

        # CSV保存
        df_vehicles.to_csv(vehicle_file, index=False, columns = ["id", "capacity"])

        return vehicles
    
    
    def interpolate_points(self, lat1, lon1, lat2, lon2, distance, interval_m=5):
        """緯度経度を指定して5m間隔で補間点を作る"""
        start = np.array([lat1, lon1])
        end = np.array([lat2, lon2])

        # 総距離（メートル単位）
        num_points = int(distance // interval_m)

        if num_points <= 1:
            return [(lat1, lon1), (lat2, lon2)]

        lats = np.linspace(lat1, lat2, num_points + 1)
        lons = np.linspace(lon1, lon2, num_points + 1)
        return list(zip(lats, lons))
    
    def latlon_to_int_id(self, lat, lon):
        return int(lat * 1e7) * 10**9 + int(lon * 1e7)

    def get_filtered_road_network(self, include_types=None, exclude_types=None, output_file="filtered_network.graphml", elev=50, n=10, nrate=0.5):
        if include_types:
            custom_filter = '["highway"~"' + "|".join(include_types) + '"]'
        elif exclude_types:
            custom_filter = '["highway"!~"' + "|".join(exclude_types) + '"]'
        else:
            custom_filter = None

        G = ox.graph_from_place("Omaezaki, Shizuoka, Japan", network_type="drive", custom_filter=custom_filter)
        
        edges = list(G.edges(keys=True, data=True))
        total = len(edges)
        
        edges_to_remove = []
        distances = []
        for idx, (u, v, key, data) in enumerate(edges):
            if "highway" not in data:
                print(f"Edge {idx+1}: highwayなし → 削除予定")
                edges_to_remove.append((u, v, key))
        
        
        for idx, (u, v, key, data) in enumerate(edges):
            try:
                lat1, lon1 = G.nodes[u]['y'], G.nodes[u]['x']
                lat2, lon2 = G.nodes[v]['y'], G.nodes[v]['x']
                distance_m = geodesic((lat1, lon1), (lat2, lon2)).meters
                distances.append(distance_m)
                #print(f"Edge {idx+1}/{total}: from {u} to {v} = {distance_m:.2f} m")
                
                
                ####対象edgeをn分割してrate%が対象標高以下ならば削除する
                points = self.interpolate_points(lat1, lon1, lat2, lon2, distance_m, interval_m=round(distance_m/n,1))
                # 各点に標高を付与
                points_with_elev = [(lat, lon, self.get_elevation_from_latlon(lat, lon)) for lat, lon in points]                
                # 条件に合う点の数をカウント
                num_high = sum(1 for _, _, e in points_with_elev if e >= elev)
                # 全体に対する割合
                ratio = num_high / len(points_with_elev)
                if ratio < nrate:
                    edges_to_remove.append((u, v, key))
                
                
                #egdeを区切る場合はうまくいかない
                """
                if distance_m > 100:
                    points = self.interpolate_points(lat1, lon1, lat2, lon2, distance_m, interval_m=50)
                    edges_to_remove.append((u, v, key))
                    # ノード作成（補間点に新IDを付ける）
                    new_ids = []
                    for lat, lon in points:
                        nid = self.latlon_to_int_id(lat, lon)
                        if nid not in G.nodes:
                            G.add_node(nid, y=lat, x=lon)
                        new_ids.append(nid)
                    
                    # new_ids の先頭と末尾を u, v に接続する（標高条件が満たされていれば）
                    start_pt = (G.nodes[new_ids[0]]['y'], G.nodes[new_ids[0]]['x'])
                    end_pt = (G.nodes[new_ids[-1]]['y'], G.nodes[new_ids[-1]]['x'])

                    start_elev = self.get_elevation_from_latlon(*start_pt)
                    u_elev = self.get_elevation_from_latlon(G.nodes[u]['y'], G.nodes[u]['x'])
                    if start_elev > elev and u_elev > elev:
                        G.add_edge(u, new_ids[0], **data)

                    end_elev = self.get_elevation_from_latlon(*end_pt)
                    v_elev = self.get_elevation_from_latlon(G.nodes[v]['y'], G.nodes[v]['x'])
                    if end_elev > elev and v_elev > elev:
                        G.add_edge(new_ids[-1], v, **data)
                        
                    # ノード間を新しいエッジでつなぐ（標高条件あり）
                    for i in range(len(new_ids) - 1):
                        pt1 = (G.nodes[new_ids[i]]['y'], G.nodes[new_ids[i]]['x'])
                        pt2 = (G.nodes[new_ids[i+1]]['y'], G.nodes[new_ids[i+1]]['x'])
                        elev1 = self.get_elevation_from_latlon(*pt1)
                        elev2 = self.get_elevation_from_latlon(*pt2)
                        if elev1 > elev and elev2 > elev:
                            G.add_edge(new_ids[i], new_ids[i+1], **data)
                            # 通常のエッジ追加に加えて逆方向も追加
                            G.add_edge(new_ids[i+1], new_ids[i], **data)
                
                else:
                    elev1 = self.get_elevation_from_latlon(lat1, lon1)
                    elev2 = self.get_elevation_from_latlon(lat2, lon2)
                    if not (elev1 > elev and elev2 > elev):
                        edges_to_remove.append((u, v, key))          
                """
            except Exception as e:
                print(f"⚠️ エラー on edge {u}-{v}: {e}")
                continue

            if idx % max(1, total // 20) == 0:
                percent = (idx + 1) / total * 100
                print(f"{percent:.1f}% 完了（{idx + 1}/{total}）")
                
        
        # 不要なエッジを削除
        G.remove_edges_from(edges_to_remove)
        print(f"{len(edges_to_remove)}/{len(edges)}件({len(edges_to_remove)/len(edges):.2})のエッジを削除しました")
        
        
        ox.save_graphml(G, filepath=output_file)
        print(f"標高{elev}m超の部分のみで構成したネットワークを {output_file} に保存しました。")
        print("egdeごとの距離")
        print("最大値:", max(distances))
        print("最小値:", min(distances))
        print("平均:", statistics.mean(distances))
        print("中央値:", statistics.median(distances))
        print("標準偏差:", statistics.stdev(distances))  # 不偏分散の標準偏差
        return G
    
    def load_elev(self, npfile, csvfile):
        # 標高データを読み込む
        self.elevation_array = np.load(npfile)

        # 緯度経度をCSVから読み込む（上記と同様）
        with open(csvfile, mode="r") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            self.lat_min = float(row["lat_min"])
            self.lat_max = float(row["lat_max"])
            self.lon_min = float(row["lon_min"])
            self.lon_max = float(row["lon_max"])

        #print("✅ elevation_array shape:", elevation_array.shape)
        #print("📍 緯度経度範囲:", lat_min, "～", lat_max, " / ", lon_min, "～", lon_max)
        
    def get_elevation_from_latlon(self, lat, lon):
        """
        指定された緯度経度から標高を取得する関数
        Get elevation from specified latitude and longitude.

        Parameters:
            lat (float): 緯度 / Latitude
            lon (float): 経度 / Longitude

        Returns:
            float: 指定地点の標高（範囲外は None） / Elevation value or None if out of bounds
        """
        nrows, ncols = self.elevation_array.shape

        # 範囲チェック / Check if coordinates are within bounds
        if not (self.lat_min <= lat <= self.lat_max) or not (self.lon_min <= lon <= self.lon_max):
            print("❌ 指定した地点はデータの範囲外です。 / Location is outside data bounds.")
            return None

        # 緯度・経度を配列インデックスに変換 / Convert lat/lon to array indices
        row_ratio = (self.lat_max - lat) / (self.lat_max - self.lat_min)
        col_ratio = (lon - self.lon_min) / (self.lon_max - self.lon_min)

        row = int(round(row_ratio * (nrows - 1)))
        col = int(round(col_ratio * (ncols - 1)))

        # 安全対策：インデックスが範囲外にならないように制限 / Clip indices to valid range
        row = min(max(row, 0), nrows - 1)
        col = min(max(col, 0), ncols - 1)

        return self.elevation_array[row, col]

    # GML標高データの読み込み関数（補間は行わない）
    # Function to parse GML elevation data (no interpolation here)
    def parse_gml_dem_10m(self, xml_file):
        ns = {'gml': "http://www.opengis.net/gml/3.2"}
        tree = ET.parse(xml_file)
        root = tree.getroot()

        try:
            # 緯度経度の範囲を取得 / Get coordinate bounds
            lower_corner = root.find('.//gml:Envelope/gml:lowerCorner', ns).text.split()
            upper_corner = root.find('.//gml:Envelope/gml:upperCorner', ns).text.split()
            lat_min, lon_min = map(float, lower_corner)
            lat_max, lon_max = map(float, upper_corner)

            # グリッドサイズ取得 / Get grid size
            grid_size = root.find('.//gml:GridEnvelope/gml:high', ns).text.split()
            grid_x, grid_y = map(int, grid_size)
            expected_size = (grid_x + 1) * (grid_y + 1)

            # 標高データ取得 / Get elevation data
            elevation_values = root.find('.//gml:tupleList', ns).text.strip().split("\n")
            elevations = np.array([float(e.split(",")[1]) for e in elevation_values])
            elevations[elevations == -9999.0] = np.nan  # 異常値をNaNに / Replace -9999 with NaN

            # サイズ調整 / Adjust size if mismatched
            if elevations.size < expected_size:
                elevations = np.pad(elevations, (0, expected_size - elevations.size), mode='edge')
            elif elevations.size > expected_size:
                elevations = elevations[:expected_size]

            # 配列の形状を設定（緯度 × 経度）/ Reshape as (lat, lon)
            elevations = elevations.reshape((grid_y + 1, grid_x + 1))

            return {
                "filename": os.path.basename(xml_file),
                "elevations": elevations,
                "lat_range": (lat_min, lat_max),
                "lon_range": (lon_min, lon_max)
            }

        except Exception as e:
            print(f"❌ エラー: {xml_file} - {e}")
            return None


    # GMLタイルを結合して補間処理を一括で行う関数
    # Merge multiple GML tiles and apply interpolation afterward
    def merge_gml_elevation_tiles_10m(self, input_dir, output_prefix="merged_10m_elevation"):
        results = []
        for filename in os.listdir(input_dir):
            if filename.endswith('.xml'):
                filepath = os.path.join(input_dir, filename)
                result = self.parse_gml_dem_10m(filepath)
                if result:
                    results.append(result)

        if not results:
            print("❌ 有効なデータが見つかりませんでした。 / No valid data found.")
            return

        print(f"📦 タイル読み込み完了: {len(results)} タイル / Loaded {len(results)} tiles")

        # 解像度を取得 / Get resolution from sample tile
        sample = results[0]
        tile_rows, tile_cols = sample["elevations"].shape
        lat_res = (sample["lat_range"][1] - sample["lat_range"][0]) / tile_rows
        lon_res = (sample["lon_range"][1] - sample["lon_range"][0]) / tile_cols

        # 全体の緯度経度範囲を算出 / Calculate overall lat/lon bounds
        lat_min_all = min(r["lat_range"][0] for r in results)
        lat_max_all = max(r["lat_range"][1] for r in results)
        lon_min_all = min(r["lon_range"][0] for r in results)
        lon_max_all = max(r["lon_range"][1] for r in results)

        # 全体の配列サイズを決定 / Compute merged array size
        total_rows = int(round((lat_max_all - lat_min_all) / lat_res))
        total_cols = int(round((lon_max_all - lon_min_all) / lon_res))
        merged_array = np.full((total_rows, total_cols), np.nan)  # 初期値は NaN / Initialize with NaNs

        # 各タイルを正しい位置に挿入 / Place each tile in correct position
        for r in results:
            elev = r["elevations"]
            lat_start, lat_end = r["lat_range"]
            lon_start, lon_end = r["lon_range"]

            row_start = int(round((lat_max_all - lat_end) / lat_res))
            row_end = row_start + elev.shape[0]
            col_start = int(round((lon_start - lon_min_all) / lon_res))
            col_end = col_start + elev.shape[1]

            merged_array[row_start:row_end, col_start:col_end] = elev

        print(f"🗺 統合完了！ shape: {merged_array.shape} / Merge complete!")

        # NaNを一括補間（最小値で埋めてから平滑化）
        # Interpolate NaNs after merging (fill with min value, then smooth)
        if np.any(np.isnan(merged_array)):
            print("🔧 NaN を一括補間中... / Interpolating NaN values...")
            filled = np.nan_to_num(merged_array, nan=np.nanmin(merged_array))
            smoothed = scipy.ndimage.gaussian_filter(filled, sigma=1)
            merged_array[np.isnan(merged_array)] = smoothed[np.isnan(merged_array)]
            print("✅ 補間完了 / Interpolation complete")

        # 結果を保存 / Save results
        np.save(f"{output_prefix}.npy", merged_array)
        np.savetxt(f"{output_prefix}.csv", merged_array, delimiter=",", fmt="%.2f")
        print(f"✅ 統合配列を保存しました: {output_prefix}.npy / .csv")

        # 緯度経度範囲をCSVに保存 / Save lat/lon range to CSV
        with open("latlon_range.csv", mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["lat_min", "lat_max", "lon_min", "lon_max"])
            writer.writerow([lat_min_all, lat_max_all, lon_min_all, lon_max_all])
        print("✅ 緯度経度範囲を latlon_range.csv に保存しました。 / Saved coordinate range to latlon_range.csv")

        return merged_array, lat_min_all, lat_max_all, lon_min_all, lon_max_all

     
if __name__ == "__main__":
    start = time.time()  # 開始時刻
    
    # クラスのインスタンス化
    geo = CVRP_Geography("r2ka22223.topojson")
    print('インスタンス化完了', time.time() - start)
    
    
    # 高度情報をcsv及びnpy化（1回だけ実行すればOK）
    input_dir = "PackDLMap_xml_10m"
    elevation_array, lat_min, lat_max, lon_min, lon_max = geo.merge_gml_elevation_tiles_10m(
        input_dir,
        output_prefix="PackDLMap_xml_10m_output"
    )
    print('高度情報をcsv及びnpy化（1回だけ実行）', time.time() - start)
    
    
    # 高度ファイルの読み込み
    geo.load_elev("PackDLMap_xml_10m_output.npy", "latlon_range.csv")
    print('高度読み込み終了', time.time() - start)
    
    elev = 0
    #対象高度(m)以下を削除する
    geo.get_filtered_road_network(output_file=f"omaezaki_≤{elev}melev.graphml", elev=elev, nrate=0.5)#exclude_types=["trunk"], 
    print('ネットワーク再作成', time.time() - start)
    
    # 可視化
    geo.plot_colored_roads(f"omaezaki_≤{elev}melev.graphml", f"omaezaki_road_map_≤{elev}melev.png")
    print('可視化', time.time() - start)
    
    # 各種ファイルのパス
    graphml_file = f"omaezaki_≤{elev}melev.graphml"  # GraphMLファイル
    nodes_csv = "omaezaki_nodes.csv"  # ノードデータCSV
    output_csv = "omaezaki_travel_time.csv"  # 通常形式の保存先
    output_matrix_csv = "omaezaki_travel_time_matrix.csv"  # 行列形式の保存先
    # 移動時間の計算と保存(時間がかかる)
    geo.calculate_travel_times(graphml_file, nodes_csv, output_csv, output_matrix_csv)

    try:
        travel_time_matrix = pd.read_csv(output_matrix_csv, index_col=0)
        # 行列の値だけを取得（インデックスや列名を除く）
        matrix_values = travel_time_matrix.values
        # データの最初の数行を表示
        print(matrix_values)
        # 上三角行列を対称行列に変換する関数
        def make_symmetric(matrix):
            # 上三角部分を下三角にコピーして対称行列を作成
            symmetric_matrix = matrix + matrix.T - np.diag(np.diag(matrix))
            return symmetric_matrix

        # 対称行列を作成
        symmetric_matrix = make_symmetric(np.array(matrix_values))

        # 結果を表示
        print("Symmetric Matrix:")
        print(symmetric_matrix)

        # 対称行列をCSVファイルに保存
        symmetric_matrix_df = pd.DataFrame(symmetric_matrix, index=travel_time_matrix.index, columns=travel_time_matrix.columns)
        symmetric_matrix_df.to_csv("omaezaki_symmetric_travel_time_matrix.csv")

    except FileNotFoundError:
        geo.calculate_travel_times(graphml_file, nodes_csv, output_csv, output_matrix_csv)

    
    print('処理完了', time.time() - start)
    
    
