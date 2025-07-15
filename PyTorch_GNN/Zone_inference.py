import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import json
import pandas as pd
import h3
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pyproj import Transformer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from GNNEncoder import GNN
from PointerDecoder import PointerDecoder
import utils

def calculate_time(id_route, stop_list):
    total_time = 0
    for src, tgt in zip(stop_list[:-1], stop_list[1:]):
        total_time += times[id_route][src][tgt]
    return total_time

path = '../almrrc2021/almrrc2021-data-evaluation/model_apply_inputs/'
path2 = '../almrrc2021/almrrc2021-data-evaluation/model_score_inputs/'

with open(path + 'eval_travel_times.json') as f:
    times = json.load(f)

with open(path2 + 'eval_actual_sequences.json') as f:
    actual = json.load(f)

with open("zone_dfs/labels_dict.json") as f:
    labels_dict = json.load(f)

df = pd.read_parquet("../df_la_test.parquet")

df.loc[:, "h3_index"] = df.apply(lambda row: h3.latlng_to_cell(row["lat"], row["lng"], res=6), axis=1)
df.loc[:, "cluster"] = df.apply(lambda row: labels_dict.get(row["h3_index"], -1.0), axis=1)

grouped = df.groupby(by="route_id")

avoid_creating = False
if os.path.exists("graph_lists/graph_list_test_zone_def.pt"):
    graph_list = torch.load("graph_lists/graph_list_test_zone_def.pt", weights_only=False)
    avoid_creating = True
else:
    graph_list = []

all_zones = df["zone_id"].unique().tolist()

zone_encoder = LabelEncoder()
zone_encoder.fit(all_zones)

zone_coords = utils.calculate_zone_coords(df)
zone_embedding = utils.init_zone_embeddings(zone_encoder, zone_coords)

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

stops_idx = {}

for route_id, route in grouped:
    stops_idx[route_id] = {i : stop for i, stop in enumerate(route.stop_id)}
    if avoid_creating:
        continue

    idx_stops = {stop: i for i, stop in enumerate(route.stop_id)}
    coords_dict = route.set_index("stop_id")[["lat", "lng"]].to_dict("index")

    route_graphs = []
    for cluster in route["cluster"].unique():
        subroute = route[route["cluster"] == cluster]

        if subroute.empty:
            continue

        local_to_global = {}
        for i, stop in enumerate(subroute["stop_id"]):
            local_to_global[i] = idx_stops[stop]
        
        idx_local = {stop: i for i, stop in enumerate(subroute.stop_id)}

        centroid_lat, centroid_lng = utils.calculate_route_centroid(subroute)
        centroid_x, centroid_y = transformer.transform(centroid_lng, centroid_lat)
        centroid = torch.tensor([centroid_x, centroid_y], dtype=torch.float)

        coords = torch.tensor(subroute[["lat", "lng"]].values, dtype=torch.float)
        coords_zs = utils.zscore(coords)
        positional_encodings = utils.positional_encoding(coords_zs)
        pe_tensor = torch.tensor(positional_encodings, dtype=torch.float)

        zone_ids = zone_encoder.transform(subroute["zone_id"])
        zone_ids = torch.tensor(zone_ids, dtype=torch.long)
        with torch.no_grad():
            zone_vectors = zone_embedding(zone_ids)

        lats = coords[:, 0].numpy()
        lngs = coords[:, 1].numpy()

        xs, ys = transformer.transform(lngs, lats)
        xy_coords = torch.tensor(list(zip(xs, ys)), dtype=torch.float)

        dist_to_centroid = torch.norm(xy_coords - centroid, dim=1).unsqueeze(1)

        x = torch.cat([pe_tensor, zone_vectors, dist_to_centroid], dim=1)

        edges = []
        edge_weights = []


        projected_coords = {
            stop_id: transformer.transform(stop["lng"], stop["lat"])
            for stop_id, stop in coords_dict.items()
        }

        dist_extremes_list = []
        if len(subroute) == 1:
            dist_extremes_list.append([0, 0])
        else:
            for i, stop in enumerate(subroute.stop_id):
                d = times[route_id][stop]
                d = {k: v for k, v in d.items() if k in subroute["stop_id"].values and k != stop}
                min_dist = min(d.values())
                max_dist = max(d.values())
                dist_extremes_list.append([min_dist, max_dist])
                for stop2 in d.keys():
                    if stop != stop2:
                        edges.append([i, idx_local[stop2]])
                        # x1, y1 = projected_coords[stop]
                        # x2, y2 = projected_coords[stop2]
                        # dist = np.hypot(x2 - x1, y2 - y1)  # Distance between stops              
                        edge_weights.append([d[stop2]])

        dist_extremes = torch.tensor(dist_extremes_list, dtype=torch.float)
        x = torch.cat([x, dist_extremes], dim=1)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)

        route_graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, local_to_global=local_to_global, idx_local=idx_local, cluster=cluster))

    start = route[route["type"] == "Station"].iloc[0]["stop_id"]
    start = idx_stops[start]

    graph_list.append(Data(graphs=route_graphs, route_id=route_id, start=start))


if not os.path.exists("graph_lists/graph_list_test_zone_def.pt"):
    torch.save(graph_list, "graph_lists/graph_list_test_zone_def.pt")

encoders = []
decoders = []

for i in range(57):
    encoder = GNN(node_features=35, edge_features=1, hidden_channels=128, heads=4)
    decoder = PointerDecoder(128)
    checkpoint = torch.load(f"zone_results/best_checkpoint_{i}.pt", weights_only=False)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    encoder.eval()
    decoder.eval()
    encoders.append(encoder)
    decoders.append(decoder)

general_encoder = GNN(node_features=35, edge_features=1, hidden_channels=128, heads=4)
general_decoder = PointerDecoder(128)
checkpoint = torch.load("results/results_definitive/best_checkpoint.pt", weights_only=False)
general_encoder.load_state_dict(checkpoint["encoder_state_dict"])
general_decoder.load_state_dict(checkpoint["decoder_state_dict"])
general_encoder.eval()
general_decoder.eval()


times_10 = []
for i in range(10):

    loader = DataLoader(graph_list, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    encoder = None
    decoder = None

    # times_pred = []
    # times_actual = []
    # times_dif = []
    # pred_tours = []
    # actual_tours = []
    # route_ids = []

    inference_time = []

    for route in tqdm(loader):
        with torch.no_grad():
            inf_time = 0
            inf_times = []
            route_id = route.route_id[0]
            route_time_dists = times[route_id]
            # route_ids.append(route_id)
            start = route.start[0].item()
            start_code = stops_idx[route_id][start]
            start_cluster = df[(df["route_id"] == route_id) & (df["stop_id"] == start_code)]["cluster"].values[0]
            next_start = start_code
            next_cluster = start_cluster
            pred_route = []
            visited_clusters = []
            for _ in range(len(route.graphs[0])):
                # print(f"Next cluster is {next_cluster}")
                graph = None
                for subgraph in route.graphs[0]:
                    # print(subgraph.cluster)
                    if subgraph.cluster == next_cluster:
                        graph = subgraph
                        break
                        
                cluster = graph.cluster
                visited_clusters.append(cluster)
                # print(visited_clusters)

                idx_local = graph.idx_local
                next_start_idx = idx_local[next_start]
                
                start_inf = time.time()
                if len(graph.x) == 1:
                    pred_subtour = [torch.tensor(next_start_idx)]
                elif len(graph.x) == 2:
                    pred_subtour = [torch.tensor(next_start_idx), torch.tensor(0) if next_start_idx == 1 else torch.tensor(1)]
                else:
                    if cluster == -1.0:
                        encoder = general_encoder
                        decoder = general_decoder
                    else:
                        encoder = encoders[int(cluster)]
                        decoder = decoders[int(cluster)]
                    embeddings = encoder(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
                    pred_subtour, _ = decoder(embeddings, torch.tensor([next_start_idx]), torch.tensor([0] * len(graph.x)))
                    pred_subtour = pred_subtour[0]
                end_inf = time.time()

                inf_time += (end_inf - start_inf)
                inf_times.append(end_inf - start_inf)

                pred_route.extend([graph.local_to_global[s.item()] for s in pred_subtour])
                
                dists = route_time_dists[stops_idx[route_id][pred_route[-1]]]
                dists = {k: v for k, v in dists.items() if df[(df["route_id"] == route_id) & (df["stop_id"] == k)]["cluster"].values[0] not in visited_clusters}
                if len(dists) > 0:

                    sorted_dists = dict(sorted(dists.items(), key=lambda item: item[1]))
                    next_start = list(sorted_dists.keys())[0]
                    next_cluster = df[(df["route_id"] == route_id) & (df["stop_id"] == next_start)]["cluster"].values[0]
            
            inference_time.append(inf_time)

            actual_route = df[df["route_id"] == route_id].sort_values(by="order")["stop_id"].tolist()         
            pred_route = [stops_idx[route_id][s] for s in pred_route]
            if set(actual_route) != set(pred_route):
                print(f"Route {route_id} has different stops set")
                break

            # pred_tours.append(pred_route)
            # actual_tours.append(actual_route)
            # pred_time = calculate_time(route_id, pred_route)
            # actual_time = calculate_time(route_id, actual_route)
            # times_pred.append(pred_time)
            # times_actual.append(actual_time)
            # times_dif.append(pred_time - actual_time)
    
    times_10.append(inference_time)

# with open("zone_times/inference_times.json", "w") as f:
#     json.dump(inference_time, f)

# with open("zone_results/times_pred.json", "w") as f:
#     json.dump(times_pred, f)

# with open("zone_results/times_actual.json", "w") as f:
#     json.dump(times_actual, f)

# with open("zone_results/times_dif.json", "w") as f:
#     json.dump(times_dif, f)

with open("zone_times/inference_times_10.json", "w") as f:
    json.dump(times_10, f)