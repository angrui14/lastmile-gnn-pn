# Torch imports
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Other imports
import pandas as pd
import numpy as np
import json
import time
import os
from pyproj import Transformer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import gc
from scipy.stats import ttest_ind

import utils

from GNNEncoder import GNN
from PointerDecoder import PointerDecoder


# Set config
def set_config(seed):
    print("Setting config...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.autograd.set_detect_anomaly(True)


# Load data
def load_data(amazon_path, df_path):
    print("Loading data...")
    with open(amazon_path + 'travel_times.json') as f:
        times = json.load(f)
    df = pd.read_parquet(df_path)
    df = df[df['order'] <= 30]
    grouped = df.groupby(by="route_id")

    return df, grouped, times


# Calculate route time
def calculate_time(times, id_route, stop_list):
    total_time = 0
    for src, tgt in zip(stop_list[:-1], stop_list[1:]):    
        total_time += times[id_route][src][tgt]
    return total_time


# Generate dataset
def generate_dataset(df, grouped, times):
    stops_idx = {}
    avoid_creating = False

    # Load graph list if it exists
    if os.path.exists("graph_lists/graph_list_tsp30.pt"):
        graph_list = torch.load("graph_lists/graph_list_tsp30.pt", weights_only=False)
        avoid_creating = True
    else:
        graph_list = []

    # graph_list = []

    all_zones = df["zone_id"].unique().tolist()

    zone_encoder = LabelEncoder()
    zone_encoder.fit(all_zones)

    zone_coords = utils.calculate_zone_coords(df)
    zone_embedding = utils.init_zone_embeddings(zone_encoder, zone_coords)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    for route_id, route in tqdm(grouped, desc="Generating graphs"):
        stops_idx[route_id] = {i : stop for i, stop in enumerate(route.stop_id)} # Mapping from string code to integer index
        if avoid_creating: # If the graph list exists in a file, skip creation
            continue
        
        idx_stops = {stop: i for i, stop in enumerate(route.stop_id)} # Mapping from integer index to string code

        centroid_lat, centroid_lng = utils.calculate_route_centroid(route)
        centroid_x, centroid_y = transformer.transform(centroid_lng, centroid_lat)
        centroid = torch.tensor([centroid_x, centroid_y], dtype=torch.float)

        coords = torch.tensor(route[["lat", "lng"]].values, dtype=torch.float)
        coords_zs = utils.zscore(coords)
        positional_encodings = utils.positional_encoding(coords_zs)
        pe_tensor = torch.tensor(positional_encodings, dtype=torch.float)

        zone_ids = zone_encoder.transform(route["zone_id"])
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

        # coords_dict = route.set_index("stop_id")[["lat", "lng"]].to_dict("index")

        # projected_coords = {
        #     stop_id: transformer.transform(stop["lng"], stop["lat"])
        #     for stop_id, stop in coords_dict.items()
        # }

        dist_extremes_list = []
        for i, stop in enumerate(route.stop_id):
            d = times[route_id][stop]
            d = {k: v for k, v in d.items() if k != stop and k in route["stop_id"].tolist()}

            min_dist = min(d.values())
            max_dist = max(d.values())
            dist_extremes_list.append([min_dist, max_dist])

            sorted_d = dict(sorted(d.items(), key=lambda item: item[1])) # Sort stops by distance

            # if route[route["stop_id"] == stop]["order"].values[0] != 0:
            #     top_d = dict(list(sorted_d.items())[:51]) # Maintain the N closest stops
            # else:
            #     top_d = sorted_d

            top_d = sorted_d

            for stop2 in top_d.keys():
                if stop != stop2:
                    edges.append([i, idx_stops[stop2]])
                    # x1, y1 = projected_coords[stop]
                    # x2, y2 = projected_coords[stop2]
                    # dist = np.hypot(x2 - x1, y2 - y1)  # Distance between stops              
                    edge_weights.append([top_d[stop2]])

        dist_extremes = torch.tensor(dist_extremes_list, dtype=torch.float)
        x = torch.cat([x, dist_extremes], dim=1)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)

        start = route[route["order"] == 0]["stop_id"].values[0]
        start = idx_stops[start]

        graph_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, route_id=route_id, start_node=start))

    # Store graph list if it does not exist
    if not os.path.exists("graph_lists/graph_list_tsp30.pt"):
        torch.save(graph_list, "graph_lists/graph_list_tsp30.pt")
        # del projected_coords
        # del coords_dict
        del idx_stops

    total_len = len(graph_list)
    val_ratio = 0.2
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len

    train_dataset, val_dataset = random_split(graph_list, [train_len, val_len]) # Split dataset into training and validation sets (80% train, 20% validation)

    del graph_list
    gc.collect()

    return train_dataset, val_dataset, stops_idx


# Compute actual mean of length of validation routes
def actual_validation_length(df, val_dataset, times):
    val_loader_test = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    tour_lengths = []
    for route in val_loader_test:
        route_id = route.route_id[0]
        actual_route = df[df["route_id"] == route_id].sort_values(by="order")["stop_id"].tolist()
        actual_time = calculate_time(times, route_id, actual_route)
        tour_lengths.append(actual_time)

    return np.mean(tour_lengths)
    

# Reinforce loss
def reinforce_loss(tour_length, log_probs, baseline=None):
    reward = -tour_length
    if baseline is None:
        advantage = reward
    else:
        advantage = reward - baseline

    # Normalize reward
    mean = advantage.mean()
    std = advantage.std(unbiased=False) + 1e-8
    advantage = (advantage - mean) / std

    probs = torch.exp(log_probs)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=0).mean()

    loss = -torch.mean(log_probs * advantage.detach()) - 0.05 * entropy
   
    return loss


# Validation
def evaluate(encoder, decoder, loader, stops_idx, times, device):
    encoder.eval()
    decoder.eval()

    tour_lengths = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(device)
            node_embeddings = encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            batch_size = batch.num_graphs

            start_nodes = []
            for i in range(batch_size):
                node_indices = (batch.batch == i).nonzero(as_tuple=False).squeeze()
                local_start = batch.start_node[i].item()
                global_start = node_indices[local_start]
                start_nodes.append(global_start)

            start_nodes = torch.stack(start_nodes)

            tours, _ = decoder(node_embeddings, start_nodes, batch.batch)

            for i, tour in enumerate(tours):
                route_id = batch.route_id[i]
                node_mask = (batch.batch == i)

                # Global to local index mapping
                # All stops are in the same array, so the indices are global
                # We need to map the global indices to local indices (indices in a single graph) to obtain the tour
                global_node_indices = node_mask.nonzero(as_tuple=False).squeeze()
                global_to_local = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(global_node_indices)}

                stop_indices = [global_to_local[node.item()] for node in tour]

                tour_length = calculate_time(times, route_id, [stops_idx[route_id][s] for s in stop_indices])

                tour_lengths.append(tour_length)

    encoder.train()
    decoder.train()

    return sum(tour_lengths) / len(tour_lengths)


def get_temperature(epoch):
    initial_temp = 2.0
    final_temp = 0.5
    total_epochs = 100
    return max(final_temp, initial_temp - (initial_temp - final_temp) * (epoch / total_epochs))


# Train
def train(encoder, decoder, baseline_encoder, baseline_decoder, optimizer, scheduler, train_loader, val_loader, epochs, stops_idx, times, device="cpu", print_every=5):
    print("Starting to train...")
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    baseline_encoder = baseline_encoder.to(device)
    baseline_encoder.eval()

    baseline_decoder = baseline_decoder.to(device)
    baseline_decoder.eval()

    train_losses = []
    avg_tour_lengths = []
    avg_tour_lenghts_val = []

    best_val_length = float('inf')
    patience = 50
    patience_counter = 0

    tau = 0.99

    epoch_times = []

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0
        epoch_tour_lengths = []
        batch_time = 0

        for batch_idx, batch in enumerate(train_loader):
            start_time = time.time()
            batch = batch.to(device)

            optimizer.zero_grad()

            # batch.batch = idx showing to which graph in the batch a node belongs to
            node_embeddings = encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            batch_size = batch.num_graphs

            start_nodes = []
            for i in range(batch.num_graphs):
                node_indices = (batch.batch == i).nonzero(as_tuple=False).squeeze()
                local_start_node = batch.start_node[i]
                global_start_node = node_indices[local_start_node]
                start_nodes.append(global_start_node)

            start_nodes = torch.stack(start_nodes)

            temp = get_temperature(epoch)
            tours, log_probs = decoder(node_embeddings, start_nodes, batch.batch, temperature=temp)

            batch_tour_lengths = []
            batch_log_probs = []

            for i in range(batch_size):
                route_id = batch.route_id[i]
                node_mask = (batch.batch == i)

                # Global to local index mapping
                # All stops are in the same array, so the indices are global
                # We need to map the global indices to local indices (indices in a single graph) to obtain the tour
                global_node_indices = node_mask.nonzero(as_tuple=False).squeeze()
                global_to_local = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(global_node_indices)}

                tour = tours[i]
                stop_indices = [global_to_local[node.item()] for node in tour]

                tour_length = calculate_time(times, route_id, [stops_idx[route_id][s] for s in stop_indices])

                batch_tour_lengths.append(tour_length)
                batch_log_probs.append(torch.sum(log_probs[i]))
                epoch_tour_lengths.append(tour_length)

            tour_lengths_tensor = torch.tensor(batch_tour_lengths, dtype=torch.float32, device=device).detach()
            log_probs_tensor = torch.stack(batch_log_probs)

            ###########################################
            ######### Moving average baseline #########
            ###########################################

            # current_mean = tour_lengths_tensor.mean().detach()
            # if moving_baseline is None:
            #     moving_baseline = current_mean
            # else:
            #     moving_baseline = alpha * moving_baseline + (1 - alpha) * current_mean

            # batch_loss = reinforce_loss(tour_lengths_tensor, log_probs_tensor, moving_baseline)

            # batch_loss.backward()

            ###########################################
            ############# Critic baseline #############
            ###########################################

            # baseline_pred = critic(node_embeddings, batch.batch)
            # critic_loss = F.mse_loss(baseline_pred, tour_lengths_tensor.detach())
            # batch_loss = reinforce_loss(tour_lengths_tensor, log_probs_tensor, baseline=baseline_pred.detach())
            # total_loss = batch_loss + critic_weight * critic_loss            
            # total_loss.backward()

            ###########################################
            ############ Rollout baseline #############
            ###########################################

            baseline_encoder.eval()
            baseline_decoder.eval()
            
            with torch.no_grad():
                rollout_embeddings = baseline_encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                rollout_tours, _ = baseline_decoder(rollout_embeddings, start_nodes, batch.batch)

            rollout_lengths = []
            
            for i in range(batch_size):
                route_id = batch.route_id[i]
                node_mask = (batch.batch == i)
                global_node_indices = node_mask.nonzero(as_tuple=False).squeeze()

                global_to_local = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(global_node_indices)}

                rollout_tour = rollout_tours[i]
                stop_indices = [global_to_local[node.item()] for node in rollout_tour]

                rollout_tour_length = calculate_time(times, route_id, [stops_idx[route_id][s] for s in stop_indices])

                rollout_lengths.append(rollout_tour_length)

            rollout_lengths_tensor = torch.tensor(rollout_lengths, dtype=torch.float32, device=device)

            loss = reinforce_loss(tour_lengths_tensor, log_probs_tensor, baseline=rollout_lengths_tensor)
            loss.backward()


            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)

            optimizer.step()

            # Free memory
            del log_probs_tensor
            del tour_lengths_tensor
            del node_embeddings, tours, log_probs
            del rollout_embeddings, rollout_tours
            del rollout_lengths_tensor
            torch.cuda.empty_cache() 

            epoch_loss += loss.item()

            end_time = time.time()

            batch_time += (end_time - start_time)

            # Print progress
            if (batch_idx + 1) % print_every == 0:
                recent_lengths = epoch_tour_lengths[-batch_size:]
                average_time = batch_time / print_every
                print(f"\033[92mEpoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Avg Tour Length: {sum(recent_lengths)/len(recent_lengths):.4f}, "
                      f"Average time per batch: {average_time:.2f}\033[0m")
                batch_time = 0

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_tour_length = torch.mean(torch.tensor(epoch_tour_lengths)).item()

        train_losses.append(avg_epoch_loss)
        avg_tour_lengths.append(avg_epoch_tour_length)

        val_avg_tour_length = evaluate(encoder, decoder, val_loader, stops_idx, times, device)
        # baseline_lengths, _ = evaluate(baseline_encoder, baseline_decoder, val_loader, stops_idx, times, device)

        # baseline_arr = np.array(baseline_lengths)
        # model_arr = np.array(model_lengths)

        # t_stat, p_value = ttest_ind(baseline_arr, model_arr, equal_var=True)
        # mean_baseline = baseline_arr.mean()
        # mean_model = model_arr.mean()
        # relative_improvement = (mean_baseline - mean_model) / mean_baseline

        avg_tour_lenghts_val.append(val_avg_tour_length)

        # scheduler.step(val_avg_tour_length)

        print(f"\033[94mEpoch {epoch+1}/{epochs} completed, Avg Loss: {avg_epoch_loss:.4f}, "
                f"Avg Tour Length: {avg_epoch_tour_length:.4f}, Validation Avg Tour Length: {val_avg_tour_length:.4f}\033[0m")
        
        epoch_end = time.time()
        epoch_times.append(epoch_end - epoch_start)
        utils.plot(avg_tour_lengths, avg_tour_lenghts_val, filename="train_results_30.png")
        
        # Save the best model based on validation tour length
        # if p_value < 0.05:
        if val_avg_tour_length < best_val_length:
            best_val_length = val_avg_tour_length
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_avg_tour_length": val_avg_tour_length
            }, "best_checkpoint_tsp30.pt")
            baseline_encoder.load_state_dict(encoder.state_dict())
            baseline_encoder.eval()
            baseline_decoder.load_state_dict(decoder.state_dict())
            baseline_decoder.eval()
            no_improvement = 0
            print(f"\033[35m New best model saved at epoch {epoch+1} with average validation tour length {val_avg_tour_length:.2f}\033[0m")
        else:
            patience_counter += 1

            # for baseline_param, model_param in zip(baseline_encoder.parameters(), encoder.parameters()):
            #     baseline_param.data = tau * baseline_param.data + (1 - tau) * model_param.data
            # baseline_encoder.eval()

            # for baseline_param, model_param in zip(baseline_decoder.parameters(), decoder.parameters()):
            #     baseline_param.data = tau * baseline_param.data + (1 - tau) * model_param.data
            # baseline_decoder.eval()

            print(f"\033[33m No improvement ({patience_counter}/{patience})\033[0m")

            if patience_counter >= patience:
                print("\033[91mEarly stopping triggered.\033[0m")
                break
        

    print("Training completed")
    return avg_tour_lengths, avg_tour_lenghts_val, best_val_length, epoch_times

        

if __name__ == "__main__":
    amazon_path = "../almrrc2021/almrrc2021-data-training/model_build_inputs/"
    df_path = "../df_la.parquet"

    set_config(16)
    df, grouped, times = load_data(amazon_path, df_path)
    train_dataset, val_dataset, stops_idx = generate_dataset(df, grouped, times)
    actual_avg_val = actual_validation_length(df, val_dataset, times)

    hidden_channels = 128
    lr = 1e-3
    epochs = 200
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    encoder = GNN(node_features=35, edge_features=1, hidden_channels=hidden_channels, heads=4)
    decoder = PointerDecoder(hidden_channels)

    baseline_encoder = GNN(node_features=35, edge_features=1, hidden_channels=hidden_channels, heads=4)
    baseline_decoder = PointerDecoder(hidden_channels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=4)

    encoder.train()
    decoder.train()

    optimizer = torch.optim.Adam([
        {'params': encoder.parameters()},
        {'params': decoder.parameters()},
    ], lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    print(f"Actual validation length is {actual_avg_val}")

    avg_tour_lengths, avg_tour_lenghts_val, best_val_length, epoch_times = train(encoder, decoder, baseline_encoder, baseline_decoder, optimizer, scheduler, train_loader, val_loader, epochs, stops_idx, times, device)
    print(f"Difference between validation tour length and actual average validation tour length: {(best_val_length - actual_avg_val):.2f} seconds")
    
    # torch.save(encoder.state_dict(), "encoder.pth")
    # torch.save(decoder.state_dict(), "decoder.pth")

    with open(f"train_times.json", "w") as f:
        json.dump(epoch_times, f)

    # Plot the results
    # utils.plot(avg_tour_lengths, avg_tour_lenghts_val)