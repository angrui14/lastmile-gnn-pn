import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from torch import nn

def zscore(x):
    mean = x.mean()
    std = x.std()
    return (x - mean) / std

def positional_encoding(coords, d_model=16):
    assert d_model % 4 == 0, "d_model must be multiple of 4"
    N = coords.shape[0]
    x = coords[:, 0]
    y = coords[:, 1]
    
    pe = np.zeros((N, d_model))
    
    for i in range(d_model // 4):
        div_term = 10000 ** (4 * i / d_model)
        pe[:, 4*i] = np.sin(x / div_term)
        pe[:, 4*i + 1] = np.cos(x / div_term)
        pe[:, 4*i + 2] = np.sin(y / div_term)
        pe[:, 4*i + 3] = np.cos(y / div_term)
    
    return pe


def calculate_route_centroid(route):
    return route["lat"].mean(), route["lng"].mean()


def calculate_zone_coords(df):
    grouped = df.groupby('zone_id')
    zone_coords = {}
    for zone_id, zone in grouped:
        if zone_id == "Unknown": continue
        x = zone['lat'].mean()
        y = zone['lng'].mean()
        zone_coords[zone_id] = (x, y)

    return zone_coords


def init_zone_embeddings(zone_encoder, zone_coords, emb_size=16):
    embedding = nn.Embedding(len(zone_encoder.classes_), emb_size)

    with torch.no_grad():
        known_indices = []
        known_embeddings = []

        for i, zone in enumerate(zone_encoder.classes_):
            if zone in zone_coords:
                zone_center = zone_coords[zone]
                coords_tensor = torch.tensor([zone_center[0], zone_center[1]], dtype=torch.float32).unsqueeze(0)
                coords_tensor = zscore(coords_tensor)
                pos_enc = positional_encoding(coords_tensor, emb_size)
                embedding.weight[i] = torch.tensor(pos_enc, dtype=torch.float32).squeeze(0)
                known_indices.append(i)
                known_embeddings.append(embedding.weight[i].clone())
        
        if known_embeddings:
            mean_embedding = torch.stack(known_embeddings).mean(dim=0)
            for i, zone in enumerate(zone_encoder.classes_):
                if zone not in zone_coords:
                    embedding.weight[i] = mean_embedding.clone()

    return embedding


def enhanced_features(coords):
    x, y = coords[:, 0], coords[:, 1]
    angles = torch.atan2(y, x)
    print(angles)


def plot(train_lengths, val_lengths, filename="train_results.png"):
    sns.set_theme(style="whitegrid")

    epochs = range(1, len(train_lengths) + 1)
    df = pd.DataFrame({
        'Epoch': list(epochs) * 2,
        'Average length (s)': train_lengths + val_lengths,
        'Set': ['Train'] * len(train_lengths) + ['Validation'] * len(val_lengths)
    })

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='Epoch', y='Average length (s)', hue='Set', style='Set',
                markers=True, dashes=False, palette=['#1f77b4', '#ff7f0e'])

    plt.title('Average Tour Length per Epoch', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Average Length (s)')
    plt.tight_layout()
    plt.grid(True)

    # Guardar resultado
    plt.savefig(filename)
    plt.close()