import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GNN(nn.Module):
    def __init__(self, node_features, edge_features, hidden_channels, heads, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATv2Conv(node_features, hidden_channels//2, heads=heads, dropout=dropout, edge_dim=edge_features, add_self_loops=False)
        self.conv2 = GATv2Conv((hidden_channels//2) * heads, hidden_channels, heads=heads//2, dropout=dropout, edge_dim=edge_features, add_self_loops=False)
        self.conv3 = GATv2Conv(hidden_channels * (heads//2), hidden_channels, heads=1, dropout=dropout, edge_dim=edge_features, add_self_loops=False, concat=False)

        self.norm1 = nn.LayerNorm((hidden_channels//2) * heads)
        self.norm2 = nn.LayerNorm(hidden_channels * (heads//2))
        self.norm3 = nn.LayerNorm(hidden_channels)


    def forward(self, x, edge_index, edge_attr, batch):
        x1 = self.conv1(x, edge_index, edge_attr=edge_attr)
        x1 = self.norm1(x1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x2 = self.conv2(x1, edge_index, edge_attr=edge_attr)
        x2 = self.norm2(x2)
        x2 = F.elu(x2 + x1)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x3 = self.conv3(x2, edge_index, edge_attr=edge_attr)
        x3 = self.norm3(x3)

        return x3