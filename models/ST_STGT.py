import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GCNConv

class Model(torch.nn.Module):
    def __init__(self, node_features, filters, output_length):
        super().__init__()
        self.spatial_gnn = GCNConv(node_features, filters)
        self.temporal_transformer = TransformerConv(filters, filters, heads=4, concat=True)
        self.linear = torch.nn.Linear(filters * 4, output_length)  # 因为 heads=4，需要乘以4

    def forward(self, x, edge_index,edge_weight):
        x = self.spatial_gnn(x, edge_index)
        x = F.relu(x)
        h = self.temporal_transformer(x, edge_index)
        x = self.linear(h)
        return (x,h)