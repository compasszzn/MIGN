import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GCNModel(torch.nn.Module):
    def __init__(self, node_features, hidden_channels, output_channels):
        super(GCNModel, self).__init__()

        # 定义两层GCN
        self.conv1 = GCNConv(node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_channels)

        self.linear = torch.nn.Linear(output_channels, output_channels)

    def forward(self, x, edge_index, edge_weight=None):
        # edge_index = torch.stack([edge_index[1],edge_index[0]],dim=0)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index, edge_weight)

        x = self.linear(x)

        return (x,)

