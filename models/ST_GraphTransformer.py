import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
import pdb
class Model(torch.nn.Module):
    def __init__(self, node_features, filters, output_length):
        super(Model, self).__init__()
        self.transformer_conv = TransformerConv(in_channels=node_features, 
                                                out_channels=filters, 
                                                heads=4, 
                                                concat=True)
        self.linear = torch.nn.Linear(filters * 4, output_length)  # 因为 heads=4，需要乘以4

    def forward(self, x, edge_index, edge_weight):
        h = self.transformer_conv(x, edge_index)
        y = F.relu(h)
        y = self.linear(y)
        return (y, h)
