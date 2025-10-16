import torch
import torch.nn.functional as F
from torch_geometric.nn import GPSConv,GCNConv
import pdb
class Model(torch.nn.Module):
    def __init__(self, node_features, filters, output_length):
        super(Model, self).__init__()
        # 使用 GPSConv 进行图卷积
        self.gps_conv = GPSConv(channels=filters, 
                                conv=GCNConv(filters,filters), 
                                heads=1)
        self.embed = torch.nn.Linear(node_features, filters)
        self.linear = torch.nn.Linear(filters, output_length)

    def forward(self, x, edge_index,edge_weight):
        # pdb.set_trace()
        h = self.gps_conv(self.embed(x), edge_index)
        y = F.relu(h)
        y = self.linear(y)
        return (y, h)
