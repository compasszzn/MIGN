import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvLSTM

class Model(torch.nn.Module):
    def __init__(self, node_features, filters, output_length):
        super(Model, self).__init__()
        self.recurrent = GConvLSTM(node_features, filters, 1)
        self.linear = torch.nn.Linear(filters, output_length)

    def forward(self, x, edge_index, edge_weight, h=None, c=None):
        # means = x.mean(1, keepdim=True).detach()
        # x = x - means
        # stdev = torch.sqrt(
        #     torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x /= stdev

        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)

        # h = h * stdev
        # h = h + means
        return h, h_0, c_0