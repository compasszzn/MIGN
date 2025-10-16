import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN

class Model(torch.nn.Module):
    def __init__(self, node_features,filters,output_length):
        super(Model, self).__init__()
        self.recurrent = TGCN(node_features, filters)
        self.linear = torch.nn.Linear(filters, output_length)

    def forward(self, x, edge_index, edge_weight, prev_hidden_state=None):
        # means = x.mean(1, keepdim=True).detach()
        # x = x - means
        # stdev = torch.sqrt(
        #     torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x /= stdev

        h = self.recurrent(x, edge_index, edge_weight, prev_hidden_state)
        y = F.relu(h)
        y = self.linear(y)

        # y = y * stdev
        # y = y + means
        return y, h