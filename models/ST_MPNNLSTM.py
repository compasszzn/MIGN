import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import MPNNLSTM
import pdb


class Model(torch.nn.Module):
    def __init__(self, node_features,filters,output_length,feature):
        super(Model, self).__init__()
        
        self.recurrent = MPNNLSTM(node_features, filters, num_nodes = None, window = 1,dropout = 0.5)

        self.linear = torch.nn.Linear(130+output_length-1, output_length)
    def forward(self, x, edge_index, edge_weight):
        # means = x.mean(1, keepdim=True).detach()
        # x = x - means
        # stdev = torch.sqrt(
        #     torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x /= stdev
        # pdb.set_trace()
        h = self.recurrent(x, edge_index, edge_weight)
        # pdb.set_trace()
        h = F.relu(h)
        h = self.linear(h)

        # h = h * stdev
        # h = h + means
        return (h,)