import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN
import pdb

class Model(torch.nn.Module):
    def __init__(self, node_features,filters, periods=4,output_length=14):
        super(Model, self).__init__()
        self.input_length=node_features
        self.recurrent = A3TGCN(node_features, filters, periods)
        self.linear = torch.nn.Linear(filters, output_length)

    def forward(self, x, edge_index, edge_weight):
        # means = x.mean(1, keepdim=True).detach()
        # x = x - means
        # stdev = torch.sqrt(
        #     torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x /= stdev
        
        
        h = self.recurrent(x.view( 1,x.shape[0], x.shape[1]), edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        # pdb.set_trace()
        # h = h * stdev
        # h = h + means

        return (h,)