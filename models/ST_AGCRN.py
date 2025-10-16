import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import AGCRN
import pdb
class Model(torch.nn.Module):
    def __init__(self, node_features,filters,output_length):
        super(Model, self).__init__()
        self.recurrent = AGCRN(number_of_nodes = None,
                              in_channels = node_features,
                              out_channels = output_length,
                              K = 2,
                              embedding_dimensions = filters)
        self.linear = torch.nn.Linear(2, filters)
        # self.e = torch.empty(self.node,4).to('cuda')

    def forward(self, x, e=None, prev_hidden_state=None):
        # x = x.view(1, -1, 2)
        # means = x.mean(1, keepdim=True).detach()
        # x = x - means
        # stdev = torch.sqrt(
        #     torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x /= stdev
        x = x.unsqueeze(0)
        # e = torch.empty(x.shape[1],4).to('cuda')
        # pdb.set_trace()
        h_0 = self.recurrent(x[:,:,0:-2], self.linear(x[0,:,-2:]), None)
        y = F.relu(h_0)
        # pdb.set_trace()
        # y = y * stdev
        # y = y + means
        # y = self.linear(y)

        return y[0], h_0