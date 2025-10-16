import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.attention import STConv
class Model(torch.nn.Module):
    def __init__(self, node_features,filters,output_length,feature):
        super(Model, self).__init__()
        self.recurrent = STConv(num_nodes=None, in_channels=node_features,hidden_channels=filters,out_channels=filters,kernel_size=1,K=1)
        # self.recurrent_2 = STConv(num_nodes=None, in_channels=filters,hidden_channels=filters,out_channels=filters,kernel_size=1,K=1)
        self.linear = torch.nn.Linear(filters, output_length)

    def forward(self, x, edge_index, edge_weight):
        # means = x.mean(1, keepdim=True).detach()
        # x = x - means
        # stdev = torch.sqrt(
        #     torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x /= stdev

        h = self.recurrent(x, edge_index, edge_weight)
        # h = self.recurrent_2(h[0][0], edge_index, edge_weight)
        y = F.relu(h)
        y = self.linear(y)
        y=y[0,0]
        # y = y * stdev
        # y = y + means
        return (y, h)