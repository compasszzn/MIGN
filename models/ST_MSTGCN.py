import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.attention import MSTGCN

class Model(torch.nn.Module):
    def __init__(self, node_features,filters,feature):
        super(Model, self).__init__()
        if feature == "fresh_snow":
            node = 33191
        elif feature == "accumulated_precipitation":
            node = 42664
        self.recurrent = MSTGCN(nb_block=1, in_channels=node_features,K=1,nb_chev_filter=1,nb_time_filter=1,time_strides=1,num_for_predict=1,len_input=1)
        self.linear = torch.nn.Linear(filters, 1)

    def forward(self, x, edge_index, edge_weight):
        # means = x.mean(1, keepdim=True).detach()
        # x = x - means
        # stdev = torch.sqrt(
        #     torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x /= stdev

        h = self.recurrent(x, edge_index)

        # h = h * stdev
        # h = h + means
        h=h[0]
        return (h, )