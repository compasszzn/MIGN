import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN

class Model(torch.nn.Module):
    def __init__(self, node_features,filters, periods=4,output_length=14,feature=None):
        super(Model, self).__init__()
        self.recurrent = A3TGCN(node_features, filters, periods)
        self.linear = torch.nn.Linear(filters, output_length)

        self.position_embed = torch.nn.Linear(9, 2)

        self.position_embedding_src =torch.load(f"/data/zzn/insitu/insitu_daily_filter_7_14/{feature}_embeddings.pt").to('cuda')

    def forward(self, x, edge_index, edge_weight):
        # means = x.mean(1, keepdim=True).detach()
        # x = x - means
        # stdev = torch.sqrt(
        #     torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x /= stdev
        x = torch.nan_to_num(x, nan=0.0)
        position = self.position_embed(self.position_embedding_src)

        x = x+position
        h = self.recurrent(x.view(x.shape[0], 1, x.shape[1]), edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)

        # h = h * stdev
        # h = h + means
        return (h,)