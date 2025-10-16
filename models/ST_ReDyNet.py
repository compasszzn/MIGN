import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.attention import STConv
import pdb
class Model(torch.nn.Module):
    def __init__(self, node_features,filters,output_length,feature):
        super(Model, self).__init__()
        self.recurrent = STConv(num_nodes=None, in_channels=node_features,hidden_channels=filters,out_channels=filters,kernel_size=1,K=1)
        # self.recurrent_2 = STConv(num_nodes=None, in_channels=filters,hidden_channels=filters,out_channels=filters,kernel_size=1,K=1)
        self.linear = torch.nn.Linear(filters, output_length)

        self.station_encoder = StationEncoder(encoder_dim=filters,
                                              station_embed_dim=filters,
                                              input_time=1)
        
        self.dygcn = DyGCN(dim_in=1 * filters, dim_out=filters, cheby_k=1,
                           embed_dim=filters)
        self.decoder = Decoder(input_dim=filters, output_dim=output_length)

        self.gcn_activation = torch.nn.GELU()
        self.union_norm = torch.nn.LayerNorm(filters)
        self.station_norm = torch.nn.LayerNorm(filters)
        self.date_norm = torch.nn.LayerNorm(filters)
        self.env_norm = torch.nn.LayerNorm(filters)

        # self.date_encoder = DateEncoder(encoder_dim=filters, date_embed_dim=filters,
        #                                 input_time=1)
        
        self.env_encoder = EnvEncoder(encoder_dim=filters, input_time=1)
        self.vae_encoder = VaeEncoder(input_dim=filters, output_dim=int(filters/2))
        self.rec_decoder = VaeDecoder(input_dim=int(filters/2), output_dim=filters)
        self.fc_mu =  torch.nn.Linear(int(filters/2), int(filters/2))
        self.fc_log_var =  torch.nn.Linear(int(filters/2),int(filters/2))
        self.rec_layer_norm =  torch.nn.LayerNorm(filters)

        self._init_parameters()
    def reparameterize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def _init_parameters(self):
        print('Initializing parameters...')
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)

    def forward(self, x, edge_index, edge_weight):
        # means = x.mean(1, keepdim=True).detach()
        # x = x - means
        # stdev = torch.sqrt(
        #     torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x /= stdev
        
        flow = x[:,:1]
        env = x[:,:2]

        flow_station = self.station_encoder(flow,env)  # [B,N,64]
        flow_env = self.env_encoder(flow, env)  # [B,N,64]

        union_origin = flow_station + flow_env  # [B,N,64]
        union_origin = self.union_norm(union_origin)  # [B,N,64]
        
        z = self.vae_encoder(union_origin)  # [B,N,64]
        mu = self.fc_mu(z)  # [B,N,64]
        log_var = self.fc_log_var(z)  # [B,N,64]
        z = self.reparameterize(mu, log_var)  # [B,N,64]
        rec = self.rec_decoder(z)  # [B,N,64]
        union = union_origin - rec  # [B,N,64]
        union = self.rec_layer_norm(union)  # [B,N,64]

        flow_station = self.station_norm(flow_station)
        # flow_date = self.date_norm(flow_date)
        flow_env = self.env_norm(flow_env)

        gcn_output = self.dygcn(flow.unsqueeze(0), union.unsqueeze(0), flow_station.unsqueeze(0), return_supports=False)  # [B,N,64]
        # pdb.set_trace()
        gcn_output = self.gcn_activation(gcn_output)  # [B,N,64]

        output = self.decoder(gcn_output)  # [B,N,4*2]
        y=output[0]

        return (y, output)

class StationEncoder(torch.nn.Module):
    def __init__(self,encoder_dim, station_embed_dim, input_time):
        super(StationEncoder, self).__init__()
        self.input_time = input_time
        self.site_embedding=torch.nn.Linear(2, encoder_dim)
        self.flow_encoder = torch.nn.Linear(input_time, encoder_dim)
        self.embed_encoder = torch.nn.Linear(station_embed_dim, encoder_dim)
        self.station_embed_dim = station_embed_dim
        self.encoder_dim = encoder_dim
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, flow,env):
        # x: [B, T, N, 2]
        
        num_sites, features = flow.shape

        site_embeds = self.site_embedding(env)
        site_embeds = self.dropout(site_embeds)  # [B, N, embed_dim]

        # flow = flow.permute(0, 2, 1, 3).reshape(batch_size, num_sites, time_len * features)  # [B, N, T*2]
        flow = self.flow_encoder(flow)  # [B, N, output_dim]

        site_features = self.embed_encoder(site_embeds)  # [B, N, output_dim]

        p = F.sigmoid(site_features)
        q = F.tanh(flow)

        station_flow_feature = p * q + (1 - p) * site_features  # [B, N, output_dim]
        station_flow_feature = station_flow_feature.view(num_sites, self.encoder_dim)  # [B, N, output_dim]
        

        return station_flow_feature

class DateEncoder(torch.nn.Module):
    def __init__(self, encoder_dim, date_embed_dim, input_time):
        super(DateEncoder, self).__init__()
        self.input_time = input_time
        self.hour_embedding = torch.nn.Embedding(24, date_embed_dim)
        self.weekday_embedding = torch.nn.Embedding(7, date_embed_dim)
        self.flow_encoder = torch.nn.Linear(input_time * 2, encoder_dim)
        self.embed_encoder = torch.nn.Linear(input_time * date_embed_dim, encoder_dim)
        self.date_embed_dim = date_embed_dim
        self.encoder_dim = encoder_dim
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, flow, date):
        # x:  [B, T, N, 2]
        batch_size, time_len, num_sites, features = flow.shape
        # date: [B, T, N, 2]
        hour = date[..., 0]  # [B, T, N]
        weekday = date[..., 1]  # [B, T, N]
        hour = hour.long()
        weekday = weekday.long()

        hour_embeds = self.hour_embedding(hour)  # [B, T, N, embed_dim]
        weekday_embeds = self.weekday_embedding(weekday)  # [B, T, N, embed_dim]
        date_embeds = hour_embeds + weekday_embeds

        date_embeds = date_embeds.permute(0, 2, 1, 3).reshape(batch_size, num_sites, time_len * self.date_embed_dim)
        date_embeds = self.dropout(date_embeds)

        flow = flow.permute(0, 2, 1, 3).reshape(batch_size, num_sites, time_len * features)  # [B, N, T*2]
        flow = self.flow_encoder(flow)  # [N, B, output_dim]

        date_feature = self.embed_encoder(date_embeds)  # [B, N, output_dim]

        p = F.sigmoid(date_feature)
        q = F.tanh(flow)

        date_flow_feature = p * q + (1 - p) * date_feature  # [B, N, output_dim]
        date_flow_feature = date_flow_feature.view(batch_size, num_sites, self.encoder_dim)  # [B, N, output_dim]

        return date_flow_feature


class EnvEncoder(torch.nn.Module):
    def __init__(self, encoder_dim, input_time):
        super(EnvEncoder, self).__init__()
        self.input_time = input_time
        self.flow_encoder = torch.nn.Linear(input_time , encoder_dim)
        self.env_encoder = torch.nn.Linear(2 , encoder_dim)
        self.encoder_dim = encoder_dim
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, flow, env):
        # x: [B, T, N, 2]
        # pdb.set_trace()
        num_sites, features = flow.shape

        # flow = flow.permute(0, 2, 1, 3).reshape(batch_size, num_sites, time_len * features)  # [B, N, T*2]
        flow = self.flow_encoder(flow)  # [B, N, output_dim]

        # env = env.permute(0, 2, 1, 3).reshape(batch_size, num_sites, time_len * env_features)  # [B, N, T*5]
        env = self.dropout(env)

        env_feature = self.env_encoder(env)  # [B, N, output_dim]

        p = F.sigmoid(env_feature)
        q = F.tanh(flow)

        enc_flow_feature = p * q + (1 - p) * env_feature
        enc_flow_feature = enc_flow_feature.view(num_sites, self.encoder_dim)  # [B, N, output_dim]

        return enc_flow_feature


class GTUnit(torch.nn.Module):
    def __init__(self, input_dim):
        super(GTUnit, self).__init__()
        self.input_dim = input_dim
        self.gate = torch.nn.Linear(input_dim, input_dim)
        self.update = torch.nn.Linear(input_dim, input_dim)
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        p = self.sigmoid(self.gate(x))
        q = self.tanh(self.update(x))
        h = p * q + (1 - p) * x
        return h


class Decoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gtu = GTUnit(input_dim)
        self.fc = torch.nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, flow):
        # num_sites, features,_ = flow.shape
        flow = self.gtu(flow)
        flow = self.fc(flow)
        return flow

class VaeEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VaeEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gtu = GTUnit(input_dim)
        # pdb.set_trace()
        self.fc = torch.nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, flow):
        num_sites, features = flow.shape
        flow = self.gtu(flow)
        flow = self.fc(flow)
        return flow


class VaeDecoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VaeDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gtu = GTUnit(input_dim)
        self.fc = torch.nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, flow):
        num_sites, features = flow.shape
        flow = self.gtu(flow)
        flow = self.fc(flow)
        return flow
    
class DyGCN(torch.nn.Module):
    def __init__(self, dim_in, dim_out, cheby_k, embed_dim, aggregate_type='sum'):
        super(DyGCN, self).__init__()
        self.cheby_k = cheby_k
        self.weights_pool = torch.nn.Parameter(torch.FloatTensor(embed_dim, cheby_k, dim_in, dim_out))
        self.bias_pool = torch.nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        if aggregate_type == 'weighted_sum':
            self.weights_cheby = torch.nn.Parameter(torch.ones(cheby_k))
        self.aggregate_type = aggregate_type

    def forward(self, x, all_emb, station_emb, return_supports=False):
        batch_size, node_num, _ = all_emb.shape
        supports = F.softmax(F.relu(torch.matmul(all_emb, all_emb.transpose(1, 2))), dim=-1)  # [B, N, N]
        t_k_0 = torch.eye(node_num).to(supports.device)  # [B, N, N]
        t_k_0 = t_k_0.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, N]
        support_set = [t_k_0, supports]
        for k in range(2, self.cheby_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports_cheby = torch.stack(support_set, dim=0)  # [cheby_k, B, N, N]
        supports_cheby = supports_cheby.permute(1, 0, 2, 3)  # [B, cheby_k, N, N]

        # B, N, cheby_k, dim_in, dim_out
        weights = torch.einsum('bni,ikop->bnkop', station_emb, self.weights_pool)
        # B, N, dim_out
        bias = torch.matmul(station_emb, self.bias_pool)
        # B, cheby_k, N, dim_in
        x_g = torch.einsum('bkij,bjd->bkid', supports_cheby, x)
        # B, N, cheby_k, dim_out
        x_g_conv = torch.einsum('bkni,bnkio->bnko', x_g, weights)
        # B, N, dim_out
        if self.aggregate_type == 'sum':
            x_g_conv = x_g_conv.sum(dim=2) + bias
        elif self.aggregate_type == 'weighted_sum':
            x_g_conv = x_g_conv * self.weights_cheby.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            x_g_conv = x_g_conv.sum(dim=2) + bias

        if return_supports:
            return x_g_conv, supports
        return x_g_conv


class MutilHeadDyGCN(torch.nn.Module):
    def __init__(self, dim_in, dim_out, cheby_k, embed_dim, num_heads, aggregate_type='sum'):
        super(MutilHeadDyGCN, self).__init__()
        self.dygcn_list = torch.nn.ModuleList([DyGCN(dim_in, dim_out // num_heads, cheby_k, embed_dim, aggregate_type)
                                         for _ in range(num_heads)])
        self.num_heads = num_heads

    def forward(self, x, all_emb, station_emb, return_supports=False):
        head_outs = []
        for i in range(self.num_heads):
            head_outs.append(self.dygcn_list[i](x, all_emb, station_emb))

        if return_supports:
            return torch.cat(head_outs, dim=-1), torch.Tensor(0).to(x.device)
        return torch.cat(head_outs, dim=-1)