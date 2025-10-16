from typing import Optional

from einops import rearrange
from torch import Tensor
from tsl.nn.models import BaseModel
from torch import nn

from models.layers import (DRNN,
                           HierPoolFactory,
                           MessagePassingMethods,
                           AttentionReadout,
                           Encoder)
import torch
from baseline.Graph_Pooling_Benchmark.Regression.models.baseline import AsymCheegerCut, Diff, MinCut, DMoN, Hosc, just_balance
from baseline.Graph_Pooling_Benchmark.Regression.models.baseline import TopK, SAG, ASAP, PAN, CO, CGI, KMIS, GSA, HGPSL
from torch_geometric.data import Data
model_dict = {
    'pool_model_MinCut': MinCut,
    'pool_model_TopK': TopK,
    'pool_model_SAG': SAG,
    'pool_model_ASAP': ASAP,
    'pool_model_PAN': PAN,
    'pool_model_CO': CO,
    'pool_model_CGI': CGI,
    'pool_model_KMIS': KMIS,
    'pool_model_GSA': GSA,
    'pool_model_HGPSL': HGPSL,
    'pool_model_AsymCheegerCut': AsymCheegerCut,
    'pool_model_Diff': Diff,
    'pool_model_DMoN': DMoN,
    'pool_model_Hosc': Hosc,
    'pool_model_just_balance': just_balance
}
class PoolModel(BaseModel):
    """The Hierarchical Downsampling Time Then Space (HD-TTS) model.

    Args:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden state in the temporal and
            spatial modules.
        n_nodes (int): The number of nodes in the graph.
        horizon (int): The prediction horizon.
        rnn_layers (int): The depth of the temporal hierarchy.
        pooling_layers (int): The depth of the spatial hierarchy.
        exog_size (int, optional): The size of the exogenous features.
            (default: ``0``)
        mask_size (int, optional): The size of the mask.
            (default: ``0``)
        dilation (int, optional): The dilation factor in the temporal
            processing. (default: ``2``)
        cell (str, optional): The type of recurrent cell in the TMP layers.
            (default: ``"gru"``)
        mp_kernel_size (int, optional): The kernel size in the SMP layers, i.e.,
            the number of hops in the message passing. (default: ``1``)
        mp_method (MessagePassingMethods, optional): The method used for SMP.
            (default: ``"diffconv"``)
        mp_stage (str, optional): The stage at which SMP is performed, i.e.,
            before, after, or both before and after each pooling layer.
            (default: ``"both"``)
        recursive_lifting (bool): If ``True``, then the lifting operation is
            performed recursively along the spatial hierarchy in a top-down
            fashion. (default: ``True``)
        fully_connected_readout (bool, optional): Whether to use fully connected
            readout in the readout module. (default: ``False``)
        multi_step_scores (bool, optional): Whether to compute multistep scores
            in the readout module. (default: ``True``)
        activation (str, optional): The activation function used in the modules.
            (default: ``"relu"``)
        dropout (float, optional): The dropout rate.
            (default: ``0.``)
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_nodes: int,
                 horizon: int=1,
                 rnn_layers: int=1,
                 pooling_layers: int=1,
                 exog_size: int = 0,
                 mask_size: int = 1,
                 # Temporal params
                 dilation: int = 1,
                 cell: str = "gru",
                 # MP params
                 mp_kernel_size: int = 1,
                 mp_method: MessagePassingMethods = "anisoconv",
                 # Pooling params
                 mp_stage: str = "both",
                 recursive_lifting: bool = True,
                 # Decoder params
                 fully_connected_readout: bool = False,
                 multi_step_scores: bool = True,
                 activation: str = "relu",
                 dropout: float = 0.,
                 feature = "fresh_snow",
                 model_name=None):
        super().__init__()
        self.pooling_layers = pooling_layers
        self.dilation = dilation
        self.feature = feature

        if mask_size > 0:
            assert mask_size == 1 or mask_size == input_size

        #  TEMPORAL MODULES  ##################################################
        self.encoder_time = Encoder(input_size=input_size,
                                    hidden_size=hidden_size,
                                    exog_size=exog_size,
                                    mask_size=mask_size,
                                    emb_size=1,
                                    n_nodes=n_nodes,
                                    activation=activation)
        self.linear = nn.Linear(2,1)

        model_class = model_dict[model_name]
        self.spatial_conv = model_class(in_channels=64,
                hidden_channels=hidden_size,
                out_channels=64,
                num_classes=2,
                lin_before_conv=False,
                ratio=0.5) 

        self.spatial_conv = HierPoolFactory(input_size=hidden_size,
                                            hidden_size=hidden_size,
                                            n_layers=pooling_layers,
                                            connect_op='mean',
                                            mp_method=mp_method,
                                            kernel_size=mp_kernel_size,
                                            mp_stage=mp_stage,
                                            recursive_lifting=recursive_lifting,
                                            keep_initial_features=True,
                                            activation=activation,feature=feature)

        #  READOUT MODULES  ###################################################
        ro_layers = (pooling_layers + 1) * rnn_layers
        self.readout = AttentionReadout(
            input_size=hidden_size,
            hidden_size=hidden_size * 2,
            output_size=input_size,
            dim_size=ro_layers,
            horizon=horizon,
            dim=1,
            fully_connected=fully_connected_readout,
            multi_step_scores=multi_step_scores,
            ff_layers=2,
            activation=activation,
            dropout=dropout)

    def forward(self,
                x: Tensor,  # [batch, time, node, input_size]
                edge_index: Tensor,
                edge_weight: Optional[Tensor] = None,
                input_mask: Optional[Tensor] = None,
                u: Optional[Tensor] = None):
        #  TEMPORAL ENCODING  #################################################
        feature = x.clone()
        if self.feature is not None:
            feature = torch.load(f"/data/zzn/insitu/insitu_daily_filter_7_14/{self.feature}_embeddings.pt").to('cuda')

        
        x = x.permute(1,0)
        x = x.unsqueeze(0)
        x = x.unsqueeze(-1)

        

        if input_mask is not None:
            input_mask = input_mask.permute(1,0)
            input_mask = input_mask.unsqueeze(0)
            input_mask = input_mask.unsqueeze(-1)
        x = self.encoder_time(x, mask=input_mask, u=u)#([1, 2, 33191, 64])#x torch.Size([1, 2, 33191, 64])
        # out_time: [batch, time_layers, nodes, hidden]
        x=x.permute(0,2,3,1)
        # out_time = self.drnn(x)#x torch.Size([1, 2, 33191, 64])
        out_time = self.linear(x)
        out_time=out_time.permute(0,3,1,2)
        out_time = out_time[0,0]

        data = Data(x=out_time, edge_index=edge_index, edge_weight=edge_weight)

        #  SPATIAL ENCODING  ##################################################
        # out_space: [batch, space_layers, time_layers, nodes, hidden]
        out_space =  self.spatial_conv(data)
        # [batch, space_layers*time_layers, nodes, hidden]
        out_space = rearrange(out_space, 'b s t ... -> b (s t) ...')

        #  READOUT  ###########################################################
        # out: [batch, time, nodes, out_size]
        # beta: [batch, time*pooling_layers, nodes]
        out, states, alpha = self.readout(out_space)

        out=out[0,0]

        return out, None, alpha, None

    def get_coarsened_graphs(self, edge_index: Tensor = None,
                             edge_attr: Optional[Tensor] = None,
                             num_nodes: int = None,
                             cached: bool = False):
        return self.spatial_conv.get_coarsened_graphs(edge_index,
                                                      edge_attr,
                                                      num_nodes,
                                                      cached)

    def predict(self,
                x: Tensor,  # [batch, time, node, input_size]
                edge_index: Tensor,
                edge_weight: Optional[Tensor] = None,
                input_mask: Optional[Tensor] = None,
                u: Optional[Tensor] = None):
        out, *_ = self(x, edge_index, edge_weight, input_mask, u)
        return out
