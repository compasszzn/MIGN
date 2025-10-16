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
import pdb

class HDTTSModel(BaseModel):
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
                 output_size: int,
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
                 feature = None,
                 pooling = 'kmis'):
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
                                    emb_size=None,
                                    n_nodes=n_nodes,
                                    activation=activation,feature=feature)
        # self.linear = nn.Linear(1,1)

        # self.drnn = DRNN(input_size=hidden_size,
        #                  hidden_size=hidden_size,
        #                  dilation=dilation,
        #                  n_layers=rnn_layers,
        #                  return_only_last_state=True,
        #                  dropout=dropout,
        #                  cell=cell)
        # self.receptive_field = self.drnn.receptive_field

        #  SPATIAL MODULES  ###################################################
        self.spatial_conv = HierPoolFactory(input_size=hidden_size,
                                            hidden_size=hidden_size,
                                            n_layers=pooling_layers,
                                            connect_op='mean',
                                            mp_method=mp_method,
                                            kernel_size=mp_kernel_size,
                                            mp_stage=mp_stage,
                                            recursive_lifting=recursive_lifting,
                                            keep_initial_features=True,
                                            activation=activation,feature=feature,pooling=pooling)

        #  READOUT MODULES  ###################################################
        ro_layers = (pooling_layers + 1) * rnn_layers
        self.readout = AttentionReadout(
            input_size=hidden_size,
            hidden_size=hidden_size * 2,
            output_size=output_size,
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
        # print(x.shape)
        if self.feature is not None:
            feature = torch.load(f"/data/zzn/insitu/insitu_daily_filter_7_14/{self.feature}_embeddings.pt").to('cuda')
        x = torch.nan_to_num(x, nan=0.0)
        
        # x = x.permute(1,0)
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)

        

        if input_mask is not None:
            input_mask = input_mask.permute(1,0)
            input_mask = input_mask.unsqueeze(0)
            input_mask = input_mask.unsqueeze(-1)
        x = self.encoder_time(x, mask=input_mask, u=u)#([1, 2, 33191, 64])#x torch.Size([1, 2, 33191, 64])
        # out_time: [batch, time_layers, nodes, hidden]
        x=x.permute(0,2,3,1)
        # out_time = self.drnn(x)#x torch.Size([1, 2, 33191, 64])
        # out_time = self.linear(x)
        out_time=x.permute(0,3,1,2)

        #  SPATIAL ENCODING  ##################################################
        # out_space: [batch, space_layers, time_layers, nodes, hidden]
        out_space, pooled = self.spatial_conv(out_time, edge_index, edge_weight,
                                              cached=False,feature=feature)
        # [batch, space_layers*time_layers, nodes, hidden]

        out_space = rearrange(out_space, 'b s t ... -> b (s t) ...')#torch.Size([1, 2, 1, 20531, 64])

        #  READOUT  ###########################################################
        # out: [batch, time, nodes, out_size]
        # beta: [batch, time*pooling_layers, nodes]
        out, states, alpha = self.readout(out_space)#torch.Size([1, 2, 20531, 64])
        # pdb.set_trace()
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
