import dgl
import torch
import torch.nn as nn
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import pickle
from models.layers.hetero import HeteroGraphConv
from models.layers.readout import AttentionReadout
import torch.nn.init as init
from einops import rearrange
import pdb


class HGNN_GCN_EDGE_WO_SH(nn.Module):
    def __init__(self, model_args,data_args):
        super(HGNN_GCN_EDGE_WO_SH, self).__init__()


        self.input_length=data_args['input_length']
        self.output_length=data_args['output_length']
        self.hidden_size = model_args['hidden_size']
        
        self.model_args=model_args
        self.data_args=data_args

        with open(self.data_args['climatology_dir'], 'rb') as f:
            self.climatology = pickle.load(f)

        self.Encoder =Graph_Encoder(self.model_args,self.data_args)
        self.Processor=MLP_Processor(self.model_args,self.data_args)
        self.Decoder = Graph_Decoder(self.model_args,self.data_args)
        if self.model_args['sh_before'] or self.model_args['sh_after']:
            if model_args['sh_level']==3:
                self.position_embedding_feature =nn.Parameter(torch.randn(9))
                self.position_embedding_healpix = nn.Parameter(torch.randn(9))
            elif model_args['sh_level']==1:
                self.position_embedding_feature =nn.Parameter(torch.randn(1)) 
                self.position_embedding_healpix = nn.Parameter(torch.randn(1))
            elif model_args['sh_level']==2:
                self.position_embedding_feature = nn.Parameter(torch.randn(4))
                self.position_embedding_healpix = nn.Parameter(torch.randn(4))
            elif model_args['sh_level']==4:
                self.position_embedding_feature =nn.Parameter(torch.randn(16))
                self.position_embedding_healpix = nn.Parameter(torch.randn(16))

 
    def forward(self, g,timestamp,sh_embedding):
        
        if self.model_args['sh_before'] or self.model_args['sh_after']:
            x=self.Encoder(g,timestamp,sh_embedding,self.position_embedding_feature,self.position_embedding_healpix )#(healpix,input_time,variable,hidden_state)
            x=self.Processor(x)#(healpix,output_time,variable,hidden_state)
            g=self.Decoder(x,g,sh_embedding,self.position_embedding_feature,self.position_embedding_healpix )

        else:
            x=self.Encoder(g,timestamp)#(healpix,input_time,variable,hidden_state)
            x=self.Processor(x)#(healpix,output_time,variable,hidden_state)
            g=self.Decoder(x,g)

        return g

class Graph_Encoder(nn.Module):
    def __init__(self, model_args,data_args):
        super(Graph_Encoder, self).__init__()

        self.edge_types=[]
        self.model_args=model_args
        self.data_args=data_args
        node_type = self.data_args['feature']
        for t in range(self.data_args['input_length']):
            edge_type = (node_type, f't{t}_to_healpix', 'healpix')
            self.edge_types.append(edge_type)
        if self.model_args['sh_before']:
            if model_args['sh_level']==3:
                self.convs = HeteroGraphConv(
                    {
                        edge_type: dglnn.GraphConv(1+9, self.model_args['hidden_size'],activation=nn.LeakyReLU())  # 可以根据类型自定义其他卷积层
                        for edge_type in self.edge_types
                    },
                    aggregate='sum' 
                )
            elif model_args['sh_level']==1:
                self.convs = HeteroGraphConv(
                    {
                        edge_type: dglnn.GraphConv(1+1, self.model_args['hidden_size'],activation=nn.LeakyReLU())  # 可以根据类型自定义其他卷积层
                        for edge_type in self.edge_types
                    },
                    aggregate='sum' 
                )
            elif model_args['sh_level']==2:
                self.convs = HeteroGraphConv(
                    {
                        edge_type: dglnn.GraphConv(1+4, self.model_args['hidden_size'],activation=nn.LeakyReLU())  # 可以根据类型自定义其他卷积层
                        for edge_type in self.edge_types
                    },
                    aggregate='sum' 
                )
            elif model_args['sh_level']==4:
                self.convs = HeteroGraphConv(
                    {
                        edge_type: dglnn.GraphConv(1+16, self.model_args['hidden_size'],activation=nn.LeakyReLU())  # 可以根据类型自定义其他卷积层
                        for edge_type in self.edge_types
                    },
                    aggregate='sum' 
                )
        else:
            self.convs = HeteroGraphConv(
                {
                    edge_type: dglnn.GraphConv(1, self.model_args['hidden_size'],activation=nn.LeakyReLU())  # 可以根据类型自定义其他卷积层
                    for edge_type in self.edge_types
                },
                aggregate='sum' 
            )

        self.norm  = dglnn.EdgeWeightNorm(norm='right')

        self.model_args=model_args
        self.data_args=data_args
        self.input_length=self.data_args['input_length']
        if self.model_args['sh_before']:

            self.position_embedding_src =torch.load(f"{self.data_args['embedding_dir']}/{node_type}_embeddings_{model_args['sh_level']}.pt").to('cuda')
            self.position_embedding_dst = torch.load(f"{self.data_args['embedding_dir']}/{data_args['refinement_level']}_healpix_embeddings_level_{model_args['sh_level']}.pt").to('cuda')

        # self.year_embedding = nn.Linear(3,64)

    def forward(self, g,timestamp,sh_embedding=None,position_embedding_feature=None,position_embedding_healpix=None):

        node_type = self.data_args['feature']
        for t in range(self.input_length):
            x_src={}
            x_dst={}
            x_src[node_type]=torch.nan_to_num(g.nodes[node_type].data[f't{t}'], nan=0.0)
            x_dst['healpix'] = g.nodes['healpix'].data[f'{node_type}_t{t}']
            subgraph = g.edge_type_subgraph([(node_type, f't{t}_to_healpix', 'healpix') for t in range(self.input_length)])

            if self.model_args['sh_before']:
                src_position_embeddings = torch.cat([self.position_embedding_src[sh_embedding[node_type][i].bool()] for i in range(sh_embedding[node_type].shape[0])],dim=0)
                dst_position_embeddings = torch.cat([self.position_embedding_dst for i in range(sh_embedding[node_type].shape[0])],dim=0)
                x_src[node_type]=torch.concat([x_src[node_type],src_position_embeddings*position_embedding_feature],dim=1)
                x_dst['healpix']=torch.concat([x_dst['healpix'],dst_position_embeddings*position_embedding_healpix],dim=1)
            g.nodes['healpix'].data[f'{node_type}_t{t}']=self.convs(subgraph,(x_src,x_dst))['healpix']
        



        concatenated_features = []
        node_type = self.data_args['feature']
        features = [g.nodes['healpix'].data[f'{node_type}_t{t}'] for t in range(self.input_length)]
        concatenated_feature = torch.stack(features, dim=1)
        concatenated_features.append(concatenated_feature)
        final_feature_tensor = torch.stack(concatenated_features, dim=1)
        return final_feature_tensor
        
class MLP_Processor(nn.Module):
    def __init__(self, model_args,data_args):
        super(MLP_Processor, self).__init__()
        self.model_args=model_args
        self.data_args=data_args
        self.input_length=self.data_args['input_length']
        self.output_length=self.data_args['output_length']
        self.hidden_size=self.model_args['hidden_size']
        self.mlp_prossesor=nn.Sequential(
                nn.Linear(self.input_length, self.output_length),
                nn.ReLU()  # 添加 ReLU 激活函数
            )
        self.atten_prossesor=nn.Sequential(
                nn.Linear(self.input_length, self.output_length),
                nn.ReLU()  # 添加 ReLU 激活函数
            )
        self.hidden_size=self.hidden_size
        self.feature_atten = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.model_args['head'],batch_first=True)
    def forward(self,x):
        attn_x = x.permute(0,2,1,3).reshape(-1,x.size(2),x.size(3))
        attn_output,_ = self.feature_atten(attn_x,attn_x,attn_x)
        attn_output = attn_output.reshape(x.size(0),x.size(2),x.size(1),x.size(3))
        attn_output = attn_output.permute(0,2,3,1)
        attn_output=self.atten_prossesor(attn_output)

        x = x.permute(0,1,3,2)
        x = self.mlp_prossesor(x)
        x = x + attn_output
        x = x.permute(0,3,1,2)
        return x
    
    
class Graph_Decoder(nn.Module):
    def __init__(self, model_args,data_args):
        super(Graph_Decoder, self).__init__()

        self.model_args=model_args
        self.data_args=data_args
        self.input_length=self.data_args['input_length']
        self.output_length=self.data_args['output_length']
        self.hidden_size=self.model_args['hidden_size']
        self.edge_types=[]
        node_type = self.data_args['feature']

        for t in range(self.data_args['output_length']):
            edge_type = ('healpix', f't{self.input_length+t}_to_{node_type}', node_type)
            self.edge_types.append(edge_type)

        if self.model_args['sh_after']:
            if model_args['sh_level']==3:
                self.convs = HeteroGraphConv(
                    {
                        edge_type: dglnn.GraphConv(self.model_args['hidden_size']+9,self.model_args['hidden_size'] ,activation=nn.LeakyReLU())
                        for edge_type in self.edge_types
                    },
                    aggregate='sum' 
                )
            elif model_args['sh_level']==1:
                self.convs = HeteroGraphConv(
                    {
                        edge_type: dglnn.GraphConv(self.model_args['hidden_size']+1,self.model_args['hidden_size'] ,activation=nn.LeakyReLU())
                        for edge_type in self.edge_types
                    },
                    aggregate='sum' 
                )
            elif model_args['sh_level']==2:
                self.convs = HeteroGraphConv(
                    {
                        edge_type: dglnn.GraphConv(self.model_args['hidden_size']+4,self.model_args['hidden_size'] ,activation=nn.LeakyReLU())
                        for edge_type in self.edge_types
                    },
                    aggregate='sum' 
                )
            elif model_args['sh_level']==4:
                self.convs = HeteroGraphConv(
                    {
                        edge_type: dglnn.GraphConv(self.model_args['hidden_size']+16,self.model_args['hidden_size'] ,activation=nn.LeakyReLU())
                        for edge_type in self.edge_types
                    },
                    aggregate='sum' 
                )
        else:
            self.convs = HeteroGraphConv(
                {
                    edge_type: dglnn.GraphConv(self.model_args['hidden_size'],self.model_args['hidden_size'] ,activation=nn.LeakyReLU())
                    for edge_type in self.edge_types
                },
                aggregate='sum' 
            )
        self.norm  = dglnn.EdgeWeightNorm(norm='right')

        if self.model_args['sh_after']:

            self.position_embedding_src = torch.load(f"{self.data_args['embedding_dir']}/{data_args['refinement_level']}_healpix_embeddings_level_{model_args['sh_level']}.pt").to('cuda')
            self.position_embedding_dst = torch.load(f"{self.data_args['embedding_dir']}/{node_type}_embeddings_{model_args['sh_level']}.pt").to('cuda')

            
        if self.model_args['sh_after']:
            if model_args['sh_level']==3:
                self.linears = nn.Linear(1+9,64)
            elif model_args['sh_level']==1:
                self.linears = nn.Linear(1+1,64)
            elif model_args['sh_level']==2:
                self.linears = nn.Linear(1+4,64)
            elif model_args['sh_level']==4:
                self.linears = nn.Linear(1+16,64)
        else:
            self.linears = nn.Linear(1,64)
        self.readout = AttentionReadout(
            input_size=self.model_args['hidden_size'],
            hidden_size=self.model_args['hidden_size']*2,
            output_size=1,
            dim_size=2,
            horizon=1,
            dim=1,
            fully_connected=False)

    def forward(self, x,g,sh_embedding=None,position_embedding_feature=None,position_embedding_healpix=None):

        node_type = self.data_args['feature']
        for t in range(self.output_length):


            g.nodes['healpix'].data[f'{node_type}_t{self.input_length+t}']=x[:,t,0]
            x_src={}
            x_dst={}
            x_src['healpix']=g.nodes['healpix'].data[f'{node_type}_t{self.input_length+t}']
            g.nodes[node_type].data[f't{self.input_length+t}'] = torch.zeros_like(g.nodes[node_type].data[f't{self.input_length+t}'])
            x_dst[node_type]=g.nodes[node_type].data[f't{self.input_length+t}']


            subgraph = g.edge_type_subgraph([('healpix', f't{self.input_length+t}_to_{node_type}', node_type) for t in range(self.output_length)])

            if self.model_args['sh_after']:
                src_position_embeddings = torch.cat([self.position_embedding_src for i in range(sh_embedding[node_type].shape[0])],dim=0)
                dst_position_embeddings = torch.cat([self.position_embedding_dst[sh_embedding[node_type][i].bool()] for i in range(sh_embedding[node_type].shape[0])],dim=0)

                x_src['healpix']=torch.concat([x_src['healpix'],src_position_embeddings*position_embedding_healpix],dim=1)
                x_dst[node_type]=torch.concat([x_dst[node_type],dst_position_embeddings*position_embedding_feature],dim=1)
            space_1 = self.convs(subgraph,(x_src,x_dst))[node_type].unsqueeze(0).unsqueeze(0).unsqueeze(0)#torch.Size([1, 1, 1, 20531, 64])
            if self.model_args['sh_after']:
                space_2 = self.linears(torch.concat([g.nodes[node_type].data[f't{self.input_length+t-1}'],dst_position_embeddings*position_embedding_feature],dim=1).unsqueeze(0).unsqueeze(0).unsqueeze(0))
            else:
                space_2 = self.linears(g.nodes[node_type].data[f't{self.input_length+t-1}'].unsqueeze(0).unsqueeze(0).unsqueeze(0))#torch.Size([1, 1, 1, 20531, 64])
            out_space = torch.cat([space_1,space_2],dim=1)
            out_space = rearrange(out_space, 'b s t ... -> b (s t) ...')#torch.Size([1, 2, 20531, 64])
            out, states, alpha = self.readout(out_space)
            g.nodes[node_type].data[f't{self.input_length+t}'] = out[0][0]

        return g