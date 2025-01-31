import os
import torch
torch.autograd.set_detect_anomaly(True)
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning.pytorch as pl
import yaml
from dgl.dataloading import GraphDataLoader
from torch.utils.data import DataLoader
from models.hgnn_gcn_edge_wo_sh import HGNN_GCN_EDGE_WO_SH
from utils import criterion
from dataset import dataset
import dgl
import pickle
from tqdm import tqdm
from dataset.dataset import TemporalHeterogeneousDataset
import copy
import pdb

class HGNNModel(pl.LightningModule):

    def __init__(
        self, 
        model_args,
        data_args,
        
    ):
        super(HGNNModel, self).__init__()
        self.save_hyperparameters()
        
        self.model_args = model_args
        self.data_args = data_args
        
        with open(self.data_args['climatology_dir'], 'rb') as f:
            self.climatology = pickle.load(f)
        self.output_length=self.data_args['output_length']
        self.input_length=self.data_args['input_length']
        # Initialize model
        # ocean_vars = self.data_args.get('ocean_vars', [])
        hidden_size = self.model_args['hidden_size'] 
        
        if 'hgnn_gcn_edge_wo_sh' == self.model_args['model_name']:
            self.model = HGNN_GCN_EDGE_WO_SH(model_args=self.model_args,data_args=self.data_args)
        self.loss = self.init_loss_fn()

        self.mae_loss = self.init_loss_mae_fn()

        self.rmse_loss = self.init_loss_rmse_fn()

    def init_loss_mae_fn(self):
        loss = criterion.MAE()
        return loss

    def init_loss_fn(self):
        loss = criterion.MSE()
        return loss
    
    def init_loss_rmse_fn(self):
        loss = criterion.RMSE()
        return loss
    
    def forward(self, graph,timestamp,sh_embedding):

        return self.model(graph,timestamp,sh_embedding)

    def training_step(self, batch, batch_idx):
        loss = 0
        graph=batch[0]
        concatenated_features = []
        for node_type in self.data_args['predict_vars']:
            features = [graph.nodes[node_type].data[f't{self.input_length+t}'] for t in range(self.output_length)]
            concatenated_feature = torch.stack(features, dim=1)#(batch*node,output_length,1)
            concatenated_features.append(concatenated_feature)
        label=torch.concatenate(concatenated_features,dim=0)

        output_graph = self(graph,batch[1],batch[2])
        output_concatenated_features = []
        for node_type in self.data_args['predict_vars']:
            features = [output_graph.nodes[node_type].data[f't{self.input_length+t}'] for t in range(self.output_length)]
            output_concatenated_feature = torch.stack(features, dim=1)

            output_concatenated_features.append(output_concatenated_feature)
        predict=torch.concatenate(output_concatenated_features,dim=0)

        loss=self.loss(predict,label)  
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.data_args['batch_size'])

        return loss
    def custom_relu(self, x, threthold,default):
        return torch.where(x < threthold, torch.full_like(x, default), x)
    def validation_step(self, batch, batch_idx):
        loss = 0
        graph=batch[0]
        concatenated_features = []
        for node_type in self.data_args['predict_vars']:
            features = [graph.nodes[node_type].data[f't{self.input_length+t}'] for t in range(self.output_length)]
            concatenated_feature = torch.stack(features, dim=1)
            concatenated_features.append(concatenated_feature)
        label=torch.concatenate(concatenated_features,dim=0)

        output_graph = self(graph,batch[1],batch[2])
        output_concatenated_features = []
        for node_type in self.data_args['predict_vars']:
            features = [output_graph.nodes[node_type].data[f't{self.input_length+t}'] for t in range(self.output_length)]
            output_concatenated_feature = torch.stack(features, dim=1)
            output_concatenated_features.append(output_concatenated_feature)
        predict=torch.concatenate(output_concatenated_features,dim=0)

        loss=self.loss(predict,label)  
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.data_args['batch_size'])
        return loss
    def test_step(self, batch, batch_idx):
        loss = 0
        graph=batch[0]
        concatenated_features = []
        for node_type in self.data_args['predict_vars']:
            features = [graph.nodes[node_type].data[f't{self.input_length+t}'] for t in range(self.output_length)]
            concatenated_feature = torch.stack(features, dim=1)
            concatenated_feature = (concatenated_feature*self.climatology[node_type]['std']) + self.climatology[node_type]['mean']
            concatenated_features.append(concatenated_feature)
        label=torch.concatenate(concatenated_features,dim=0)
        
        output_graph = self(graph,batch[1],batch[2])
        output_concatenated_features = []
        for node_type in self.data_args['predict_vars']:
            features = [output_graph.nodes[node_type].data[f't{self.input_length+t}'] for t in range(self.output_length)]
            output_concatenated_feature = torch.stack(features, dim=1)
            # if node_type=="accumulated_precipitation":
            #     output_concatenated_feature = self.custom_relu(output_concatenated_feature,0.015,-0.3261)
            # elif node_type=="fresh_snow":
            #     output_concatenated_feature = self.custom_relu(output_concatenated_feature,0.025,-0.1451)
            # else:
            #     raise KeyError
            output_concatenated_feature = (output_concatenated_feature*self.climatology[node_type]['std']) + self.climatology[node_type]['mean']
            output_concatenated_features.append(output_concatenated_feature)
        predict=torch.concatenate(output_concatenated_features,dim=0)

        loss=self.loss(predict,label) 
        # mae_loss = self.mae_loss(predict,label)   
        # self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.data_args['batch_size'])
        # self.log("mae_loss", mae_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.data_args['batch_size'])
        assert not torch.isnan(loss).any()
        flag=0
        for i,node_type in enumerate(self.data_args['predict_vars']):
            for day in range(self.output_length):
                each_loss=self.loss(predict[flag:flag+concatenated_features[i].shape[0],day],label[flag:flag+concatenated_features[i].shape[0],day])
                each_loss_mae=self.mae_loss(predict[flag:flag+concatenated_features[i].shape[0],day],label[flag:flag+concatenated_features[i].shape[0],day])
                each_loss_rmse=self.rmse_loss(predict[flag:flag+concatenated_features[i].shape[0],day],label[flag:flag+concatenated_features[i].shape[0],day])
                assert not torch.isnan(each_loss).any()
                self.log(f"{node_type} in day {day+1} rmse",each_loss_rmse, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.data_args['batch_size'])
                self.log(f"{node_type} in day {day+1} mse",each_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.data_args['batch_size'])
                self.log(f"{node_type} in day {day+1} mae",each_loss_mae, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.data_args['batch_size'])
            flag=flag+concatenated_features[i].shape[0]

        return loss
    def predict_step(self,  batch, batch_idx):
        loss = 0
        graph=batch[0]
        concatenated_features = []
        for node_type in self.data_args['predict_vars']:
            features = [graph.nodes[node_type].data[f't{self.input_length+t}'] for t in range(self.output_length)]
            concatenated_feature = torch.stack(features, dim=1)
            concatenated_feature = (concatenated_feature*self.climatology[node_type]['std']) + self.climatology[node_type]['mean']
            concatenated_features.append(concatenated_feature)
        label=torch.concatenate(concatenated_features,dim=0)
        
        output_graph = self(graph,batch[1],batch[2])
        output_concatenated_features = []
        for node_type in self.data_args['predict_vars']:
            features = [output_graph.nodes[node_type].data[f't{self.input_length+t}'] for t in range(self.output_length)]
            output_concatenated_feature = torch.stack(features, dim=1)
            # if node_type=="accumulated_precipitation":
            #     output_concatenated_feature = self.custom_relu(output_concatenated_feature,0.015,-0.3261)
            # elif node_type=="fresh_snow":
            #     output_concatenated_feature = self.custom_relu(output_concatenated_feature,0.025,-0.1451)
            # else:
            #     raise KeyError
            output_concatenated_feature = (output_concatenated_feature*self.climatology[node_type]['std']) + self.climatology[node_type]['mean']
            output_concatenated_features.append(output_concatenated_feature)
        predict=torch.concatenate(output_concatenated_features,dim=0)
        return predict[:,0,0],label[:,0,0], graph.nodes[node_type].data['latitude'][:,0], graph.nodes[node_type].data['longitude'][:,0], None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.model_args['learning_rate'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': CosineAnnealingLR(optimizer, T_max=self.model_args['t_max'], eta_min=self.model_args['learning_rate'] / 10),
                'interval': 'epoch',
            }
        }


    def setup(self, stage=None):
        self.train_dataset=TemporalHeterogeneousDataset(data_dir=self.data_args['data_dir'],years=self.data_args['train_years'],input_length=self.data_args['input_length'],output_length=self.data_args['output_length'],climatology=self.climatology)
        self.val_dataset=TemporalHeterogeneousDataset(data_dir=self.data_args['data_dir'],years=self.data_args['val_years'],input_length=self.data_args['input_length'],output_length=self.data_args['output_length'],climatology=self.climatology)
        self.test_dataset=TemporalHeterogeneousDataset(data_dir=self.data_args['data_dir'],years=self.data_args['test_years'],input_length=self.data_args['input_length'],output_length=self.data_args['output_length'],climatology=self.climatology)
        self.predict_dataset = TemporalHeterogeneousDataset(data_dir=self.data_args['data_dir'],years=self.data_args['test_years'],input_length=self.data_args['input_length'],output_length=self.data_args['output_length'],climatology=self.climatology)
        

    def train_dataloader(self):
        return GraphDataLoader(self.train_dataset, 
                          batch_size=self.data_args['batch_size'], shuffle=True, num_workers=self.model_args['num_workers'])
    def val_dataloader(self):
        return GraphDataLoader(self.val_dataset, 
                          batch_size=self.data_args['batch_size'], shuffle=False, num_workers=self.model_args['num_workers'])
    def test_dataloader(self):
        return GraphDataLoader(self.test_dataset, 
                          batch_size=self.data_args['batch_size'], shuffle=False, num_workers=self.model_args['num_workers'])
    def predict_dataloader(self):
        return GraphDataLoader(self.predict_dataset, 
                          batch_size=1, shuffle=False, num_workers=self.model_args['num_workers'])