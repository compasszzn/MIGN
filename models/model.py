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
from dataset.dataset import TemporalHeterogeneousDataset,TemporalHeterogeneousRealTimeDataset
import copy
import pdb
from thop import profile
from torchstat import stat
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
        # flops, params = profile(self.model, inputs=(graph,timestamp,sh_embedding))
        return self.model(graph,timestamp,sh_embedding)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.batch_start_time = torch.cuda.Event(enable_timing=True)
        self.batch_end_time = torch.cuda.Event(enable_timing=True)
        self.batch_start_time.record()

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.batch_end_time.record()
        torch.cuda.synchronize()
        elapsed_time = self.batch_start_time.elapsed_time(self.batch_end_time) / 1000.0
        self.log("train_full_step_time", elapsed_time, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.test_batch_start_time = torch.cuda.Event(enable_timing=True)
        self.test_batch_end_time = torch.cuda.Event(enable_timing=True)
        self.test_batch_start_time.record()

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.test_batch_end_time.record()
        torch.cuda.synchronize()
        elapsed_time = self.test_batch_start_time.elapsed_time(self.test_batch_end_time) / 1000.0  # 单位为秒
        self.log("test_step_time", elapsed_time, on_step=True, prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        loss = 0
        graph=batch[0]
        concatenated_features = []
        node_type = self.data_args['feature']
        features = [graph.nodes[node_type].data[f't{self.input_length+t}'] for t in range(self.output_length)]
        concatenated_feature = torch.stack(features, dim=1)#(batch*node,output_length,1)
        concatenated_features.append(concatenated_feature)
        label=torch.concatenate(concatenated_features,dim=0)

        output_graph = self(graph,batch[1],batch[2])
        output_concatenated_features = []
        node_type = self.data_args['feature']
        features = [output_graph.nodes[node_type].data[f't{self.input_length+t}'] for t in range(self.output_length)]
        output_concatenated_feature = torch.stack(features, dim=1)
        output_concatenated_features.append(output_concatenated_feature)
        predict=torch.concatenate(output_concatenated_features,dim=0)
        # pdb.set_trace()
        loss=self.loss(predict,label)  
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.data_args['batch_size'])

        return loss
    def custom_relu(self, x, threthold,default):
        return torch.where(x < threthold, torch.full_like(x, default), x)
    def validation_step(self, batch, batch_idx):
        loss = 0
        graph=batch[0]
        concatenated_features = []
        node_type = self.data_args['feature']
        features = [graph.nodes[node_type].data[f't{self.input_length+t}'] for t in range(self.output_length)]
        concatenated_feature = torch.stack(features, dim=1)
        concatenated_features.append(concatenated_feature)
        label=torch.concatenate(concatenated_features,dim=0)

        output_graph = self(graph,batch[1],batch[2])
        output_concatenated_features = []
        node_type = self.data_args['feature']
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
        node_type = self.data_args['feature']
        features = [graph.nodes[node_type].data[f't{self.input_length+t}'] for t in range(self.output_length)]
        concatenated_feature = torch.stack(features, dim=1)
        concatenated_feature = (concatenated_feature*self.climatology[node_type]['std']) + self.climatology[node_type]['mean']
        concatenated_features.append(concatenated_feature)
        label=torch.concatenate(concatenated_features,dim=0)
        
        output_graph = self(graph,batch[1],batch[2])
        output_concatenated_features = []
        node_type = self.data_args['feature']
        features = [output_graph.nodes[node_type].data[f't{self.input_length+t}'] for t in range(self.output_length)]
        output_concatenated_feature = torch.stack(features, dim=1)
        output_concatenated_feature = (output_concatenated_feature*self.climatology[node_type]['std']) + self.climatology[node_type]['mean']
        output_concatenated_features.append(output_concatenated_feature)
        predict=torch.concatenate(output_concatenated_features,dim=0)


        loss=self.loss(predict,label) 
        rmse_loss = self.rmse_loss(predict,label)
        mae_loss = self.mae_loss(predict,label)
        self.log("rmse_loss", rmse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.data_args['batch_size'])
        self.log("mse_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.data_args['batch_size'])
        self.log("mae_loss", mae_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.data_args['batch_size'])
        for day in range(self.output_length):
            each_loss=self.loss(predict[:,day],label[:,day])
            self.log(f"day {day+1}",each_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.data_args['batch_size'])


        return loss

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
        self.train_dataset=TemporalHeterogeneousDataset(data_dir=self.data_args['data_dir'],years=self.data_args['train_years'],input_length=self.data_args['input_length'],output_length=self.data_args['output_length'],climatology=self.climatology,data_args=self.data_args)
        self.val_dataset=TemporalHeterogeneousDataset(data_dir=self.data_args['data_dir'],years=self.data_args['val_years'],input_length=self.data_args['input_length'],output_length=self.data_args['output_length'],climatology=self.climatology,data_args=self.data_args)
        self.test_dataset=TemporalHeterogeneousDataset(data_dir=self.data_args['data_dir'],years=self.data_args['test_years'],input_length=self.data_args['input_length'],output_length=self.data_args['output_length'],climatology=self.climatology,data_args=self.data_args)

    def train_dataloader(self):
        return GraphDataLoader(self.train_dataset, 
                          batch_size=self.data_args['batch_size'], shuffle=True, num_workers=self.model_args['num_workers'])
    def val_dataloader(self):
        return GraphDataLoader(self.val_dataset, 
                          batch_size=self.data_args['batch_size'], shuffle=False, num_workers=self.model_args['num_workers'])
    def test_dataloader(self):
        return GraphDataLoader(self.test_dataset, 
                          batch_size=self.data_args['batch_size'], shuffle=False, num_workers=self.model_args['num_workers'])
