import os
import torch
torch.autograd.set_detect_anomaly(True)
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning.pytorch as pl
import yaml
from models import ST_MSTGCN, ST_GraphTransformer,ST_STSGNN,ST_A3TGCN_sh,ST_GConvGRU,ST_DCRNN,ST_GConvLSTM,ST_GCLSTM,ST_DyGrAE,ST_TGCN,ST_A3TGCN,ST_MPNNLSTM,ST_AGCRN,ST_STConv
from models import ST_ReDyNet,time_and_graph_isotropic,ST_STGT,ST_GPS,time_and_graph_anisotropic,time_then_graph_isotropic,time_then_graph_anisotropic,hdtts_model
from utils import criterion
from dataset import dataset
import pickle
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from dataset.dataset import SpatialTemporalDataset,STDataset,STRealTimeDataset
import pdb
from thop import profile
import copy

class SpatialTemporal(pl.LightningModule):

    def __init__(
        self, 
        model_args,
        data_args,
        
    ):
        super(SpatialTemporal, self).__init__()
        self.save_hyperparameters()
        
        self.model_args = model_args
        self.data_args = data_args

        self.output_length=self.data_args['output_length']
        self.input_length=self.data_args['input_length']
        
        with open(f"{self.data_args['data_dir']}/climatology.npy", 'rb') as f:
            self.climatology = pickle.load(f)
        

        input_feature=2
        if 'DCRNN' == self.model_args['model_name']:
            self.model = ST_DCRNN.Model(node_features=self.data_args['input_length']+2,filters=self.model_args['hidden_size'],output_length=self.output_length)
        elif 'GConvGRU' == self.model_args['model_name']:
            self.model = ST_GConvGRU.Model(node_features=self.data_args['input_length']+2,filters=self.model_args['hidden_size'],output_length=self.output_length)
        elif 'GConvLSTM' == self.model_args['model_name']:
            self.model = ST_GConvLSTM.Model(node_features=self.data_args['input_length']+2,filters=self.model_args['hidden_size'],output_length=self.output_length)
        elif 'GCLSTM' == self.model_args['model_name']:
            self.model = ST_GCLSTM.Model(node_features=self.data_args['input_length']+2,filters=self.model_args['hidden_size'],output_length=self.output_length)
        elif 'DyGrAE' == self.model_args['model_name']:
            self.model = ST_DyGrAE.Model(node_features=self.data_args['input_length']+input_feature,filters=self.model_args['hidden_size'],output_length=self.output_length)
        elif 'TGCN' == self.model_args['model_name']:
            self.model = ST_TGCN.Model(node_features=self.data_args['input_length']+input_feature,filters=self.model_args['hidden_size'],output_length=self.output_length)
        elif 'STConv' == self.model_args['model_name']:
            self.model = ST_STConv.Model(node_features=self.data_args['input_length']+input_feature,filters=self.model_args['hidden_size'],output_length=self.output_length,feature=self.data_args['feature'])
        elif 'A3TGCN' == self.model_args['model_name']:
            self.model = ST_A3TGCN.Model(periods=1,node_features=self.data_args['input_length']+2,filters=self.model_args['hidden_size'],output_length=self.output_length)
        elif 'A3TGCN_sh' == self.model_args['model_name']:
            self.model = ST_A3TGCN_sh.Model(periods=self.data_args['input_length'],node_features=self.data_args['input_length']+2,filters=self.model_args['hidden_size'],output_length=self.output_length,feature=self.data_args['feature'])
        elif 'MPNNLSTM' == self.model_args['model_name']:
            self.model = ST_MPNNLSTM.Model(node_features=self.data_args['input_length']+2,filters=self.model_args['hidden_size'],output_length=self.output_length,feature=self.data_args['feature'])
        elif 'AGCRN' == self.model_args['model_name']:
            self.model = ST_AGCRN.Model(node_features=self.data_args['input_length'],filters=self.model_args['hidden_size'],output_length=self.output_length)
        elif 'MSTGCN' == self.model_args['model_name']:
            self.model = ST_MSTGCN.Model(node_features=self.data_args['input_length']+input_feature,filters=self.model_args['hidden_size'],feature=self.data_args['feature'])
        elif 'GTN' == self.model_args['model_name']:
            self.model = ST_GraphTransformer.Model(node_features=self.data_args['input_length']+input_feature,filters=self.model_args['hidden_size'],output_length=self.output_length)
        elif 'GPS' == self.model_args['model_name']:
            self.model = ST_GPS.Model(node_features=self.data_args['input_length']+input_feature,filters=self.model_args['hidden_size'],output_length=self.output_length)
        elif 'STAR' == self.model_args['model_name']:
            self.model = ST_STGT.Model(node_features=self.data_args['input_length']+input_feature,filters=self.model_args['hidden_size'],output_length=self.output_length)
        elif 'tasimp' == self.model_args['model_name']:
            self.model = time_and_graph_isotropic.TimeAndGraphIsoModel(input_size=self.data_args['input_length']+input_feature,horizon=1,n_nodes=None,output_size=self.output_length)
        elif 'tasamp' == self.model_args['model_name']:
            self.model = time_and_graph_anisotropic.TimeAndGraphAnisoModel(input_size=self.data_args['input_length']+input_feature,horizon=1,n_nodes=None,output_size=self.output_length)
        elif 'ttsimp' == self.model_args['model_name']:
            self.model = time_then_graph_isotropic.TimeThenGraphIsoModel(input_size=self.data_args['input_length']+input_feature,horizon=1,n_nodes=None,output_size=self.output_length,graph_layers=2)
        elif 'ttsamp' == self.model_args['model_name']:
            self.model = time_then_graph_anisotropic.TimeThenGraphAnisoModel(input_size=self.data_args['input_length']+input_feature,horizon=1,n_nodes=None,output_size=self.output_length)
        elif 'hdtts' == self.model_args['model_name']:
            self.model = hdtts_model.HDTTSModel(input_size=self.data_args['input_length']+input_feature,output_size=self.data_args['output_length'],hidden_size=self.model_args['hidden_size'],n_nodes=None,mp_method=["anisoconv"],pooling='kmis',feature=None)

        elif 'ReDyNet' == self.model_args['model_name']:
            self.model = ST_ReDyNet.Model(node_features=self.data_args['input_length']+input_feature,filters=self.model_args['hidden_size'],output_length=self.output_length,feature=self.data_args['feature'])
        elif 'DualCast' == self.model_args['model_name']:
            self.model = ST_STSGNN.Model(node_features=self.data_args['input_length']+input_feature,filters=self.model_args['hidden_size'],output_length=self.output_length,feature=self.data_args['feature'])
            
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
    
    def forward(self, graph):

            return self.model(torch.concatenate([graph.x,graph.latitudes,graph.longitudes],dim=1), graph.edge_index, graph.edge_attr)

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

    def training_step(self, graph, batch_idx):

        y_hat, *_ = self(graph)
        if self.output_length==1:
            y_hat=y_hat[:,0]
        loss = self.loss(y_hat,graph.y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.data_args['batch_size'])
        return loss
    def validation_step(self, graph, batch_idx):

        y_hat, *_ = self(graph)
        # pdb.set_trace()
        if self.output_length==1:
            y_hat=y_hat[:,0]
        loss = self.loss(y_hat,graph.y) 
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.data_args['batch_size'])
        return loss
    def test_step(self, graph, batch_idx):

        y_hat, *_ = self(graph)
        if self.output_length==1:
            y_hat=y_hat[:,0]
        graph.y = (graph.y*self.climatology[self.data_args['feature']]['std']) + self.climatology[self.data_args['feature']]['mean']
        y_hat = (y_hat*self.climatology[self.data_args['feature']]['std']) + self.climatology[self.data_args['feature']]['mean']
        
        rmse_loss = self.rmse_loss(y_hat,graph.y)
        mae_loss = self.mae_loss(y_hat,graph.y)
        loss = self.loss(y_hat,graph.y)


        self.log("rmse_loss", rmse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.data_args['batch_size'])
        self.log("mse_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.data_args['batch_size'])
        self.log("mae_loss", mae_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.data_args['batch_size'])
        if self.output_length!=1:
            for day in range(self.output_length):
                each_loss=self.loss(y_hat[:,day],graph.y[:,day])
                assert not torch.isnan(each_loss).any()
                self.log(f"day {day+1}",each_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.data_args['batch_size'])
        return loss
    def predict_step(self,  graph, batch_idx):
        
        if len(graph.y.shape) ==1:
            y_hat, *_ = self(graph)#graph.x.shape [11030,1]
            pdb.set_trace()
            if self.output_length==1:
                y_hat=y_hat[:,0]#y_hat.shape torch.Size([11030])
            # pdb.set_trace()
            graph.y = (graph.y*self.climatology[self.data_args['feature']]['std']) + self.climatology[self.data_args['feature']]['mean']
            y_hat = (y_hat*self.climatology[self.data_args['feature']]['std']) + self.climatology[self.data_args['feature']]['mean']
            return graph.y,y_hat,graph.latitudes,graph.longitudes,graph.mask
        else:
            
            predict=[]
            for i in range(graph.y.shape[1]):
                input_graph = copy.deepcopy(graph)
                if i ==0:
                    pass
                else:
                    input_graph.x = y_hat
                # pdb.set_trace()
                y_hat, *_ = self(input_graph)
                predict.append(y_hat)
            # pdb.set_trace()
            predict = torch.concatenate(predict,dim=1)
            # pdb.set_trace()
            graph.y = (graph.y*self.climatology[self.data_args['feature']]['std']) + self.climatology[self.data_args['feature']]['mean']
            predict = (predict*self.climatology[self.data_args['feature']]['std']) + self.climatology[self.data_args['feature']]['mean']
            total_mae_loss = self.mae_loss(predict,graph.y)
            total_mse_loss = self.loss(predict,graph.y)
            mae_loss =[]
            loss =[]
            for k in range( predict.shape[1]):
                rmse_loss = self.rmse_loss(predict[:,k],graph.y[:,k])
                mae_loss.append(self.mae_loss(predict[:,k],graph.y[:,k]))
                loss.append(self.loss(predict[:,k],graph.y[:,k]))


            return mae_loss,loss,total_mae_loss,total_mse_loss,graph.latitudes,graph.longitudes,graph.mask
                

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
        self.train_dataset=STDataset(data_dir=self.data_args['data_dir'],input_length=self.data_args['input_length'],output_length=self.data_args['output_length'],feature=self.data_args['feature'],split='train',data_args=self.data_args)
        self.val_dataset=STDataset(data_dir=self.data_args['data_dir'],input_length=self.data_args['input_length'],output_length=self.data_args['output_length'],feature=self.data_args['feature'],split='val',data_args=self.data_args)
        self.test_dataset=STDataset(data_dir=self.data_args['data_dir'],input_length=self.data_args['input_length'],output_length=self.data_args['output_length'],feature=self.data_args['feature'],split='test',data_args=self.data_args)
        self.predict_dataset = STDataset(data_dir=self.data_args['data_dir'],input_length=self.data_args['input_length'],output_length=3,feature=self.data_args['feature'],split='test',data_args=self.data_args)
        # self.train_dataset=STRealTimeDataset(data_dir=self.data_args['data_dir'],input_length=self.data_args['input_length'],output_length=self.data_args['output_length'],feature=self.data_args['feature'],split='train',data_args=self.data_args)
        # self.val_dataset=STRealTimeDataset(data_dir=self.data_args['data_dir'],input_length=self.data_args['input_length'],output_length=self.data_args['output_length'],feature=self.data_args['feature'],split='val',data_args=self.data_args)
        # self.test_dataset=STRealTimeDataset(data_dir=self.data_args['data_dir'],input_length=self.data_args['input_length'],output_length=self.data_args['output_length'],feature=self.data_args['feature'],split='test',data_args=self.data_args)
        # self.predict_dataset = STRealTimeDataset(data_dir=self.data_args['data_dir'],input_length=self.data_args['input_length'],output_length=self.data_args['output_length'],feature=self.data_args['feature'],split='train',data_args=self.data_args)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.data_args['batch_size'], shuffle=True, num_workers=self.model_args['num_workers'])
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.data_args['batch_size'], shuffle=False, num_workers=self.model_args['num_workers'])
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.data_args['batch_size'], shuffle=False, num_workers=self.model_args['num_workers'])
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, 
                          batch_size=self.data_args['batch_size'], shuffle=False, num_workers=self.model_args['num_workers'])