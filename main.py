import os
from models import model,model_st
import argparse
from pathlib import Path
import yaml
from lightning.pytorch.loggers import WandbLogger
import torch
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping
import pandas as pd
import pdb

def main(args):
    pl.seed_everything(args.seed)
    if args.model in ['STConv','DyGrAE','STAR','GTN','MPNNLSTM','GPS','tasimp','tasamp','ttsimp','ttsamp','ReDyNet','DualCast','GCLSTM','DCRNN','GConvGRU','GConvLSTM','TGCN','AGCRN','MSTGCN','GWNET','hdtts']:
        config_filepath = 'configs/spatial_temporal.yaml'
        with open(config_filepath, 'r') as config_filepath:
            hyperparams = yaml.load(config_filepath, Loader=yaml.FullLoader)
        model_args = hyperparams['model_args']
        data_args = hyperparams['data_args']
        model_args['model_name']=args.model
        data_args['input_length'] = args.input_length
        data_args['output_length'] = args.output_length
        data_args['feature']=args.feature
        baseline = model_st.SpatialTemporal(model_args=model_args,data_args=data_args)

    elif 'MIGN' in args.model:
        config_filepath = 'configs/hgnn_gcn_edge.yaml'
        with open(config_filepath, 'r') as config_filepath:
            hyperparams = yaml.load(config_filepath, Loader=yaml.FullLoader)
        model_args = hyperparams['model_args']
        model_args['model_name']=args.model
        model_args['sh_before']=args.sh_before
        model_args['sh_after']=args.sh_after
        model_args['sh_level']=args.sh_level
        data_args = hyperparams['data_args']
        data_args['feature'] = args.feature
        data_args['refinement_level'] = 3
        data_args['neighbor'] = 10
        data_args['input_length'] = args.input_length
        data_args['output_length'] = args.output_length
        baseline = model.HGNNModel(model_args=model_args,data_args=data_args)
    baseline.setup()
    
    log_dir = Path('logs') / model_args['model_name']
    wandb_logger = WandbLogger(project=args.project, name=model_args['model_name'], save_dir=str(log_dir))

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min')


    early_stop_callback = EarlyStopping(
        monitor='val_loss',       # The metric to monitor
        patience=3,               # Number of epochs to wait for improvement
        verbose=True,             # Display messages when early stopping is triggered
        mode='min',               # 'min' means stop when the value decreases
        check_on_train_epoch_end=False  # Whether to check after every training epoch
    )
    trainer = pl.Trainer(
        devices=1,
        accelerator='gpu',
        strategy='auto',#
        max_epochs=40,
        logger=wandb_logger,
        callbacks=[checkpoint_callback,early_stop_callback]
     )

    trainer.fit(baseline)
    trainer.test(baseline, ckpt_path="best")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='hdtts')#tasamp,ttsamp,dygrae,stgcn
    parser.add_argument('--pooling',default='kmis')
    parser.add_argument('--feature',default='WDSP',help="['MAX','MIN','DEWP',  'SLP', 'WDSP', 'MXSPD']")
    parser.add_argument('--project',default='flops')
    parser.add_argument('--input',default='0')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--input_length',type=int,default=1)
    parser.add_argument('--output_length',type=int,default=1)
    parser.add_argument('--ratio',type=int,default=0)
    parser.add_argument('--sh_before',action='store_false', default=True)
    parser.add_argument('--sh_after',action='store_false', default=True)
    parser.add_argument('--sh_level',type=int, default=3)
    parser.add_argument('--refinement_level',type=int, default=3)
    args = parser.parse_args()
    main(args)