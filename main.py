import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import argparse
from pathlib import Path
import yaml
from models import model
from lightning.pytorch.loggers import WandbLogger
import torch
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping


def main(args):

    pl.seed_everything(args.seed)

    config_filepath = 'configs/hgnn_gcn_edge.yaml'
    with open(config_filepath, 'r') as config_filepath:
        hyperparams = yaml.load(config_filepath, Loader=yaml.FullLoader)
    model_args = hyperparams['model_args']
    model_args['model_name']=args.model
    model_args['sh_before']=args.sh_before
    model_args['sh_after']=args.sh_after
    model_args['sh_level']=args.sh_level
    data_args = hyperparams['data_args']
    data_args['vars'] = [args.feature]
    data_args['predict_vars'] = [args.feature]
    baseline = model.HGNNModel(model_args=model_args,data_args=data_args)


    baseline.setup()
    
    # Initialize training
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
        max_epochs=model_args['epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback,early_stop_callback]
     )

    trainer.fit(baseline)
    trainer.test(baseline, ckpt_path="best")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='hgnn_gcn_edge_wo_sh')
    parser.add_argument('--feature',default='air_temperature_max',help="['accumulated_precipitation','air_temperature_max','air_temperature_mean',  'air_temperature_min', 'fresh_snow', 'snow_depth','wind_speed']")
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--project',default='MIGN')
    parser.add_argument('--sh_before',action='store_true', default=False)
    parser.add_argument('--sh_after',action='store_true', default=False)
    parser.add_argument('--sh_level',type=int, default=3)
    args = parser.parse_args()
    main(args)