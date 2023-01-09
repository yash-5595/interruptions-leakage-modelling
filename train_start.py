import numpy as np
# import optuna
import mlflow
import torch
from mlflow import pytorch
from argparse import Namespace
from torch.utils.data import Dataset, DataLoader
from pprint import pformat
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
from config import exp_dict_all

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

from trainer import Trainer
from encoder_decoder import EncoderDecoderWrapper

SEGMENT = "segment_1460_1465"

params = Namespace(batch_size=50,
   in_channels = 2*(len(exp_dict_all[SEGMENT]['inputs']['phases'])), out_channels = 1, sequence_len = 20,rnn_hid_size = 50, output_size=2, teacher_forcing=0.3,
    lr=1e-4,
    num_epochs=1000,
    patience=10,
TIME_SLICE_NAME = 'exemplarid',
store_path = '/blue/ranka/yashaswikarnati/interruption/leakage_modelling/train_data/',
                   processed_run_name = 'all_segments_sr436',  data_params = {'inp_agg_level':4,
                          'oup_agg_level':20,
                      'oup_window_use': (0,40)}, segment_name = SEGMENT
)


logging.info(f"{torch.cuda.is_available()}, {torch.cuda.get_device_name(0)}")
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

# Set tracking URI
MODEL_REGISTRY = Path("experiments/")
Path(MODEL_REGISTRY).mkdir(exist_ok=True) # create experiments dir
mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))
EXP_NAME = SEGMENT
mlflow.set_experiment(experiment_name=EXP_NAME)
LOAD_PRETRAINED = False
pretrained_epoch = 0



print(f"starting exp name {EXP_NAME}")
with mlflow.start_run(run_name=EXP_NAME) as run:
    # logging.info(f"model initialized")
    model =  EncoderDecoderWrapper(in_channels = params.in_channels, out_channels = params.out_channels,
                                    sequence_len = params.sequence_len,rnn_hid_size = params.rnn_hid_size, device = device, spatial_inp_size= 4 + int(params.in_channels/2),output_size=params.output_size,
                                    teacher_forcing=params.teacher_forcing,learning_rate = params.lr)
    # logging.info(f"model loaded")
#     model = nn.DataParallel(model).cuda()
    
    if(LOAD_PRETRAINED):
        print(f"loading pretrained model {EXP_NAME}_epoch_{pretrained_epoch}.pth")
        model.load_state_dict(torch.load(f'pthfiles/{EXP_NAME}_epoch_{pretrained_epoch}.pth'))

    trainer_obj =  Trainer(
        model  = model,
        device = device,
        exp_name = EXP_NAME,
        exp = 'inflow',
        loss_fn = torch.nn.MSELoss()
    )

    mlflow.log_params(vars(params))
    train_loader, val_loader = trainer_obj.get_dataloaders(params.batch_size,params.store_path,params.processed_run_name,params.data_params, params.segment_name)
    logging.info(f"train and val loader objects created")
    trainer_obj.train(pretrained_epoch, params.num_epochs, params.patience, train_loader, val_loader)