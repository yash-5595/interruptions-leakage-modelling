import os
# import optuna
import mlflow
import torch
from mlflow import pytorch
import numpy as np
import pandas as pd
from argparse import Namespace
from torch.utils.data import Dataset, DataLoader
import logging
from dataset import LeakageDataset
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
from torch.utils.data.sampler import SubsetRandomSampler
import pickle 




class Trainer(object):
    def __init__(self, model, device,exp_name,exp, loss_fn=None, 
                 optimizer=None, scheduler=None):

        # Set params
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = loss_fn
        self.exp_name = exp_name
        self.exp = exp
        
    def train_step(self, train_loader, epoch):
        self.model.train()
        num_batches = len(train_loader)
        # logging.info(f"num_batches {num_batches}")
        train_set_size = len(train_loader.dataset)
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # logging.info(f"batch idx {batch_idx}")
            # batch = batch.to(device)
            # print(f"batch_idx {batch_idx}")
            # logging.info(f"grad zero")
            
            self.model.zero_grad()
            # logging.info(f"model input done")
            output = self.model(batch)
            # logging.info(f"model output ** done")
            # print(f"output shape {output.shape}")
#             change to cumulative sum

            target = batch['y'].float().to(self.device)
#             target  = torch.cumsum(target, axis =1)
            # logging.info(f"calculating loss")
            loss = self.criterion(output, target)
            # logging.info(f" loss done")
#             loss = self.calc_loss( batch['y'].float().to(self.device), output)
            loss.backward()
            # logging.info(f"back prop done (******) ")
            train_loss += loss.item()
            # print(f"loss for this epoch {epoch} batch  {batch_idx} --> {loss.item()}")
            self.model.optimizer_step()
            # logging.info(f"optimizer step done (******) ")
            if batch_idx % 10 == 0:
                batch_size = len(batch['y'])
                print(f"Train Epoch: {epoch}  {batch_idx} {batch_idx * batch_size}/{train_set_size}") 

        avg_train_loss = train_loss / num_batches
        return avg_train_loss
    
    
    def eval_step(self,val_loader, epoch):
        self.model.eval()
        val_set_size = len(val_loader.dataset)
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data in val_loader:
                output = self.model(data)
                val_loss += self.criterion(output, data['y'].float().to(self.device)).item() # sum up batch loss         


        val_loss /=  len(val_loader)
        
        print(f"Test set: Average loss: {val_loss:.8f}\n")
        return val_loss
    
    def train(self, start_epoch, num_epochs, patience, train_dataloader, val_dataloader):
        best_val_loss = np.inf
        # logging.info(f"started training")
        for epoch in range(start_epoch, num_epochs):
            train_loss = self.train_step(train_dataloader, epoch)
            val_loss = self.eval_step(val_dataloader, epoch)
            
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model
                _patience = patience  # reset _patience
            else:
                _patience -= 1
#             if not _patience:  # 0
#                 print("Stopping early!")
#                 break
                
                
            
            # Tracking
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
            )
            if(epoch%2 == 0):
                
                DICT_SAVE = self.model.state_dict()
                torch.save(self.model.state_dict(), f'pthfiles/{self.exp_name}_epoch_{epoch}.pth')


            
            # Logging
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.7f}, "
                f"val_loss: {val_loss:.7f}, "
#                 f"lr: {self.optimizer.param_groups[0]['lr']:.2E}, "
                f"_patience: {_patience}"
            )
    
    def get_dataloaders(self,batch_size,store_path,processed_run_name,data_params, segment_name):
        dataset =  LeakageDataset(store_path,processed_run_name, segment_name, data_params)
        dataset_size = len(dataset)
        validation_split = 0.2
        shuffle_dataset = True
        random_seed= 42
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # store train and val indices for this experiment at experiments/EXP_NAME
        MY_DIR = f"experiments/{self.exp_name}"
        if not os.path.isdir(MY_DIR):
            os.makedirs(MY_DIR)
            logging.info(f"created folder : {MY_DIR} for storing train and val indices " )

        logging.info(f"TRAIN SET SIZE {len(train_indices)} VALIDATION SET SIZE {len(val_indices)}")
        with open(f'{MY_DIR}/train_indices.pkl', 'wb') as handle:
            pickle.dump(train_indices, handle)

        with open(f'{MY_DIR}/val_indices.pkl', 'wb') as handle:
            pickle.dump(val_indices, handle)

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                                sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
        return train_loader, validation_loader
            
 