import torch
import os
from torch.utils.data import Dataset, DataLoader
from numpy.random import Generator, PCG64
import uuid
from functools import partial
import pickle
import numpy as np

import random


def loop_inplace_sum(arrlist):
    # assumes len(arrlist) > 0
    sum = arrlist[0].copy()
    for a in arrlist[1:]:
        sum += a
    return sum

def return_aggregated_x(X_batch, j):
    return X_batch.reshape(-1,j).sum(axis=1).reshape(X_batch.shape[0],-1)


class LeakageDataset(Dataset):
    def __init__(self, store_path,processed_run_name,segment_name,  data_params):
        
        self.store_path = store_path
        self.processed_data_path = os.path.join(self.store_path, processed_run_name,segment_name)
        self.samples_name = 'exemplar_'
        self.all_files = os.listdir(self.processed_data_path)
        
        
#         all processing constants
        self.inp_agg_level = data_params['inp_agg_level']
        self.oup_agg_level = data_params['oup_agg_level']
        self.oup_window_use = data_params['oup_window_use']
        
        
        
        
    def __len__(self):

        return len(os.listdir(self.processed_data_path))
    
    
    @property
    def raw_file_names(self):
        return os.listdir(self.processed_data_path)

    @property
    def processed_file_names(self):
        return os.listdir(self.processed_data_path)
    
    
    def __getitem__(self, idx):
        random_choice = random.choice(self.all_files)
        data = torch.load(os.path.join(self.processed_data_path,random_choice ))        
        x,sig,y,params, timestamp = self.return_isc_x_y(data)
        
        time_split_arr = [timestamp.year, timestamp.month, timestamp.day,timestamp.hour, timestamp.minute, timestamp.second ]
        return {'x':x.transpose(1,0),'sig':sig.transpose(1,0),'y':y.reshape(-1,1) ,'params':params, 'timestamp':time_split_arr}
    
    
    
    def return_isc_x_y(self, data):
        inp_arr = [return_aggregated_x(x.reshape(1,-1),self.inp_agg_level ) for x in data['inp'] ]
        oup_inp =return_aggregated_x(data['oup'][self.oup_window_use[0]:self.oup_window_use[1]].reshape(1,-1),self.oup_agg_level)
        oup_arr = return_aggregated_x(data['oup'][self.oup_window_use[1]:].reshape(1,-1),self.oup_agg_level)
        sig_timing = [return_aggregated_x(np.array(data['sig'][k]).reshape(1,-1),self.inp_agg_level ) for k in data['sig']]
        tod_dow= np.array([data['hour'], data['day_of_week']])
        
        timestamp = data['timestamp']
        
        x = np.concatenate(inp_arr,axis=0)
        sig = np.concatenate(sig_timing,axis=0)
        y  = oup_arr.reshape(-1)
        params = np.concatenate([oup_inp.reshape(-1),tod_dow ],axis=0)
        
        
        return x,sig,y,params, timestamp

    


        
    
    
    
    
            
            