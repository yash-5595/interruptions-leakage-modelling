import torch
import os
from torch.utils.data import Dataset, DataLoader
from numpy.random import Generator, PCG64
import uuid
from functools import partial
from multiprocessing.pool import ThreadPool as Pool
from layers import RNNEncoder, AttentionDecoderCell, output_layer


import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

class EncoderDecoderWrapper():
    def __init__(self, in_channels, out_channels, sequence_len,rnn_hid_size, device,spatial_inp_size, output_size=1, teacher_forcing=0.3,learning_rate = 0.001):
        super().__init__()
        
        self.LR =learning_rate
        self.encoder = RNNEncoder(rnn_num_layers=1, input_feature_len=in_channels,
                                  sequence_len=sequence_len, hidden_size=rnn_hid_size).to(device)
                                  
        self.decoder_cell = AttentionDecoderCell( hidden_size = rnn_hid_size, 
                                                 sequence_len = sequence_len, out_size= out_channels, spatial_inp_size = spatial_inp_size).to(device)
        self.out_channels = out_channels
        self.output_size = output_size+1
        self.teacher_forcing = teacher_forcing
        self.sequence_len = sequence_len
        
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.LR)
        self.decoder_optimizer = torch.optim.Adam(self.decoder_cell.parameters(), lr=self.LR)
        self.device = device
        
    def train(self):
        
        self.encoder.train()
        self.decoder_cell.train()
        
    def eval(self):
        
        self.encoder.eval()
        self.decoder_cell.eval()
        
    def state_dict(self):
        return {

            'encoder': self.encoder.state_dict(),
            'decoder_cell': self.decoder_cell.state_dict()
        }
    
    def zero_grad(self):
        
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
    def optimizer_step(self):
        
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
    
    def load_state_dict(self, state_dict):
       
        self.encoder.load_state_dict(state_dict['encoder'])
        self.decoder_cell.load_state_dict(state_dict['decoder_cell'])

    def __call__(self, data):
#         gc_out = self.gcn_layer(data)
#         concat_x = torch.cat([local_batch.x.reshape(-1,7,self.sequence_len),gc_out], dim=IDX_CHANNEL)
#         logging.info(f"shitting inside the model")
        ybb = data['y'].float().to(self.device)
        yb = torch.zeros((ybb.shape[0],ybb.shape[1]+1,ybb.shape[2])).to(self.device)
        yb[:,1:,:] = ybb
        sig_timing = data['sig'].float().to(self.device)
        sim_params = data['params'].float().to(self.device)

        input_seq = torch.cat((data['x'], data['sig']), 2).float().to(self.device)
#         input_seq = concat_x.transpose(1, 2)
#         logging.info(f"input seq. {input_seq.shape}")

        encoder_output, encoder_hidden = self.encoder(input_seq)
        # logging.info(f"encoder output done")
        # logging.info(f"encoder output {encoder_output.shape} hidden {encoder_hidden.shape}")
        
        prev_hidden = encoder_hidden
        if torch.cuda.is_available():
            outputs = torch.zeros(input_seq.size(0), self.output_size, self.out_channels, device=self.device)
        else:
            outputs = torch.zeros(input_seq.size(0), self.output_size, self.out_channels)
        y_prev = sim_params[:,1].unsqueeze(-1)
#         logging.info(f"y_prev output {y_prev.shape}")
        outputs[:, 0,:] = sim_params[:,1].unsqueeze(-1)
#         logging.info(f"about to mf decode")
        for i in range(1,self.output_size):
#             logging.info(f"decoding step {i}")
            if (yb is not None) and (i > 0) and (torch.rand(1) < self.teacher_forcing):
                y_prev = yb[:, i-1,:]
#             logging.info(f"y_prev  {y_prev.shape}")
            rnn_output, prev_hidden = self.decoder_cell(encoder_output, prev_hidden, y_prev, sig_timing[:,i-1,:],i, sim_params)
#             logging.info(f"rnn output {rnn_output.shape}")
#             logging.info(f"prev hidden {prev_hidden.shape}")
            y_prev = rnn_output
            outputs[:, i,:] = rnn_output
#         logging.info(f"decoding done")
        return outputs[:,1:,:]