import copy
import logging
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.e2e_common import to_cuda, th_accuracy
from data.lm_data_loader import ParallelSequentialIterator


def zoneout(new_h, new_c, h, c, h_keep, c_keep, is_training):
    mask_c = torch.ones_like(c)
    mask_h = torch.ones_like(h)
    
    c_dropout = nn.Dropout(1 - c_keep)
    h_dropout = nn.Dropout(1 - h_keep)
    
    if is_training:
      	mask_c = c_dropout(mask_c)
      	mask_h = h_dropout(mask_h)
    
    mask_c *= c_keep
    mask_h *= h_keep
    
    h = new_h * mask_h + (-mask_h + 1.) * h
    c = new_c * mask_c + (-mask_c + 1.) * c
    
    return h, c
 
 
class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y        
        
        
class FSRNNLM(nn.Module):
    def __init__(self, n_vocab, emb_size, fast_layers, fast_cell_size, slow_cell_size, 
                 zoneout_keep_h, zoneout_keep_c, dropout_rate=0.5, embed_vecs_init=None):
        super(FSRNNLM, self).__init__()        
        self.n_vocab = n_vocab         
        self.emb_size = emb_size                     
        self.dropout_rate = dropout_rate 
        self.fast_layers = fast_layers
        self.F_size = fast_cell_size
        self.S_size = slow_cell_size
        self.zoneout_keep_h = zoneout_keep_h
        self.zoneout_keep_c = zoneout_keep_c
        
        self.embed = torch.nn.Embedding(self.n_vocab, self.emb_size)
        self.fast_cells = torch.nn.ModuleList()       
        self.fast_cells += [torch.nn.LSTMCell(emb_size, fast_cell_size)]
        for l in range(1, self.fast_layers):
            self.fast_cells += [torch.nn.LSTMCell(fast_cell_size, fast_cell_size)]                 
        self.slow_cell  = torch.nn.LSTMCell(fast_cell_size, slow_cell_size)
        
        self.d0 = torch.nn.Dropout(dropout_rate)
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.d3 = torch.nn.Dropout(dropout_rate)
        self.out = torch.nn.Linear(fast_cell_size, n_vocab)
        
        # initialize parameters from uniform distribution
        for param in self.parameters():
            param.data.uniform_(-0.1, 0.1)
                   
        if embed_vecs_init is not None: 
            self.embed.weight.data.copy_(torch.from_numpy(embed_vecs_init))
            logging.info('load embed_vecs_init, shape ' + str(embed_vecs_init.shape[0]) + ' ' + str(embed_vecs_init.shape[1]))        
    
    def zero_state(self, batchsize, n_units):
        return torch.zeros(batchsize, n_units).float()

    def forward(self, state, x):
        if state is None:
            state = {
                'F_h': to_cuda(self, self.zero_state(x.size(0), self.F_size)),
                'F_c': to_cuda(self, self.zero_state(x.size(0), self.F_size)),
                'S_h': to_cuda(self, self.zero_state(x.size(0), self.S_size)),
                'S_c': to_cuda(self, self.zero_state(x.size(0), self.S_size))
            }
        inputs = self.embed(x)
        inputs = self.d0(inputs)   
        F_h = state['F_h']
        F_c = state['F_c']              
        F_h_new, F_c_new = self.fast_cells[0](inputs, (F_h, F_c))  
        F_h, F_c = zoneout(F_h_new, F_c_new, F_h, F_c, self.zoneout_keep_h, self.zoneout_keep_c, self.training)   
        F_output_drop = self.d1(F_h)
        
        S_h = state['S_h']
        S_c = state['S_c']  
        S_h_new, S_c_new = self.slow_cell(F_output_drop, (S_h, S_c))
        S_h, S_c = zoneout(S_h_new, S_c_new, S_h, S_c, self.zoneout_keep_h, self.zoneout_keep_c, self.training)   
        S_output_drop = self.d2(S_h)
    
        F_h_new, F_c_new  = self.fast_cells[1](S_output_drop, (F_h, F_c))
        F_h, F_c = zoneout(F_h_new, F_c_new, F_h, F_c, self.zoneout_keep_h, self.zoneout_keep_c, self.training)       
        for i in range(2, self.fast_layers):
            F_h_new, F_c_new = self.fast_cells[i](F_h * 0.0, (F_h, F_c))
            F_h, F_c = zoneout(F_h_new, F_c_new, F_h, F_c, self.zoneout_keep_h, self.zoneout_keep_c, self.training)
        
        F_output_drop = self.d3(F_h)
        logits = self.out(F_output_drop)
        state = {'F_h': F_h, 'F_c': F_c, 'S_h': S_h, 'S_c': S_c}    
        return state, logits