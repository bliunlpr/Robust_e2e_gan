import logging
import math
import sys
import numpy as np
import six
from progressbar import ProgressBar

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from model.e2e_common import ModelBase, lecun_normal_init_parameters, set_forget_bias_to_one, to_cuda
from model.e2e_model import E2E
from model.enhance_model import FbankModel, EnhanceModel
        
                    
class E2EModel(ModelBase):
    def __init__(self, args):
        super(E2EModel, self).__init__()
        self.opt = args
        self.fbank_model = FbankModel(args)
        args.idim = args.fbank_dim
        self.e2e = E2E(args)
        
        self.sum = np.zeros(shape=[1, args.fbank_dim], dtype=np.float32)
        self.sum_sq = np.zeros(shape=[1, args.fbank_dim], dtype=np.float32)
        self.fbank_cmvn = np.zeros(shape=[2, args.fbank_dim], dtype=np.float32)        
        self.cmvn_num = min(args.train_dataset_len, args.num_utt_cmvn)
        self.cmvn_processed_num = 0
        self.frame_count = 0
        self.pbar = ProgressBar().start()
        print(">> compute fbank_cmvn using {} utterance ".format(self.cmvn_num))
        
    def forward(self, data, fbank_cmvn=None, scheduled_sample_rate=0.0):
                        
        utt_ids, spk_ids, inputs, targets, input_sizes, target_sizes = data
        inputs = to_cuda(self, inputs)
        ilens = to_cuda(self, input_sizes)
        fbank_cmvn = to_cuda(self, fbank_cmvn)
        fbank_features = self.fbank_model(inputs)
        if fbank_cmvn is not None:
            fbank_features = (fbank_features + fbank_cmvn[0, :]) * fbank_cmvn[1, :]
        loss_ctc, loss_att, acc = self.e2e(fbank_features, targets, input_sizes, target_sizes, scheduled_sample_rate)
    
        return loss_ctc, loss_att, acc
        
    def compute_cmvn(self, data):        
        utt_ids, spk_ids, inputs, targets, input_sizes, target_sizes = data
        inputs = to_cuda(self, inputs)
        ilens = to_cuda(self, input_sizes)
        fbank_features = self.fbank_model(inputs)
               
        if self.cmvn_processed_num < self.cmvn_num:
            for x in range(len(utt_ids)):
                input_size = int(input_sizes[x]) 
                feature_mat = fbank_features[x].data.cpu().numpy()
                feature_mat = feature_mat[:input_size, :]           
                sum_1utt = np.sum(feature_mat, axis=0)
                self.sum = np.add(self.sum, sum_1utt)
                feature_mat_square = np.square(feature_mat)
                sum_sq_1utt = np.sum(feature_mat_square, axis=0)
                self.sum_sq = np.add(self.sum_sq, sum_sq_1utt)
                self.frame_count += feature_mat.shape[0]            
                self.cmvn_processed_num += 1
                self.pbar.update(int((self.cmvn_processed_num / (self.cmvn_num - 1)) * 100))
            return None
        else:
            self.pbar.finish()
            mean = self.sum / self.frame_count
            var = self.sum_sq / self.frame_count - np.square(mean)
            print (self.frame_count)
            print (mean)
            print (var)
            self.fbank_cmvn[0, :] = -mean
            self.fbank_cmvn[1, :] = 1 / np.sqrt(var)
            return self.fbank_cmvn
    
    def recognize(self, x, fbank_cmvn, recog_args, char_list, rnnlm=None, kenlm=None):
        '''E2E beam search

        :param x:
        :param recog_args:
        :param char_list:
        :return:
        '''
        prev = self.training
        self.eval()
        
        x = to_cuda(self, x)
        fbank_cmvn = to_cuda(self, fbank_cmvn)
        fbank_features = self.fbank_model(x)
        fbank_features = (fbank_features + fbank_cmvn[0, :]) * fbank_cmvn[1, :]   
        y = self.e2e.recognize(fbank_features, recog_args, char_list, rnnlm, kenlm)
        
        if prev:
            self.train()
        return y
                
    def calculate_all_attentions(self, data, fbank_cmvn=None):
        '''E2E attention calculation

        :param list data: list of dicts of the input (B)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
         :rtype: float ndarray
        '''
        torch.set_grad_enabled(False)

        utt_ids, spk_ids, inputs, targets, input_sizes, target_sizes = data
        inputs = to_cuda(self, inputs)
        fbank_cmvn = to_cuda(self, fbank_cmvn)
        fbank_features = self.fbank_model(inputs)
        fbank_features = (fbank_features + fbank_cmvn[0, :]) * fbank_cmvn[1, :]
        data = (utt_ids, spk_ids, fbank_features, targets, input_sizes, target_sizes)
        
        att_ws = self.e2e.calculate_all_attentions(data)
        return att_ws