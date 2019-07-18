from __future__ import print_function
import argparse
import os
import math
import random
import shutil
import psutil
import time 
import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F

from options.test_options import TestOptions
from model.enhance_model import EnhanceModel, EnhanceConditionModel
from data.mix_data_loader import MixSequentialDataset, MixSequentialDataLoader, BucketingSampler
from data import kaldi_io
from utils import utils 


manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed) 
                                                  
def main():
    opt = TestOptions().parse()
    device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if len(opt.gpu_ids) > 0 and torch.cuda.is_available() else "cpu")

    # data
    logging.info("Building dataset.")
    val_dataset = MixSequentialDataset(opt, opt.enhance_dir, os.path.join(opt.dict_dir, 'train_units.txt'),)
    val_loader = MixSequentialDataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers)
    opt.idim = val_dataset.get_feat_size()
    opt.odim = val_dataset.get_num_classes()
    opt.char_list = val_dataset.get_char_list()
    print('len(val_dataset)', len(val_dataset))
    logging.info('#input dims : ' + str(opt.idim))
    logging.info('#output dims: ' + str(opt.odim))
    logging.info("Dataset ready!")
    
    # Setup a model  
    if opt.resume:
        model_path = os.path.join(opt.works_dir, opt.resume)
        if os.path.isfile(model_path):
            package = torch.load(model_path, map_location=lambda storage, loc: storage)
            if opt.enhance_type == 'enhance_condition':
                model = EnhanceConditionModel.load_model(model_path, 'enhance_state_dict') 
            else:
                model = EnhanceModel.load_model(model_path, 'enhance_state_dict') 
            logging.info('Loading model {}'.format(model_path))
        else:
            raise Exception("no checkpoint found at {}".format(opt.resume))     
    else:
        raise Exception("no checkpoint found at {}".format(opt.resume))
    model.cuda()
    
    torch.set_grad_enabled(False)
    feat_dir = os.path.join(opt.exp_path, opt.enhance_out_dir)
    clean_feat_dir = os.path.join(feat_dir, 'clean_enhanced')
    mix_feat_dir = os.path.join(feat_dir, 'mix_enhanced')
    if not os.path.exists(clean_feat_dir):
        os.makedirs(clean_feat_dir)
    if not os.path.exists(mix_feat_dir):
        os.makedirs(mix_feat_dir)
        
    clean_ark_scp_output = 'ark:| copy-feats --compress=true ark:- ark,scp:{0}/feats.ark,{0}/feats.scp'.format(clean_feat_dir)    
    clean_feats_write = kaldi_io.open_or_fd(clean_ark_scp_output, 'wb')
    clean_utt2spk_write = open(os.path.join(clean_feat_dir, 'utt2spk'), 'w', encoding='utf-8')
    clean_text_write = open(os.path.join(clean_feat_dir, 'text_char'), 'w', encoding='utf-8')
    
    mix_ark_scp_output = 'ark:| copy-feats --compress=true ark:- ark,scp:{0}/feats.ark,{0}/feats.scp'.format(mix_feat_dir)
    mix_feats_write = kaldi_io.open_or_fd(mix_ark_scp_output, 'wb')
    mix_utt2spk_write = open(os.path.join(mix_feat_dir, 'utt2spk'), 'w', encoding='utf-8')
    mix_text_write = open(os.path.join(mix_feat_dir, 'text_char'), 'w', encoding='utf-8')

    for i, (data) in enumerate(val_loader):
        utt_ids, spk_ids, clean_inputs, clean_log_inputs, mix_inputs, mix_log_inputs, cos_angles, targets, input_sizes, target_sizes = data            
        clean_outputs = model(clean_inputs, clean_log_inputs, input_sizes)
        mix_outputs = model(mix_inputs, mix_log_inputs, input_sizes)
        offset = 0 
        for x in range(len(utt_ids)):
            utt_id = utt_ids[x]
            input_size = int(input_sizes[x])
            clean_output = clean_outputs[x, :input_size, :].data.cpu().numpy()
            mix_output = mix_outputs[x, :input_size, :].data.cpu().numpy()
            kaldi_io.write_mat(clean_feats_write, clean_output, key=utt_id)
            kaldi_io.write_mat(mix_feats_write, mix_output, key=utt_id)
            clean_utt2spk_write.write(utt_id + ' ' + spk_ids[x] + '\n')
            mix_utt2spk_write.write(utt_id + ' ' + spk_ids[x] + '\n')
            target = targets[offset:offset + int(target_sizes[x])]
            offset += int(target_sizes[x])        
            seq_true = [opt.char_list[int(idx)] for idx in target]
            text_token = " ".join(seq_true)
            clean_text_write.write(utt_id + ' ' + text_token + '\n')
            mix_text_write.write(utt_id + ' ' + text_token + '\n')
            
    clean_feats_write.close()
    mix_feats_write.close() 
    clean_utt2spk_write.close()
    mix_utt2spk_write.close()
    clean_text_write.close()
    mix_text_write.close()       
    print('finish')
    exit()
                   
                                      
if __name__ == '__main__':
    main()
