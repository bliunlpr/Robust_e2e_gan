# This code is ported from the following implementation written in Torch.
# https://github.com/chainer/chainer/blob/master/examples/ptb/train_ptb_custom_loop.py

import argparse
import logging
import numpy as np
import os
import platform
import random
import subprocess
import sys

from model import lm


def main():
    parser = argparse.ArgumentParser()
    # general configuration
    parser.add_argument('--ngpu', default=0, type=int, help='Number of GPUs')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory')
    parser.add_argument('--debugmode', default=1, type=int, help='Debugmode')
    parser.add_argument('--dict', type=str, required=True, help='Dictionary')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    parser.add_argument('--minibatches', '-N', type=int, default='-1', help='Process only N minibatches (for debug)')
    parser.add_argument('--verbose', '-V', default=0, type=int, help='Verbose option')
    parser.add_argument('--embed-init-file', type=str, default='', help='Dictionary')
    
    # task related
    parser.add_argument('--train-label', type=str, required=True, help='Filename of train label data (json)')
    parser.add_argument('--valid-label', type=str, required=True, help='Filename of validation label data (json)')
    parser.add_argument('--lm-type', type=str, default='rnnlm', help='rnnlm fsrnnlm')
    
    # LSTMLM training configuration
    parser.add_argument('--batchsize', '-b', type=int, default=2048, help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=35, help='Number of words in each mini-batch (= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=25, help='Number of sweeps over the dataset to train')
    parser.add_argument('--gradclip', '-c', type=float, default=5, help='Gradient norm threshold to clip')
    parser.add_argument('--input-unit', type=int, default=650, help='Number of LSTM units in each layer')
    parser.add_argument('--unit', '-u', type=int, default=650, help='Number of LSTM units in each layer')
    parser.add_argument('--dropout-rate', type=float, default=0.2, help='dropout_rate')
    
    # FSLSTMLM training configuration
    parser.add_argument('--fast_cell_size', type=int, default=650, help='fast_cell_size')
    parser.add_argument('--slow_cell_size', type=int, default=650, help='slow_cell_size')
    parser.add_argument('--fast_layers', type=int, default=2, help='fast_layers')
    parser.add_argument('--zoneout_keep_c', type=float, default=0.5, help='zoneout_c')
    parser.add_argument('--zoneout_keep_h', type=float, default=0.9, help='zoneout_h')
    args = parser.parse_args()

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(
            level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
        logging.warning('Skip DEBUG/INFO messages')

    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warn("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

    # seed setting
    nseed = args.seed
    random.seed(nseed)
    np.random.seed(nseed)

    # load dictionary
    with open(args.dict, 'rb') as f:
        dictionary = f.readlines()
    char_list = [entry.decode('utf-8').split(' ')[0] for entry in dictionary]
    char_list.insert(0, '<blank>')
    char_list.append('<eos>')
    args.char_list_dict = {x: i for i, x in enumerate(char_list)}
    args.n_vocab = len(char_list)
    
    lm.train(args)
    

if __name__ == '__main__':
    main()