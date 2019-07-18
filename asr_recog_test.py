from __future__ import print_function
import argparse
import os
import math
import random
import shutil
import psutil
import time 
import numpy as np
from tqdm import tqdm
import json
import logging
import torch
from multiprocessing import Process

from options.test_options import TestOptions
from model.e2e_model import E2E
from model import lm, extlm
from data.data_loader import SequentialDataset, SequentialDataLoader, BucketingSampler
from utils.visualizer import Visualizer 
from utils import utils 


def recognize(opt, recog_dir, result_label, log_file):  
    # logging info
    '''if opt.verbose == 1:
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif opt.verbose == 2:
        logging.basicConfig(filename=log_file, level=logging.DEBUG,
                            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(filename=log_file, level=logging.WARN, 
                            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")'''
    
    if opt.resume:
        model_path = os.path.join(opt.works_dir, opt.resume)
        if not os.path.isfile(model_path):
            raise Exception("no checkpoint found at {}".format(model_path))
        
        package = torch.load(model_path, map_location=lambda storage, loc: storage)        
        model = E2E.load_model(model_path, 'state_dict') 
        logging.info('Loading model {}'.format(model_path))
    else:
        raise Exception("no checkpoint found at {}".format(opt.resume))
    
    def cpu_loader(storage, location):
        return storage
    
    if opt.lmtype == 'rnnlm':         
        # read rnnlm
        if opt.rnnlm:
            rnnlm = lm.ClassifierWithState(
                lm.RNNLM(len(opt.char_list), 650, 650))
            rnnlm.load_state_dict(torch.load(opt.rnnlm, map_location=cpu_loader))
            rnnlm.eval()
        else:
            rnnlm = None
        
        if opt.word_rnnlm:
            if not opt.word_dict:
                logging.error('word dictionary file is not specified for the word RNNLM.')
                sys.exit(1)

            word_dict = load_labeldict(opt.word_dict)
            char_dict = {x: i for i, x in enumerate(opt.char_list)}
            word_rnnlm = lm.ClassifierWithState(lm.RNNLM(len(word_dict), 650))
            word_rnnlm.load_state_dict(torch.load(opt.word_rnnlm, map_location=cpu_loader))
            word_rnnlm.eval()

            if rnnlm is not None:
                rnnlm = lm.ClassifierWithState(
                    extlm.MultiLevelLM(word_rnnlm.predictor,
                                               rnnlm.predictor, word_dict, char_dict))
            else:
                rnnlm = lm.ClassifierWithState(
                    extlm.LookAheadWordLM(word_rnnlm.predictor,
                                                  word_dict, char_dict))
        kenlm = None                                           
    elif opt.lmtype == 'kenlm':   
        if opt.kenlm: 
            kenlm = extlm.NgramCharacterKenLM(opt.kenlm, opt.char_list)
        else:
            kenlm = None 
        rnnlm = None
    else:
        rnnlm = None
        kenlm = None
                
    logger = logging.getLogger('mylogger') 
    logger.setLevel(logging.DEBUG)  
    fh = logging.FileHandler(log_file) 
    fh.setLevel(logging.DEBUG)   
    formatter = logging.Formatter('[%(asctime)s][%(thread)d][%(filename)s][line: %(lineno)d][%(levelname)s] ## %(message)s')
    fh.setFormatter(formatter)    
    logger.addHandler(fh)   
        
    recog_dataset = SequentialDataset(opt, recog_dir, os.path.join(opt.dict_dir, 'train_units.txt'),) 
    recog_loader = SequentialDataLoader(recog_dataset, batch_size=1, num_workers=opt.num_workers, shuffle=False)
    
    torch.set_grad_enabled(False)
    new_json = {}
    for i, (data) in enumerate(recog_loader, start=0):
        utt_ids, spk_ids, inputs, targets, input_sizes, target_sizes = data
        name = utt_ids[0]
        logger.info("name: " + name)
        nbest_hyps = model.recognize(inputs, opt, opt.char_list, rnnlm=rnnlm, kenlm=kenlm)
        # get 1best and remove sos
        y_hat = nbest_hyps[0]['yseq'][1:]
        ##y_true = map(int, targets[0].split())
        y_true = targets
        
        # print out decoding result
        seq_hat = [opt.char_list[int(idx)] for idx in y_hat]
        seq_true = [opt.char_list[int(idx)] for idx in y_true]
        seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
        seq_true_text = "".join(seq_true).replace('<space>', ' ')
        logger.info("groundtruth[%s]: " + seq_true_text, name)
        logger.info("prediction [%s]: " + seq_hat_text, name)
        # copy old json info
        new_json[name] = dict()
        new_json[name]['utt2spk'] = spk_ids[0]

        # added recognition results to json
        logging.debug("dump token id")
        out_dic = dict()
        out_dic['name'] = 'target1'
        out_dic['text'] = seq_true_text
        out_dic['token'] = " ".join(seq_true)
        out_dic['tokenid'] = " ".join([str(int(idx)) for idx in y_true])

        # TODO(karita) make consistent to chainer as idx[0] not idx
        out_dic['rec_tokenid'] = " ".join([str(int(idx)) for idx in y_hat])
        #logger.debug("dump token")
        out_dic['rec_token'] = " ".join(seq_hat)
        #logger.debug("dump text")
        out_dic['rec_text'] = seq_hat_text

        new_json[name]['output'] = [out_dic]
        # TODO(nelson): Modify this part when saving more than 1 hyp is enabled
        # add n-best recognition results with scores
        if opt.beam_size > 1 and len(nbest_hyps) > 1:
            for i, hyp in enumerate(nbest_hyps):
                y_hat = hyp['yseq'][1:]
                seq_hat = [opt.char_list[int(idx)] for idx in y_hat]
                seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
                new_json[name]['rec_tokenid' + '[' + '{:05d}'.format(i) + ']'] = " ".join([str(idx) for idx in y_hat])
                new_json[name]['rec_token' + '[' + '{:05d}'.format(i) + ']'] = " ".join(seq_hat)
                new_json[name]['rec_text' + '[' + '{:05d}'.format(i) + ']'] = seq_hat_text
                new_json[name]['score' + '[' + '{:05d}'.format(i) + ']'] = hyp['score']

    # TODO(watanabe) fix character coding problems when saving it
    with open(result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_json}, indent=4, sort_keys=True).encode('utf_8'))
         
                                                   
def main():
    
    opt = TestOptions().parse()
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)  
        
    # data
    recog_dir = os.path.join(opt.recog_dir, '1')
    tmp_recog_dataset = SequentialDataset(opt, recog_dir, os.path.join(opt.dict_dir, 'train_units.txt'),) 
    opt.idim = tmp_recog_dataset.get_feat_size()
    opt.odim = tmp_recog_dataset.get_num_classes()
    opt.char_list = tmp_recog_dataset.get_char_list()
    opt.labeldist = tmp_recog_dataset.get_labeldist()
    print('#input dims : ' + str(opt.idim))
    print('#output dims: ' + str(opt.odim))
        
    log_file_dir = os.path.join(opt.result_label, 'log')    
    if not os.path.exists(log_file_dir):
        os.makedirs(log_file_dir)
        
    p_list = [] 
    num_works = opt.nj
    for x in range(num_works):
        recog_dir = os.path.join(opt.recog_dir, str(x+1))
        result_label = os.path.join(opt.result_label, 'data.{}.json'.format(x+1))
        log_file = os.path.join(log_file_dir, 'decode.{}.log'.format(x+1))
        p_list.append(Process(target=recognize, args=(opt, recog_dir, result_label, log_file)))
    for xx in p_list:
        xx.start()

    for xx in p_list:
        xx.join()
        
              
if __name__ == '__main__':
    main()
