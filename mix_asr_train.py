from __future__ import print_function
import argparse
import os
import math
import random
import time 
import itertools
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F

from options.train_options import TrainOptions
from model.e2e_model import E2E
from model.feat_model import FbankModel
from data.mix_data_loader import MixSequentialDataset, MixSequentialDataLoader
from data.data_sampler import BucketingSampler, DistributedBucketingSampler
from utils.visualizer import Visualizer 
from utils import utils 
    
manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed) 
                                                     
def main():
    
    opt = TrainOptions().parse()    
    device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if len(opt.gpu_ids) > 0 and torch.cuda.is_available() else "cpu")
     
    visualizer = Visualizer(opt)  
    logging = visualizer.get_logger()
    acc_report = visualizer.add_plot_report(['train/acc', 'val/acc'], 'acc.png')
    loss_report = visualizer.add_plot_report(['train/loss', 'val/loss'], 'loss.png')
     
    # data
    logging.info("Building dataset.")
    train_dataset = MixSequentialDataset(opt, os.path.join(opt.dataroot, 'train'), os.path.join(opt.dict_dir, 'train_units.txt')) 
    val_dataset = MixSequentialDataset(opt, os.path.join(opt.dataroot, 'dev'), os.path.join(opt.dict_dir, 'train_units.txt'), data_type='dev')    
    train_sampler = BucketingSampler(train_dataset, batch_size=opt.batch_size) 
    train_loader = MixSequentialDataLoader(train_dataset, num_workers=opt.num_workers, batch_sampler=train_sampler)
    val_loader = MixSequentialDataLoader(val_dataset, batch_size=int(opt.batch_size/2), num_workers=opt.num_workers, shuffle=False)
    opt.idim = train_dataset.get_feat_size()
    opt.odim = train_dataset.get_num_classes()
    opt.char_list = train_dataset.get_char_list()
    opt.train_dataset_len = len(train_dataset)
    logging.info('#input dims : ' + str(opt.idim))
    logging.info('#output dims: ' + str(opt.odim))
    logging.info("Dataset ready!")
    
    # Setup a model
    asr_model = E2E(opt)
    lr = opt.lr
    eps = opt.eps
    iters = opt.iters
    epoch_iters = 0
    start_epoch = opt.start_epoch    
    best_loss = opt.best_loss
    best_acc = opt.best_acc
    if opt.resume:
        model_path = os.path.join(opt.works_dir, opt.resume)
        if os.path.isfile(model_path):
            package = torch.load(model_path, map_location=lambda storage, loc: storage)
            lr = package.get('lr', opt.lr)
            eps = package.get('eps', opt.eps)        
            best_loss = package.get('best_loss', float('inf'))
            best_acc = package.get('best_acc', 0)
            start_epoch = int(package.get('epoch', 0))   
            iters = int(package.get('iters', 0))
            epoch_iters = int(package.get('epoch_iters', 0))
            
            acc_report = package.get('acc_report', acc_report)
            loss_report = package.get('loss_report', loss_report)
            visualizer.set_plot_report(acc_report, 'acc.png')
            visualizer.set_plot_report(loss_report, 'loss.png')
            
            asr_model = E2E.load_model(model_path, 'asr_state_dict')  
            logging.info('Loading model {} and iters {}'.format(model_path, iters))
        else:
            print("no checkpoint found at {}".format(model_path))                
    asr_model.cuda()
    print(asr_model)
  
    # Setup an optimizer
    parameters = filter(lambda p: p.requires_grad, itertools.chain(asr_model.parameters()))
    if opt.opt_type == 'adadelta':
        optimizer = torch.optim.Adadelta(parameters, rho=0.95, eps=eps)
    elif opt.opt_type == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=lr, betas=(opt.beta1, 0.999))                       
           
    asr_model.train()
    sample_rampup = utils.ScheSampleRampup(opt.sche_samp_start_iter, opt.sche_samp_final_iter, opt.sche_samp_final_rate)  
    clean_rampup = utils.CleanRatioRampup(opt.clean_wav_start_ratio, opt.clean_wav_start_epoch, opt.clean_wav_final_ratio, opt.clean_wav_final_epoch)  
    sche_samp_rate = sample_rampup.update(iters)
                        
    for epoch in range(start_epoch, opt.epochs):
        clean_wav_ratio = clean_rampup.update(epoch)
        train_dataset = MixSequentialDataset(opt, os.path.join(opt.dataroot, 'train'), os.path.join(opt.dict_dir, 'train_units.txt'), clean_wav_ratio) 
        train_sampler = BucketingSampler(train_dataset, batch_size=opt.batch_size) 
        train_sampler.shuffle(epoch)
        train_loader = MixSequentialDataLoader(train_dataset, num_workers=opt.num_workers, batch_sampler=train_sampler)               
        for i, (data) in enumerate(train_loader, start=epoch_iters):
            utt_ids, spk_ids, fbank_features, targets, input_sizes, target_sizes = data
            loss_ctc, loss_att, acc, context = asr_model(fbank_features, targets, input_sizes, target_sizes, sche_samp_rate) 
            loss = opt.mtlalpha * loss_ctc + (1 - opt.mtlalpha) * loss_att
            optimizer.zero_grad()  # Clear the parameter gradients
            loss.backward()          
            # compute the gradient norm to check if it is normal or not 'fbank_state_dict': fbank_model.state_dict(), 
            grad_norm = torch.nn.utils.clip_grad_norm_(asr_model.parameters(), opt.grad_clip)
            if math.isnan(grad_norm):
                logging.warning('grad norm is nan. Do not update model.')
            else:
                optimizer.step()
                
            iters += 1
            epoch_iters += 1
            errors = {'train/loss': loss.item(), 'train/loss_ctc': loss_ctc.item(), 
                      'train/acc': acc, 'train/loss_att': loss_att.item()}
            visualizer.set_current_errors(errors)
            if iters % opt.print_freq == 0:
                visualizer.print_current_errors(epoch, iters)
                state = {'asr_state_dict': asr_model.state_dict(), 
                         'opt': opt, 'epoch': epoch, 'iters': iters, 
                         'eps': opt.eps, 'lr': opt.lr, 'epoch_iters': epoch_iters,                                 
                         'best_loss': best_loss, 'best_acc': best_acc, 
                         'acc_report': acc_report, 'loss_report': loss_report}
                filename='latest'
                utils.save_checkpoint(state, opt.exp_path, filename=filename)
                    
            if iters % opt.validate_freq == 0:
                sche_samp_rate = sample_rampup.update(iters)
                print("iters {} sche_samp_rate {}".format(iters, sche_samp_rate))  
                asr_model.eval()
                torch.set_grad_enabled(False)
                num_saved_attention = 0
                for i, (data) in tqdm(enumerate(val_loader, start=0)):
                    utt_ids, spk_ids, fbank_features, targets, input_sizes, target_sizes = data
                    loss_ctc, loss_att, acc, context = asr_model(fbank_features, targets, input_sizes, target_sizes, 0.0) 
                    loss = opt.mtlalpha * loss_ctc + (1 - opt.mtlalpha) * loss_att                            
                    errors = {'val/loss': loss.item(), 'val/loss_ctc': loss_ctc.item(), 
                              'val/acc': acc, 'val/loss_att': loss_att.item()}
                    visualizer.set_current_errors(errors)
                    
                    if opt.num_save_attention > 0 and opt.mtlalpha != 1.0:
                        if num_saved_attention < opt.num_save_attention:
                            att_ws = asr_model.calculate_all_attentions(fbank_features, targets, input_sizes, target_sizes)                            
                            for x in range(len(utt_ids)):
                                att_w = att_ws[x]
                                utt_id = utt_ids[x]
                                file_name = "{}_ep{}_it{}.png".format(utt_id, epoch, iters)
                                dec_len = int(target_sizes[x])
                                enc_len = int(input_sizes[x]) 
                                visualizer.plot_attention(att_w, dec_len, enc_len, file_name) 
                                num_saved_attention += 1
                                if num_saved_attention >= opt.num_save_attention:   
                                    break                                                       
                asr_model.train()
                torch.set_grad_enabled(True)
                
                visualizer.print_epoch_errors(epoch, iters)  
                acc_report = visualizer.plot_epoch_errors(epoch, iters, 'acc.png') 
                loss_report = visualizer.plot_epoch_errors(epoch, iters, 'loss.png') 
                val_loss = visualizer.get_current_errors('val/loss')
                val_acc = visualizer.get_current_errors('val/acc')  
                filename = None              
                if opt.criterion == 'acc' and opt.mtl_mode is not 'ctc':
                    if val_acc < best_acc:
                        logging.info('val_acc {} > best_acc {}'.format(val_acc, best_acc))
                        opt.eps = utils.adadelta_eps_decay(optimizer, opt.eps_decay)
                    else:
                        filename='model.acc.best'                    
                    best_acc = max(best_acc, val_acc)
                    logging.info('best_acc {}'.format(best_acc))  
                elif args.criterion == 'loss':
                    if val_loss > best_loss:
                        logging.info('val_loss {} > best_loss {}'.format(val_loss, best_loss))
                        opt.eps = utils.adadelta_eps_decay(optimizer, opt.eps_decay)
                    else:
                        filename='model.loss.best'    
                    best_loss = min(val_loss, best_loss)
                    logging.info('best_loss {}'.format(best_loss))                  
                state = {'asr_state_dict': asr_model.state_dict(), 
                         'opt': opt, 'epoch': epoch, 'iters': iters, 
                         'eps': opt.eps, 'lr': opt.lr, 'epoch_iters': epoch_iters,                                       
                         'best_loss': best_loss, 'best_acc': best_acc, 
                         'acc_report': acc_report, 'loss_report': loss_report}
                utils.save_checkpoint(state, opt.exp_path, filename=filename)                    
                visualizer.reset() 
        
          
if __name__ == '__main__':
    main()
