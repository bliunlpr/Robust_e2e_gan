from __future__ import print_function
import argparse
import os
import math
import random
import shutil
import psutil
import time 
import itertools
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F

from options.train_options import TrainOptions
from model.enhance_model import EnhanceModel
from model.feat_model import FFTModel, FbankModel
from model.e2e_model import E2E
from model.gan_model import GANModel, GANLoss
from model.e2e_common import set_requires_grad
from data.data_loader import SequentialDataset, SequentialDataLoader, BucketingSampler
from utils.visualizer import Visualizer 
from utils import utils 

manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed) 


def compute_cmvn_epoch(opt, train_loader, enhance_model, feat_model):
    enhance_model.eval()
    feat_model.eval() 
    torch.set_grad_enabled(False)
    enhance_cmvn_file = os.path.join(opt.exp_path, 'enhance_cmvn.npy')
    if os.path.exists(enhance_cmvn_file):
        enhance_cmvn = np.load(enhance_cmvn_file)
    else:
        for i, (data) in enumerate(train_loader, start=0):
            utt_ids, spk_ids, inputs, log_inputs, targets, input_sizes, target_sizes = data
            enhance_out = enhance_model.enhance_out(inputs, log_inputs, input_sizes) 
            enhance_cmvn = feat_model.compute_cmvn(enhance_out, input_sizes)
            if enhance_cmvn is not None:
                np.save(enhance_cmvn_file, enhance_cmvn)
                print('save enhance_cmvn to {}'.format(enhance_cmvn_file))
                break
    enhance_cmvn = torch.FloatTensor(enhance_cmvn)
    enhance_model.train()
    feat_model.train()
    torch.set_grad_enabled(True)  
    return enhance_cmvn
    
         
def main():    
    opt = TrainOptions().parse()    
    device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if len(opt.gpu_ids) > 0 and torch.cuda.is_available() else "cpu")
     
    visualizer = Visualizer(opt)  
    logging = visualizer.get_logger()
    acc_report = visualizer.add_plot_report(['train/acc', 'val/acc'], 'acc.png')
    loss_report = visualizer.add_plot_report(['train/loss', 'val/loss'], 'loss.png')
     
    # data
    logging.info("Building dataset.")
    train_dataset = SequentialDataset(opt, os.path.join(opt.dataroot, 'train'), os.path.join(opt.dict_dir, 'train_units.txt'),) 
    val_dataset = SequentialDataset(opt, os.path.join(opt.dataroot, 'dev'), os.path.join(opt.dict_dir, 'train_units.txt'),)    
    train_sampler = BucketingSampler(train_dataset, batch_size=opt.batch_size) 
    train_loader = SequentialDataLoader(train_dataset, num_workers=opt.num_workers, batch_sampler=train_sampler)
    val_loader = SequentialDataLoader(val_dataset, batch_size=int(opt.batch_size/2), num_workers=opt.num_workers, shuffle=False)
    opt.idim = train_dataset.get_feat_size()
    opt.odim = train_dataset.get_num_classes()
    opt.char_list = train_dataset.get_char_list()
    opt.train_dataset_len = len(train_dataset)
    logging.info('#input dims : ' + str(opt.idim))
    logging.info('#output dims: ' + str(opt.odim))
    logging.info("Dataset ready!")
    
    # Setup an model
    lr = opt.lr
    eps = opt.eps
    iters = opt.iters   
    best_acc = opt.best_acc 
    best_loss = opt.best_loss  
    start_epoch = opt.start_epoch
    
    enhance_model_path = None
    if opt.enhance_resume:
        enhance_model_path = os.path.join(opt.works_dir, opt.enhance_resume)
        if os.path.isfile(enhance_model_path):
            enhance_model = EnhanceModel.load_model(enhance_model_path, 'state_dict', opt)
        else:
            print("no checkpoint found at {}".format(enhance_model_path))     
    
    asr_model_path = None
    if opt.asr_resume:
        asr_model_path = os.path.join(opt.works_dir, opt.asr_resume)
        if os.path.isfile(asr_model_path):
            asr_model = E2E.load_model(asr_model_path, 'state_dict', opt)
        else:
            print("no checkpoint found at {}".format(asr_model_path))  
                                        
    joint_model_path = None
    if opt.joint_resume:
        joint_model_path = os.path.join(opt.works_dir, opt.joint_resume)
        if os.path.isfile(joint_model_path):
            package = torch.load(joint_model_path, map_location=lambda storage, loc: storage)
            lr = package.get('lr', opt.lr)
            eps = package.get('eps', opt.eps)  
            best_acc = package.get('best_acc', 0)      
            best_loss = package.get('best_loss', float('inf'))
            start_epoch = int(package.get('epoch', 0))   
            iters = int(package.get('iters', 0)) - 1           
            loss_report = package.get('loss_report', loss_report)
            visualizer.set_plot_report(loss_report, 'loss.png')
        else:
            print("no checkpoint found at {}".format(joint_model_path))
    if joint_model_path is not None or enhance_model_path is None:     
        enhance_model = EnhanceModel.load_model(joint_model_path, 'enhance_state_dict', opt)
    feat_model = FbankModel.load_model(joint_model_path, 'fbank_state_dict', opt) 
    if joint_model_path is not None or asr_model_path is None:  
        asr_model = E2E.load_model(joint_model_path, 'asr_state_dict', opt) 
    ##set_requires_grad([enhance_model], False)    
    
    # Setup an optimizer
    enhance_parameters = filter(lambda p: p.requires_grad, enhance_model.parameters())
    asr_parameters = filter(lambda p: p.requires_grad, asr_model.parameters())
    if opt.opt_type == 'adadelta':
        enhance_optimizer = torch.optim.Adadelta(enhance_parameters, rho=0.95, eps=eps)
        asr_optimizer = torch.optim.Adadelta(asr_parameters, rho=0.95, eps=eps)
    elif opt.opt_type == 'adam':
        enhance_optimizer = torch.optim.Adam(enhance_parameters, lr=lr, betas=(opt.beta1, 0.999))   
        asr_optimizer = torch.optim.Adam(asr_parameters, lr=lr, betas=(opt.beta1, 0.999))                       
           
    # Training	
    enhance_cmvn = compute_cmvn_epoch(opt, train_loader, enhance_model, feat_model) 
    sample_rampup = utils.ScheSampleRampup(opt.sche_samp_start_epoch, opt.sche_samp_final_epoch, opt.sche_samp_final_rate)  
    enhance_model.train()
    feat_model.train()
    asr_model.train()               	                    
    for epoch in range(start_epoch, opt.epochs):         
        sche_samp_rate = sample_rampup.update(epoch)
        print("epoch {} sche_samp_rate {}".format(epoch, sche_samp_rate))              
        if epoch > opt.shuffle_epoch:
            print("Shuffling batches for the following epochs")
            train_sampler.shuffle(epoch)  
        for i, (data) in enumerate(train_loader, start=0):
            utt_ids, spk_ids, inputs, log_inputs, targets, input_sizes, target_sizes = data
            enhance_out = enhance_model.enhance_out(inputs, log_inputs, input_sizes)
            enhance_feat = feat_model(enhance_out, enhance_cmvn)
            loss_ctc, loss_att, acc = asr_model(enhance_feat, targets, input_sizes, target_sizes, sche_samp_rate)             
            loss = opt.mtlalpha * loss_ctc + (1 - opt.mtlalpha) * loss_att
            enhance_optimizer.zero_grad()
            asr_optimizer.zero_grad()  # Clear the parameter gradients
            loss.backward()          
            # compute the gradient norm to check if it is normal or not
            grad_norm = torch.nn.utils.clip_grad_norm_(asr_model.parameters(), opt.grad_clip)
            if math.isnan(grad_norm):
                logging.warning('grad norm is nan. Do not update model.')
            else:
                enhance_optimizer.step()
                asr_optimizer.step()                
                                           
            iters += 1
            errors = {'train/loss': loss.item(), 'train/loss_ctc': loss_ctc.item(), 
                      'train/acc': acc, 'train/loss_att': loss_att.item()}
            visualizer.set_current_errors(errors)
            if iters % opt.print_freq == 0:
                visualizer.print_current_errors(epoch, iters)
                state = {'asr_state_dict': asr_model.state_dict(), 
                         'fbank_state_dict': feat_model.state_dict(), 
                         'enhance_state_dict': enhance_model.state_dict(), 
                         'opt': opt, 'epoch': epoch, 'iters': iters, 
                         'eps': opt.eps, 'lr': opt.lr,                                    
                         'best_loss': best_loss, 'best_acc': best_acc, 
                         'acc_report': acc_report, 'loss_report': loss_report}
                filename='latest'
                utils.save_checkpoint(state, opt.exp_path, filename=filename)
                    
            if iters % opt.validate_freq == 0:
                enhance_model.eval() 
                feat_model.eval() 
                asr_model.eval()
                torch.set_grad_enabled(False)                
                num_saved_attention = 0 
                for i, (data) in tqdm(enumerate(val_loader, start=0)):
                    utt_ids, spk_ids, inputs, log_inputs, targets, input_sizes, target_sizes = data
                    enhance_out = enhance_model.enhance_out(inputs, log_inputs, input_sizes)
                    enhance_feat = feat_model(enhance_out, enhance_cmvn)
                    loss_ctc, loss_att, acc = asr_model(enhance_feat, targets, input_sizes, target_sizes, sche_samp_rate)             
                    loss = opt.mtlalpha * loss_ctc + (1 - opt.mtlalpha) * loss_att                            
                    errors = {'val/loss': loss.item(), 'val/loss_ctc': loss_ctc.item(), 
                              'val/acc': acc, 'val/loss_att': loss_att.item()}
                    visualizer.set_current_errors(errors)
                
                    if opt.num_save_attention > 0 and opt.mtlalpha != 1.0:
                        if num_saved_attention < opt.num_save_attention:
                            att_ws = asr_model.calculate_all_attentions(enhance_feat, targets, input_sizes, target_sizes)                            
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
                enhance_model.train()
                feat_model.train()
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
                         'fbank_state_dict': feat_model.state_dict(), 
                         'enhance_state_dict': enhance_model.state_dict(), 
                         'opt': opt, 'epoch': epoch, 'iters': iters, 
                         'eps': opt.eps, 'lr': opt.lr,                                    
                         'best_loss': best_loss, 'best_acc': best_acc, 
                         'acc_report': acc_report, 'loss_report': loss_report}
                utils.save_checkpoint(state, opt.exp_path, filename=filename)                  
                visualizer.reset()  
                enhance_cmvn = compute_cmvn_epoch(opt, train_loader, enhance_model, feat_model)  
                
if __name__ == '__main__':
    main()
