import numpy as np
import time
import torch
import os
import sys
import logging
import time
from datetime import timedelta


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class LogFormatter():
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)
        
def create_output_dir(opt):
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    filepath = os.path.join(expr_dir, 'main.log')

    # Safety check
    if os.path.exists(filepath) and opt.resume == "":
        logging.warning("Experiment already exists!")

    # Create logger
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # quite down visdom
    logging.getLogger("requests").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    logger.info(opt)
    return logger


def check_grad(params, clip_th, ignore_th):
    befgad = torch.nn.utils.clip_grad_norm_(params, clip_th)
    return (not np.isfinite(befgad) or (befgad > ignore_th))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_acc(output, target):
    pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    correct = pred.eq(target.view_as(pred)).sum().item()
    batch_size = target.size(0)
    correct *= (100.0 / batch_size)
    return correct         


def save_checkpoint(state, save_path, is_best=False, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if filename is not None:
        torch.save(state, os.path.join(save_path, filename))
        if is_best:
            shutil.copyfile(os.path.join(save_path, filename),
                            os.path.join(save_path, 'model_best.pth.tar'))

def adjust_learning_rate_by_factor(optimizer, lr, factor):
    """Adjusts the learning rate according to the given factor"""
    lr = lr * factor
    lr = max(lr, 0.000005)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
    
    
def adadelta_eps_decay(optimizer, eps_decay):
    '''Extension to perform adadelta eps decay'''
    for p in optimizer.param_groups:
        p["eps"] *= eps_decay
        logging.info('adadelta eps decayed to ' + str(p["eps"]))
        return p["eps"]
    return 0
    

class ScheSampleRampup(object):
    """Computes and stores the average and current value"""
    def __init__(self, start_epoch, final_epoch, final_rate):
        self.epoch = 0
        self.start_epoch = start_epoch
        self.final_epoch = final_epoch
        self.final_rate = final_rate
        self.linear = float(final_rate) / (final_epoch - start_epoch)
        
    def reset(self):
        self.epoch = 0

    def update(self, epoch):
        if epoch < self.start_epoch:
            sche_samp_rate = 0.0
        elif epoch < self.final_epoch:
            sche_samp_rate = self.linear * (epoch - self.start_epoch)
        else:
            sche_samp_rate = self.final_rate
        return sche_samp_rate
        
        
                                                    