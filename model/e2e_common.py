import logging
import math
import sys
import numpy as np
import six
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable


class ModelBase(torch.nn.Module):
    """
    ModelBase class for sharing code among various model.
    """
    def forward(self, x):
        raise NotImplementedError
    
    @classmethod
    def load_model(cls, path, state_dict, opt=None):
        if path is not None:
            package = torch.load(path, map_location=lambda storage, loc: storage)
            model = cls(args=package['opt'])
            print('model.state_dict() is', model.state_dict().keys()) 
            if state_dict in package and package[state_dict] is not None:
                model.load_state_dict(package[state_dict])  
                print('package.state_dict() is', package[state_dict].keys()) 
                print("checkpoint found at {} {}".format(path, state_dict)) 
        else:
            model = cls(opt)
            print("no checkpoint found, so init model") 
        if opt is not None and len(opt.gpu_ids) > 0: 
            model = model.cuda() 
        print(model)     
        return model
        
    @staticmethod
    def serialize(model, state_dict, optimizer=None, optim_dict=None):
        model_is_cuda = next(model.parameters()).is_cuda
        model = model.module if model_is_cuda else model
        package = {
            'opt': model.args,
            'state_dict': model.state_dict()            
        }
        if optimizer is not None:
            package[optim_dict] = optimizer.state_dict()
        return package
        
    @staticmethod    
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params
        
    def num_flat_features(self, x):
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features  


# set requies_grad=Fasle to avoid computation
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def to_cuda(m, x):
    assert isinstance(m, torch.nn.Module)
    device_id = torch.cuda.device_of(next(m.parameters()).data).idx
    if device_id == -1:
        return x
    return x.cuda(device_id)


def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.)

def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)

def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1) * mel2hz(melpoints )/ samplerate)

    fbank = np.zeros([nfilt, nfft//2+1])
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank


def lecun_normal_init_parameters(module):
    for p in module.parameters():
        data = p.data
        if data.dim() == 1:
            # bias
            data.zero_()
        elif data.dim() == 2:
            # linear weight
            n = data.size(1)
            stdv = 1. / math.sqrt(n)
            data.normal_(0, stdv)
        elif data.dim() == 4:
            # conv weight
            n = data.size(1)
            for k in data.size()[2:]:
                n *= k
            stdv = 1. / math.sqrt(n)
            data.normal_(0, stdv)
        else:
            raise NotImplementedError


# get output dim for latter BLSTM
def _get_cnn2l_odim(idim, in_channel=3, out_channel=128):
    idim = idim / in_channel
    return int(idim) * out_channel  # numer of channels
    
    
# get output dim for latter BLSTM
def _get_vgg2l_odim(idim, in_channel=3, out_channel=128):
    idim = idim / in_channel
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 1st max pooling
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 2nd max pooling
    return int(idim) * out_channel  # numer of channels


# get output dim for latter BLSTM
def _get_max_pooled_size(idim, out_channel=128, n_layers=2, ksize=2, stride=2):
    for _ in range(n_layers):
        idim = math.floor((idim - (ksize - 1) - 1) / stride)
    return idim  # numer of channels


def linear_tensor(linear, x):
    '''Apply linear matrix operation only for the last dimension of a tensor

    :param Link linear: Linear link (M x N matrix)
    :param Variable x: Tensor (D_1 x D_2 x ... x M matrix)
    :return:
    :param Variable x: Tensor (D_1 x D_2 x ... x N matrix)
    '''
    y = linear(x.contiguous().view((-1, x.size()[-1])))
    return y.view((x.size()[:-1] + (-1,)))


def mask_by_length(xs, length, fill=0):
    assert xs.size(0) == len(length)
    ret = Variable(xs.data.new(*xs.size()).fill_(fill))
    for i, l in enumerate(length):
        ret[i, :l] = xs[i, :l]
    return ret
    
    
def th_accuracy(y_all, pad_target, ignore_label):
    pad_pred = y_all.data.view(pad_target.size(
        0), pad_target.size(1), y_all.size(1)).max(2)[1]
    mask = pad_target.data != ignore_label
    numerator = torch.sum(pad_pred.masked_select(
        mask) == pad_target.data.masked_select(mask))
    denominator = torch.sum(mask)
    return float(numerator) / float(denominator)
    

def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].data.new(
        n_batch, max_len, * xs[0].size()[1:]).zero_() + pad_value

    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]

    return pad


def set_forget_bias_to_one(bias):
    n = bias.size(0)
    start, end = n // 4, n // 2
    bias.data[start:end].fill_(1.)
    

def end_detect(ended_hyps, i, M=3, D_end=np.log(1 * np.exp(-10))):
    '''End detection

    desribed in Eq. (50) of S. Watanabe et al
    "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition"

    :param ended_hyps:
    :param i:
    :param M:
    :param D_end:
    :return:
    '''
    if len(ended_hyps) == 0:
        return False
    count = 0
    best_hyp = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[0]
    for m in six.moves.range(M):
        # get ended_hyps with their length is i - m
        hyp_length = i - m
        hyps_same_length = [x for x in ended_hyps if len(x['yseq']) == hyp_length]
        if len(hyps_same_length) > 0:
            best_hyp_same_length = sorted(hyps_same_length, key=lambda x: x['score'], reverse=True)[0]
            if best_hyp_same_length['score'] - best_hyp['score'] < D_end:
                count += 1

    if count == M:
        return True
    else:
        return False
        
        
# TODO(takaaki-hori): add different smoothing methods
def label_smoothing_dist(odim, lsm_type, transcript=None, blank=0):
    '''Obtain label distribution for loss smoothing

    :param odim:
    :param lsm_type:
    :param blank:
    :param transcript:
    :return:
    '''
    if transcript is not None:
        with open(transcript) as f:
            trans_json = json.load(f)['utts']

    if lsm_type == 'unigram':
        assert transcript is not None, 'transcript is required for %s label smoothing' % lsm_type
        labelcount = np.zeros(odim)
        for k, v in trans_json.items():
            ids = np.array([int(n) for n in v['output'][0]['tokenid'].split()])
            # to avoid an error when there is no text in an uttrance
            if len(ids) > 0:
                labelcount[ids] += 1
        labelcount[odim - 1] = len(transcript)  # count <eos>
        labelcount[labelcount == 0] = 1  # flooring
        labelcount[blank] = 0  # remove counts for blank
        labeldist = labelcount.astype(np.float32) / np.sum(labelcount)
    else:
        logging.error(
            "Error: unexpected label smoothing type: %s" % lsm_type)
        sys.exit()

    return labeldist
    
    
def init_net(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
	
	
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
