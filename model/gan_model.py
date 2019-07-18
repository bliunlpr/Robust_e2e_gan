import logging
import math
import sys
import numpy as np
import six
import functools
from progressbar import ProgressBar

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

from model.e2e_common import ModelBase, lecun_normal_init_parameters, set_forget_bias_to_one, to_cuda
        

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
	
	
# Defines the PatchGAN discriminator with the specified arguments.
def init_NLayerDiscriminator(input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):	
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
		    use_bias = norm_layer == nn.InstanceNorm2d

    kw = 4
    padw = 1
    sequence = [
    	nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
    	nn.LeakyReLU(0.2, True)
    ]
    
    nf_mult = 1
    nf_mult_prev = 1
    for n in range(1, n_layers):
        nf_mult_prev = nf_mult
        nf_mult = min(2**n, 8)
        sequence += [
    		    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
    				  kernel_size=kw, stride=2, padding=padw, bias=use_bias),
    		    norm_layer(ndf * nf_mult),
    		    nn.LeakyReLU(0.2, True)
    	  ]

    nf_mult_prev = nf_mult
    nf_mult = min(2**n_layers, 8)
    sequence += [
        nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
    		  kernel_size=kw, stride=1, padding=padw, bias=use_bias),
        norm_layer(ndf * nf_mult),
        nn.LeakyReLU(0.2, True)
      ]
    
    sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
    
    if use_sigmoid:
        sequence += [nn.Sigmoid()]
    
    model = nn.Sequential(*sequence)
    return model
		

def init_PixelDiscriminator(input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d
    
    sequence = [
    	  nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
    	  nn.LeakyReLU(0.2, True),
    	  nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
    	  norm_layer(ndf * 2),
    	  nn.LeakyReLU(0.2, True),
    	  nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
    
    if use_sigmoid:
        sequence.append(nn.Sigmoid())
    
    model = nn.Sequential(*sequence)
    return model

		
class GANModel(ModelBase):
    def __init__(self, args):
        super(GANModel, self).__init__()
        self.opt = args
        ndf = args.ndf 
        norm = args.norm_D
        input_nc = args.input_nc       
        n_layers_D = args.n_layers_D 		
        use_sigmoid = args.no_lsgan
        init_type='normal'
        init_gain=0.02
        norm_layer = get_norm_layer(norm_type=norm)
        if args.netD_type == 'basic':
            self.model = init_NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
        elif args.netD_type == 'n_layers':
            self.model = init_NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
        elif args.netD_type == 'pixel':     # classify if each pixel is real or fake
            self.model = init_PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
        else:
            raise NotImplementedError('Discriminator model name [%s] is not recognized' % args.netD_type)
        init_net(self.model, init_type, init_gain)
        
    def forward(self, input):
        input = to_cuda(self, input)
        if len(input.shape) == 3:
            input = input.unsqueeze(1)  
        return self.model(input)	
    
           
# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

