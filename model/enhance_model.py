import logging
import math
import sys
import six
import numpy as np
import functools

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable

from model.e2e_common import ModelBase, lecun_normal_init_parameters, set_forget_bias_to_one, to_cuda, get_filterbanks
from model.e2e_common import get_norm_layer, init_net 
from model.e2e_encoder import BLSTMP, BLSTM, CNN2L, VGG2L


class SequenceWise(torch.nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        n, t = x.size(0), x.size(1)
        x = x.view(n * t, -1)
        x = self.module(x)
        x = x.view(n, t, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr
        
        
class EnhanceModel(ModelBase):
    def __init__(self, args):
        super(EnhanceModel, self).__init__()
        self.opt = args
        idim = args.idim 
        odim = args.idim 
        self.enhance_type = args.enhance_type
        self.verbose = args.verbose
        enhance_layers = args.enhance_layers
        enhance_units = args.enhance_units
        enhance_projs = args.enhance_projs
        dropout = args.dropout_rate
        subsample_type = args.subsample_type
        
        if self.enhance_type == 'unet_128' or self.enhance_type == 'unet_256':
            input_nc = args.enhance_input_nc
            output_nc = args.enhance_output_nc
            ngf = args.enhance_ngf
            init_type='normal'
            init_gain=0.02
            norm_layer = get_norm_layer(norm_type=args.enhance_norm)
        
        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer)
        subsample = np.ones(args.enhance_layers + 1, dtype=np.int)
        if args.enhance_type == 'blstmp' or args.enhance_type == 'cnnblstmp':
            ss = args.subsample.split("_")
            for j in range(min(args.enhance_layers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                'Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.')
        logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample
                           
        if self.enhance_type == 'blstm':
            self.enc1 = BLSTM(idim, enhance_layers, enhance_units, enhance_projs, dropout)
            logging.info('BLSTM without projection for enhance_mask')
        elif self.enhance_type == 'blstmp':
            self.enc1 = BLSTMP(idim, enhance_layers, enhance_units, enhance_projs, 
                               self.subsample, subsample_type, dropout)
            logging.info('BLSTM with every-layer projection for enhance_mask')
        elif self.enhance_type == 'unet_128':
            self.enc1 = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=dropout)
            init_net(self.enc1, init_type, init_gain)
            logging.info('Use unet_128 for enhance_mask')
        elif self.enhance_type == 'unet_256':
            self.enc1 = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=dropout)
            init_net(self.enc1, init_type, init_gain)
            logging.info('Use unet_256 for enhance_mask')
        elif self.enhance_type == 'vggblstmp':
            self.enc1 = VGG2L(in_channel)
            self.enc2 = BLSTMP(_get_vgg2l_odim(idim, in_channel=in_channel),
                               enhance_layers, enhance_units, enhance_projs,
                               self.subsample, subsample_type, dropout)
            logging.info('Use CNN-VGG + BLSTMP for enhance_mask')
        elif self.enhance_type == 'vggblstm':
            self.enc1 = VGG2L(in_channel)
            self.enc2 = BLSTM(_get_vgg2l_odim(idim, in_channel=in_channel),
                              enhance_layers, enhance_units, enhance_projs, dropout)
            logging.info('Use CNN-VGG + BLSTM for enhance_mask')        
        else:
            logging.error("Error: need to specify an appropriate enhance_mask archtecture")
            sys.exit()
            
        fully_connected = torch.nn.Sequential(
            ##torch.nn.BatchNorm1d(enhance_projs),
            torch.nn.Linear(enhance_projs, odim, bias=False)
        )
        self.fc = torch.nn.Sequential(
            SequenceWise(fully_connected),
        )
        
        # weight initialization
        self.init_like_chainer()
        # additional forget-bias init in encoder ?
        # for m in self.modules():
        #     if isinstance(m, torch.nn.LSTM):
        #         for name, p in m.named_parameters():
        #             if "bias_ih" in name:
        #                 set_forget_bias_to_one(p)
        
    def forward(self, mix_inputs, mix_log_inputs, input_sizes, clean_inputs=None, cos_angles=None):
        '''Encoder forward
        :param xs:
        :param ilens:
        :return:
        '''                               
        mix_inputs = to_cuda(self, mix_inputs)
        mix_log_inputs = to_cuda(self, mix_log_inputs)        
        ilens = to_cuda(self, input_sizes)
        
        if self.enhance_type == 'blstm':
            xs, hlens = self.enc1(mix_log_inputs, ilens)
        elif self.enhance_type == 'blstmp':
            xs, hlens = self.enc1(mix_log_inputs, ilens)
        elif self.enhance_type == 'unet_128':
            xs, hlens = self.enc1(mix_log_inputs, ilens)           
        elif self.enhance_type == 'unet_256':
            xs, hlens = self.enc1(mix_log_inputs, ilens)    
        elif self.enhance_type == 'vggblstmp':
            xs, hlens = self.enc1(mix_log_inputs, ilens)
            xs, hlens = self.enc2(xs, hlens)
        elif self.enhance_type == 'vggblstm':
            xs, hlens = self.enc1(mix_log_inputs, ilens)
            xs, hlens = self.enc2(xs, hlens)
        else:
            logging.error("Error: need to specify an appropriate enhance archtecture")
            sys.exit()
        
        if self.enhance_type == 'unet_128' or self.enhance_type == 'unet_256':
            linear_out = xs.squeeze(1)           
        else:              
            linear_out = self.fc(xs)     
        out = torch.sigmoid(linear_out) 
        mask = to_cuda(self, torch.ByteTensor(out.size()).fill_(0))
        for i, length in enumerate(ilens):
            length = length.item()
            if (mask[i].size(0) - length) > 0:
                mask[i].narrow(0, length, mask[i].size(0) - length).fill_(1)
        out = out.masked_fill(mask, 0) 
        enhance_out = out * mix_inputs 
        
        if clean_inputs is not None:
            clean_inputs = to_cuda(self, clean_inputs)
            cos_angles = to_cuda(self, cos_angles)     
            ##loss = F.mse_loss(enhance_out, clean_inputs * cos_angles, size_average=False) 
            loss = F.l1_loss(enhance_out, clean_inputs * cos_angles, size_average=False)            
            loss /= torch.sum(ilens).float()        
            return loss, enhance_out
        else:
            return enhance_out
        
    def init_like_chainer(self):
        """Initialize weight like chainer

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)

        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """
        lecun_normal_init_parameters(self)
    
    def calculate_all_specgram(self, mix_inputs, mix_log_inputs, input_sizes):
        torch.set_grad_enabled(False)        
        mix_inputs = to_cuda(self, mix_inputs)
        mix_log_inputs = to_cuda(self, mix_log_inputs)
        ilens = to_cuda(self, input_sizes)
        
        if self.enhance_type == 'blstm':
            xs, hlens = self.enc1(mix_log_inputs, ilens)
        elif self.enhance_type == 'blstmp':
            xs, hlens = self.enc1(mix_log_inputs, ilens)
        elif self.enhance_type == 'unet_128':
            xs, hlens = self.enc1(mix_log_inputs, ilens)           
        elif self.enhance_type == 'unet_256':
            xs, hlens = self.enc1(mix_log_inputs, ilens)    
        elif self.enhance_type == 'vggblstmp':
            xs, hlens = self.enc1(mix_log_inputs, ilens)
            xs, hlens = self.enc2(xs, hlens)
        elif self.enhance_type == 'vggblstm':
            xs, hlens = self.enc1(mix_log_inputs, ilens)
            xs, hlens = self.enc2(xs, hlens)
        else:
            logging.error("Error: need to specify an appropriate enhance archtecture")
            sys.exit()

        linear_out = self.fc(xs)
        out = F.sigmoid(linear_out)    
        enhanced_out = out * mix_inputs            
        torch.set_grad_enabled(True)

        return enhanced_out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=0.0):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, x, ilens):
        """
        :param x: The input of size BxCxDxT
        :param ilens: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        x = self.model(x)
        return x, ilens


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=0.0):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout > 0.0:
                model = down + [submodule] + up + [nn.Dropout(use_dropout)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
            