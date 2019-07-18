import logging
import math
import sys
from progressbar import ProgressBar
import numpy as np
import six
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable

from model.e2e_common import ModelBase, lecun_normal_init_parameters, set_forget_bias_to_one, to_cuda


fbank80_offset = [1, 2, 3, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 28, 29, 31, 33, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 55, 57, 59, 62, 65, 67, 70, 73, 76, 79, 82, 85, 89, 92, 96, 99, 103, 107, 111, 115, 120, 124, 129, 133, 138, 143, 148, 154, 159, 165, 171, 177, 183, 189, 196, 203, 210, 217, 224, 232, 240]
fbank80_weight = ['0.50398 0.135725', '0.864275', '0.830078', '0.169922 0.574852', '0.425148 0.366303', '0.633697 0.20108', '0.79892 0.0761817', '0.923818', '0.988912', '0.0110882 0.93684', '0.0631596 0.917753', '0.0822473 0.929643', '0.0703572 0.970687 0.0392122', '0.0293132 0.960788 0.133686', '0.866314 0.252702', '0.747298 0.394967', '0.605033 0.559286', '0.440714 0.744549', '0.255451 0.949737 0.173894', '0.0502626 0.826106 0.416138', '0.583862 0.675643', '0.324357 0.95164 0.243419', '0.0483595 0.756581 0.550292', '0.449708 0.871642 0.20687', '0.128358 0.79313 0.55542', '0.44458 0.916768 0.290421', '0.0832323 0.709579 0.675923 0.0728174', '0.324077 0.927183 0.480703', '0.519297 0.899179 0.327876', '0.100821 0.672124 0.766441 0.214539', '0.233559 0.785461 0.671853 0.138072', '0.328147 0.861928 0.612924 0.0961203', ' 0.387076 0.90388 0.587403 0.0865262', '0.412597 0.913474 0.593246 0.107348', '0.406754 0.892652 0.628601 0.1568', '0.371399 0.8432 0.691757 0.23327', '0.308243 0.76673 0.781159 0.335252', '0.218841 0.664748 0.895376 0.461372 0.0330859', '0.104624 0.538628 0.966914 0.610371 0.193086', '0.389629 0.806914 0.781084 0.374247', '0.218916 0.625753 0.972431 0.575525 0.183407', '0.0275686 0.424475 0.816593 0.795964 0.413082 0.0346561', '0.204036 0.586918 0.965344 0.660585 0.290774', '0.339415 0.709226 0.925118 0.563533 0.205923', '0.0748823 0.436467 0.794077 0.85221 0.502296 0.156113', '0.14779 0.497704 0.843887 0.813583 0.474613 0.139153', '0.186417 0.525387 0.860847 0.807114 0.478431 0.153043', '0.192886 0.521569 0.846957 0.830874 0.511874 0.195962', '0.169126 0.488126 0.804038 0.883103 0.573226 0.266271', '0.116897 0.426774 0.733729 0.962186 0.660919 0.362424 0.0666366', '0.0378145 0.339081 0.637576 0.933363 0.773522 0.483028 0.195107', '0.226478 0.516972 0.804893 0.90971 0.626798 0.346327 0.0682598', '0.0902901 0.373202 0.653673 0.93174 0.792546 0.519152 0.248036', '0.207454 0.480848 0.751964 0.97917 0.712505 0.448012 0.185653', '0.0208296 0.287495 0.551988 0.814347 0.925392 0.667197 0.411041 0.156884', '0.0746079 0.332803 0.588959 0.843116 0.9047 0.65446 0.406131 0.159683', '0.0953 0.34554 0.593869 0.840317 0.91509 0.67232 0.431356 0.19216', '0.0849101 0.32768 0.568644 0.80784 0.954714 0.718987 0.484961 0.25261 0.0218999', '0.0452857 0.281013 0.51504 0.74739 0.9781 0.792823 0.565344 0.33945 0.115112', '0.207177 0.434656 0.66055 0.884888 0.89231 0.671028 0.451246 0.232936 0.0160764', '0.10769 0.328972 0.548754 0.767064 0.983924 0.800661 0.58666 0.374068 0.162842', '0.199339 0.41334 0.625932 0.837158 0.952989 0.744483 0.537307 0.331441 0.126871', '0.0470108 0.255517 0.462693 0.668559 0.873129 0.923582 0.721553 0.52077 0.321219 0.122887', '0.0764177 0.278447 0.47923 0.678781 0.877113 0.925758 0.729827 0.535058 0.34145 0.148984', '0.0742418 0.270173 0.464942 0.65855 0.851016 0.957665 0.76746 0.578354 0.390353 0.203429 0.0175692', '0.042335 0.23254 0.421646 0.609647 0.796571 0.982431 0.832771 0.649015 0.466287 0.284595 0.103902', '0.167229 0.350985 0.533713 0.715405 0.896098 0.924209 0.745502 0.567774 0.391003 0.215205 0.0403425', '0.075791 0.254498 0.432226 0.608997 0.784795 0.959657 0.866424 0.693428 0.52134 0.350168 0.179883 0.0104923', '0.133576 0.306572 0.47866 0.649832 0.820117 0.989508 0.841983 0.674333 0.507556 0.341624 0.176537 0.0122738', '0.158017 0.325667 0.492444 0.658376 0.823463 0.987726 0.848848 0.686231 0.524424 0.363421 0.203212 0.043793', '0.151152 0.313769 0.475575 0.636579 0.796788 0.956207 0.885148 0.727271 0.570161 0.413805 0.258195 0.103331', '0.114852 0.272729 0.429839 0.586195 0.741805 0.896668 0.949201 0.795796 0.64311 0.491149 0.339885 0.189318 0.0394479', '0.0507992 0.204204 0.35689 0.508851 0.660115 0.810682 0.960552 0.890275 0.741777 0.593948 0.446803 0.300305 0.154476 0.00928814', '0.109725 0.258223 0.406052 0.553197 0.699695 0.845524 0.990712 0.864748 0.720856 0.57759 0.434958 0.292946 0.151554 0.010774', '0.135252 0.279144 0.42241 0.565042 0.707054 0.848446 0.989226 0.870601 0.731026 0.59205 0.453665 0.315872 0.178656 0.0420182', '0.129399 0.268974 0.40795 0.546335 0.684128 0.821344 0.957982 0.905957 0.770451 0.635516 0.501137 0.367315 0.234034 0.101296', '0.0940433 0.229549 0.364484 0.498863 0.632685 0.765966 0.898704 0.969108 0.837447 0.706321 0.575724 0.445648 0.316086 0.187044 0.0585033', '0.0308924 0.162553 0.293679 0.424276 0.554352 0.683914 0.812955 0.941497 0.930483 0.802956 0.675922 0.549395 0.423346 0.297791 0.172721 0.0481308', '0.0695167 0.197044 0.324078 0.450605 0.576654 0.702209 0.827279 0.951869 0.924005 0.800366 0.677192 0.554482 0.432223 0.310436 0.189099 0.0682135', '0.0759946 0.199634 0.322808 0.445518 0.567777 0.689564 0.810901 0.931786 0.947778 0.827786 0.70823 0.589118 0.470435 0.352189 0.234372 0.116971']


def get_filterbanks(nfilt=80, nfft=512):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)"""
    
    if nfilt == 80:
        fbank = np.zeros([nfilt, nfft//2+1])
        for j in range(0, nfilt):
            offset = fbank80_offset[j]
            weight = fbank80_weight[j]
            weight = weight.split()
            for i in range(len(weight)):
                fbank[j, i+offset] = float(weight[i])        
        return fbank
    else:
        print('nfilt must = 80, but get {}'.format(nfilt))
    
    
class FFTModel(ModelBase):
    def __init__(self, args):
        super(FFTModel, self).__init__()
        self.opt = args
        idim = args.idim         
        self.sum = np.zeros(shape=[1, idim], dtype=np.float32)
        self.sum_sq = np.zeros(shape=[1, idim], dtype=np.float32)
        self.fbank_cmvn = np.zeros(shape=[2, idim], dtype=np.float32)        
        self.cmvn_num = min(args.train_dataset_len, args.num_utt_cmvn)
        self.cmvn_processed_num = 0
        self.frame_count = 0
        
    def forward(self, xs, fft_cmvn=None):
        '''FbankModel forward
        :param xs:
        :return:
        '''
        ##self.fc = torch.exp(self.fc)
        xs = to_cuda(self, xs)        
        xs[xs <= 1e-7] = 1e-7
        out = torch.log(xs)        
        if fft_cmvn is not None:
            fft_cmvn = to_cuda(self, fft_cmvn)
            out = (out + fft_cmvn[0, :]) * fft_cmvn[1, :]        
        return out
        
    def compute_cmvn(self, inputs, input_sizes):
        if self.cmvn_processed_num == 0:  
            print(">> compute fbank_cmvn using {} utterance ".format(self.cmvn_num))
            self.pbar = ProgressBar().start()        
        features = self.forward(inputs)               
        if self.cmvn_processed_num < self.cmvn_num:
            for x in range(input_sizes.shape[0]):
                input_size = int(input_sizes[x]) 
                feature_mat = features[x].data.cpu().numpy()
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
            
            
class FbankModel(FFTModel):
    def __init__(self, args):
        super(FFTModel, self).__init__()
        self.opt = args
        idim = args.idim 
        odim = args.fbank_dim 
        filterbanks = get_filterbanks(nfilt=odim) 
        ##filterbanks[filterbanks <= 1e-3] = 1e-3
        ##log_filterbanks = np.log(filterbanks)
        if args.enhance_type == 'unet_128' or args.enhance_type == 'unet_256':
            idim = 256
            filterbanks = filterbanks[:, :256]
        self.fc = torch.nn.Parameter(torch.Tensor(idim, odim))   
        self.fc.data.copy_(torch.from_numpy(filterbanks.T))
        print('filterbanks, shape ', filterbanks.shape)        
        if args.fbank_opti_type == 'frozen':           
            self.fc.requires_grad_(False)    
        
        self.sum = np.zeros(shape=[1, args.fbank_dim], dtype=np.float32)
        self.sum_sq = np.zeros(shape=[1, args.fbank_dim], dtype=np.float32)
        self.fbank_cmvn = np.zeros(shape=[2, args.fbank_dim], dtype=np.float32)        
        self.cmvn_num = min(args.train_dataset_len, args.num_utt_cmvn)
        self.cmvn_processed_num = 0
        self.frame_count = 0
        
    def forward(self, xs, fbank_cmvn=None):
        '''FbankModel forward
        :param xs:
        :return:
        '''
        ##self.fc = torch.exp(self.fc)
        xs = to_cuda(self, xs)        
        xs = (xs ** 2)
        n, t = xs.size(0), xs.size(1)
        xs = xs.view(n * t, -1)
        xs = torch.mm(xs, self.fc)
        out = xs.view(n, t, -1)
        out[out <= 1e-7] = 1e-7
        out = torch.log(out)        
        if fbank_cmvn is not None:
            fbank_cmvn = to_cuda(self, fbank_cmvn)
            out = (out + fbank_cmvn[0, :]) * fbank_cmvn[1, :]        
        return out