import logging
import math
import sys

import numpy as np
import six
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from model.e2e_encoder import Encoder
from model.e2e_decoder import Decoder
from model.e2e_ctc import CTC
from model.e2e_common import ModelBase, label_smoothing_dist, lecun_normal_init_parameters, set_forget_bias_to_one, to_cuda
from model.e2e_attention import NoAtt, AttDot, AttAdd, AttLoc, AttCov, AttLoc2D, AttLocRec, AttCovLoc
from model.e2e_attention import AttMultiHeadDot, AttMultiHeadAdd, AttMultiHeadLoc, AttMultiHeadMultiResLoc 
from model import lm, extlm, fsrnn


class E2E(ModelBase):
    def __init__(self, args):
        super(E2E, self).__init__()
        self.opt = args
        idim = args.fbank_dim 
        odim = args.odim 
        self.etype = args.etype
        self.verbose = args.verbose
        self.char_list = args.char_list
        ##self.outdir = args.outdir
        self.mtlalpha = args.mtlalpha

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1

        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer)
        subsample = np.ones(args.elayers + 1, dtype=np.int)
        if args.etype == 'blstmp' or args.etype == 'cnnblstmp':
            ss = args.subsample.split("_")
            for j in range(min(args.elayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                'Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.')
        logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample

        # label smoothing info
        if args.lsm_type:
            logging.info("Use label smoothing with " + args.lsm_type)
            ##labeldist = label_smoothing_dist(odim, args.lsm_type, args.char_list, transcript=args.train_text)
            labeldist = args.labeldist
        else:
            labeldist = None

        # encoder
        self.enc = Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs,
                           self.subsample, args.subsample_type, args.dropout_rate)
        # ctc
        self.ctc = CTC(odim, args.eprojs, args.dropout_rate)
        # attention
        if args.atype == 'noatt':
            self.att = NoAtt()
        elif args.atype == 'dot':
            self.att = AttDot(args.eprojs, args.dunits, args.adim)
        elif args.atype == 'add':
            self.att = AttAdd(args.eprojs, args.dunits, args.adim)
        elif args.atype == 'location':
            self.att = AttLoc(args.eprojs, args.dunits,
                              args.adim, args.aconv_chans, args.aconv_filts, 'softmax')
        elif args.atype == 'location2d':
            self.att = AttLoc2D(args.eprojs, args.dunits,
                                args.adim, args.awin, args.aconv_chans, args.aconv_filts)
        elif args.atype == 'location_recurrent':
            self.att = AttLocRec(args.eprojs, args.dunits,
                                 args.adim, args.aconv_chans, args.aconv_filts)
        elif args.atype == 'coverage':
            self.att = AttCov(args.eprojs, args.dunits, args.adim)
        elif args.atype == 'coverage_location':
            self.att = AttCovLoc(args.eprojs, args.dunits, args.adim,
                                 args.aconv_chans, args.aconv_filts)
        elif args.atype == 'multi_head_dot':
            self.att = AttMultiHeadDot(args.eprojs, args.dunits,
                                       args.aheads, args.adim, args.adim)
        elif args.atype == 'multi_head_add':
            self.att = AttMultiHeadAdd(args.eprojs, args.dunits,
                                       args.aheads, args.adim, args.adim)
        elif args.atype == 'multi_head_loc':
            self.att = AttMultiHeadLoc(args.eprojs, args.dunits,
                                       args.aheads, args.adim, args.adim,
                                       args.aconv_chans, args.aconv_filts)
        elif args.atype == 'multi_head_multi_res_loc':
            self.att = AttMultiHeadMultiResLoc(args.eprojs, args.dunits,
                                               args.aheads, args.adim, args.adim,
                                               args.aconv_chans, args.aconv_filts)
        else:
            logging.error(
                "Error: need to specify an appropriate attention archtecture")
            sys.exit()
            
        # rnnlm            
        try:
            if args.fusion == 'deep_fusion' or args.fusion == 'cold_fusion':
                if args.lmtype == 'rnnlm' and args.rnnlm:
                    rnnlm = lm.ClassifierWithState(lm.RNNLM(len(args.char_list), 300, 650))                
                    rnnlm.load_state_dict(torch.load(args.rnnlm, map_location=lambda storage, loc: storage))
                    print('load rnnlm from ', args.rnnlm)
                    rnnlm.eval()
                    for p in rnnlm.parameters():
                        p.requires_grad_(False)
                elif args.lmtype == 'fsrnnlm' and args.rnnlm:
                    rnnlm = lm.ClassifierWithState(
                              fsrnn.FSRNNLM(len(args.char_list), 300, args.fast_layers, args.fast_cell_size, 
                              args.slow_cell_size, args.zoneout_keep_h, args.zoneout_keep_c))
                    rnnlm.load_state_dict(torch.load(args.rnnlm, map_location=lambda storage, loc: storage))
                    print('load rnnlm from ', args.rnnlm)
                    rnnlm.eval()
                    for p in rnnlm.parameters():
                        p.requires_grad_(False)
                else:
                    rnnlm = None
            else:
                rnnlm = None
                fusion = None 
                model_unit = 'char'
                space_loss_weight = 0.0
        except:
            rnnlm = None
            fusion = None 
            model_unit = 'char'
            space_loss_weight = 0.0
        # decoder
        self.dec = Decoder(args.eprojs, odim, args.dlayers, args.dunits, self.sos, self.eos, 
                           self.att, self.verbose, self.char_list, labeldist, args.lsm_weight, 
                           fusion, rnnlm, model_unit, space_loss_weight)

        # weight initialization
        self.init_like_chainer()
        # additional forget-bias init in encoder ?
        # for m in self.modules():
        #     if isinstance(m, torch.nn.LSTM):
        #         for name, p in m.named_parameters():
        #             if "bias_ih" in name:
        #                 set_forget_bias_to_one(p)        
        
    def init_like_chainer(self):
        """Initialize weight like chainer

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)

        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """
        lecun_normal_init_parameters(self)

        # exceptions
        # embed weight ~ Normal(0, 1)
        self.dec.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for l in six.moves.range(len(self.dec.decoder)):
            set_forget_bias_to_one(self.dec.decoder[l].bias_ih)

    # x[i]: ('utt_id', {'ilen':'xxx',...}})
    def forward(self, inputs, targets, input_sizes, target_sizes, scheduled_sampling_rate=0.0):
        '''E2E forward

        :param data:
        :return:
        '''
        xpad = to_cuda(self, inputs)
        ilens = to_cuda(self, input_sizes)
        ys = []
        offset = 0
        for size in target_sizes:
            ys.append(targets[offset:offset + size])
            offset += size        
        ys = [to_cuda(self, y) for y in ys]

        # 1. encoder
        #xpad = pad_list(hs, 0.0)
        hpad, hlens = self.enc(xpad, ilens)

        # # 3. CTC loss
        if self.mtlalpha == 0:
            loss_ctc = None
        else:
            loss_ctc = self.ctc(hpad, hlens, ys)

        # 4. attention loss
        if self.mtlalpha == 1:
            loss_att = None
            acc = None
            space_acc = None
        else:
            loss_att, acc = self.dec(hpad, hlens, ys, scheduled_sampling_rate)

        return loss_ctc, loss_att, acc

    def recognize(self, x, recog_args, char_list, rnnlm=None, fstlm=None):
        '''E2E beam search

        :param x:
        :param recog_args:
        :param char_list:
        :return:
        '''
        prev = self.training
        self.eval()
        
        # subsample frame
        ##x = x[::self.subsample[0], :]
        ilen = [x.shape[1]]
        h = to_cuda(self, x)

        # 1. encoder
        # make a utt list (1) to use the same interface for encoder
        h, _ = self.enc(h, ilen)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(h).data[0]
        else:
            lpz = None

        # 2. decoder
        # decode the first utterance
        y = self.dec.recognize_beam(h[0], lpz, recog_args, char_list, rnnlm, fstlm)

        if prev:
            self.train()
        return y

    def calculate_all_attentions(self, inputs, targets, input_sizes, target_sizes):
        '''E2E attention calculation

        :param list data: list of dicts of the input (B)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
         :rtype: float ndarray
        '''
        torch.set_grad_enabled(False)
        xpad = to_cuda(self, inputs)
        ilens = to_cuda(self, input_sizes)
        ys = []
        offset = 0
        for size in target_sizes:
            ys.append(targets[offset:offset + size])
            offset += size        
        ys = [to_cuda(self, y) for y in ys]
        hpad, hlens = self.enc(xpad, ilens)

        # decoder
        att_ws = self.dec.calculate_all_attentions(hpad, hlens, ys)
        
        torch.set_grad_enabled(True)

        return att_ws
