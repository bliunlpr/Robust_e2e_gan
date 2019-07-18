import logging
import math
import sys
import random
from jiwer import wer

import numpy as np
import six
import torch
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from model import e2e_attention
from model.e2e_ctc import CTCPrefixScore
from model.e2e_common import end_detect, mask_by_length, to_cuda, pad_list, th_accuracy

CTC_SCORING_RATIO = 1.5
KENLM_SCORING_RATIO = 1.5
MAX_DECODER_OUTPUT = 5


# ------------- Decoder Network ----------------------------------------------------------------------------------------
class Decoder(torch.nn.Module):
    def __init__(self, eprojs, odim, dlayers, dunits, sos, eos, att, verbose=0,
                 char_list=None, labeldist=None, lsm_weight=0., fusion=None, rnnlm=None, 
                 model_unit='char', space_loss_weight=0.1):
        super(Decoder, self).__init__()
        self.dunits = dunits
        self.dlayers = dlayers
        self.embed = torch.nn.Embedding(odim, dunits)
        self.decoder = torch.nn.ModuleList()
        self.fusion = fusion
        self.rnnlm = rnnlm          
        self.decoder += [torch.nn.LSTMCell(dunits + eprojs, dunits)]
        for l in six.moves.range(1, self.dlayers):
            self.decoder += [torch.nn.LSTMCell(dunits, dunits)]
        self.ignore_id = -1
        
        if self.fusion == 'deep_fusion' and self.rnnlm is not None:
            lm_units = 650
            self.gate_linear = torch.nn.Linear(lm_units, 1)
            self.output = torch.nn.Linear(dunits + lm_units, odim)
        elif self.fusion == 'cold_fusion' and self.rnnlm is not None:
            lm_units = 650
            lm_project_units = dunits
            self.lm_linear = torch.nn.Linear(odim, lm_project_units)
            self.gate_linear = torch.nn.Linear(dunits + lm_project_units, lm_project_units)
            self.output = torch.nn.Linear(dunits + lm_project_units, odim)
        else:
            self.output = torch.nn.Linear(dunits, odim)

        self.loss = None
        self.att = att
        self.dunits = dunits
        self.sos = sos
        self.eos = eos
        self.verbose = verbose
        self.char_list = char_list
        
        # for word model_unit smoothing
        if model_unit == 'char':
            self.space_loss_weight = 0
        elif model_unit == 'word':
            self.space_loss_weight = space_loss_weight
            self.space_output = torch.nn.Linear(lm_units, 2)
                
        # for label smoothing
        self.labeldist = labeldist
        self.vlabeldist = None
        self.lsm_weight = lsm_weight

    def zero_state(self, hpad):
        return Variable(hpad.data.new(hpad.size(0), self.dunits).zero_())

    def forward(self, hpad, hlen, ys, scheduled_sampling_rate):
        '''Decoder forward

        :param hs:
        :param ys:
        :return:
        '''
        hpad = mask_by_length(hpad, hlen, 0)
        hlen = list(map(int, hlen))

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = Variable(ys[0].data.new([self.eos]))
        sos = Variable(ys[0].data.new([self.sos]))
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        pad_ys_in = pad_list(ys_in, self.eos)
        pad_ys_out = pad_list(ys_out, self.ignore_id)
        # get dim, length info
        batch = pad_ys_out.size(0)
        olength = pad_ys_out.size(1)
        ##logging.info(self.__class__.__name__ + ' input lengths:  ' + str(hlen))
        ##logging.info(self.__class__.__name__ + ' output lengths: ' + str([y.size(0) for y in ys_out]))

         # initialization
        c_list = [self.zero_state(hpad)]
        z_list = [self.zero_state(hpad)]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hpad))
            z_list.append(self.zero_state(hpad))
        att_w = None
        z_all = []
        y_all = []
        self.att.reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.embed(pad_ys_in)  # utt x olen x zdim
        
        rnnlm_state_prev = None
        # loop for an output sequence                 
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hpad, hlen, z_list[0], att_w)
            if random.random() < scheduled_sampling_rate and i > 0:
                topv, topi = y_i.topk(1)
                topi = topi.squeeze(1)
                ey_top = self.embed(topi)  # utt x zdim
                ey = torch.cat((ey_top, att_c), dim=1)  # utt x (zdim + hdim)              
            else:
                topi = pad_ys_in[:, i]
                ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)                              
            z_list[0], c_list[0] = self.decoder[0](ey, (z_list[0], c_list[0]))    
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
                    
            if self.fusion == 'deep_fusion' and self.rnnlm is not None:
                rnnlm_state, lm_scores = self.rnnlm.predict(rnnlm_state_prev, topi)
                lm_state = rnnlm_state['h2']        
                gi = F.sigmoid(self.gate_linear(lm_state))
                output_in = torch.cat((z_list[-1], gi * lm_state), dim=1)            
                rnnlm_state_prev = rnnlm_state  
            elif self.fusion == 'cold_fusion' and self.rnnlm is not None:
                rnnlm_state, lm_scores = self.rnnlm.predict(rnnlm_state_prev, topi)
                lm_state = F.relu(self.lm_linear(lm_scores))       
                gi = F.sigmoid(self.gate_linear(torch.cat((lm_state, z_list[-1]), dim=1)))
                output_in = torch.cat((z_list[-1], gi * lm_state), dim=1)            
                rnnlm_state_prev = rnnlm_state                                       
            else:
                output_in = z_list[-1]
            y_i = self.output(output_in)                     
            y_all.append(y_i)                   
            z_all.append(z_list[-1])          

        y_all = torch.stack(y_all, dim=0).transpose(0, 1).contiguous().view(batch * olength, -1)                    
        self.loss = F.cross_entropy(y_all, pad_ys_out.view(-1),
                                    ignore_index=self.ignore_id,
                                    size_average=True)
        # -1: eos, which is removed in the loss computation
        self.loss *= (np.mean([len(x) for x in ys_in]) - 1)
                
        acc = th_accuracy(y_all, pad_ys_out, ignore_label=self.ignore_id)
        if self.labeldist is not None:
            if self.vlabeldist is None:
                self.vlabeldist = to_cuda(self, Variable(torch.from_numpy(self.labeldist)))
            loss_reg = - torch.sum((F.log_softmax(y_all, dim=1) * self.vlabeldist).view(-1), dim=0) / len(ys_in)
            self.loss = (1. - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg

        return self.loss, acc


    def recognize_beam(self, h, lpz, recog_args, char_list, rnnlm=None, fstlm=None):
        '''beam search implementation

        :param Variable h:
        :param Namespace recog_args:
        :param char_list:
        :return:
        '''
        logging.info('input lengths: ' + str(h.size(0)))
        # initialization
        c_list = [self.zero_state(h.unsqueeze(0))]
        z_list = [self.zero_state(h.unsqueeze(0))]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(h.unsqueeze(0)))
            z_list.append(self.zero_state(h.unsqueeze(0)))
        a = None
        self.att.reset()  # reset pre-computation of h

        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprate sos
        y = self.sos        
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list,
                   'z_prev': z_list, 'a_prev': a, 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'z_prev': z_list, 'a_prev': a}
        if fstlm is not None:
            hyp['fstlm_prev'] = None
            
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz.cpu().numpy(), 0, self.eos, np)
            hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
            hyp['ctc_score_prev'] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []
        
        rnnlm_state_prev = None    
        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy.unsqueeze(1)
                vy[0] = hyp['yseq'][i]
                ey = self.embed(vy)           # utt list (1) x zdim
                ey.unsqueeze(0)
                att_c, att_w = self.att(h.unsqueeze(0), [h.size(0)], hyp['z_prev'][0], hyp['a_prev'])
                ey = torch.cat((ey, att_c), dim=1)   # utt(1) x (zdim + hdim)
                z_list[0], c_list[0] = self.decoder[0](ey, (hyp['z_prev'][0], hyp['c_prev'][0]))
                for l in six.moves.range(1, self.dlayers):
                    z_list[l], c_list[l] = self.decoder[l](
                        z_list[l - 1], (hyp['z_prev'][l], hyp['c_prev'][l]))
                
                if self.fusion == 'deep_fusion' and self.rnnlm is not None:
                    rnnlm_state, lm_scores = self.rnnlm.predict(rnnlm_state_prev, vy)
                    lm_state = rnnlm_state['h2']        
                    gi = F.sigmoid(self.gate_linear(lm_state))
                    output_in = torch.cat((z_list[-1], gi * lm_state), dim=1)            
                    rnnlm_state_prev = rnnlm_state  
                elif self.fusion == 'cold_fusion' and self.rnnlm is not None:
                    rnnlm_state, lm_scores = self.rnnlm.predict(rnnlm_state_prev, vy)
                    lm_state = F.relu(self.lm_linear(lm_scores))       
                    gi = F.sigmoid(self.gate_linear(torch.cat((lm_state, z_list[-1]), dim=1)))
                    output_in = torch.cat((z_list[-1], gi * lm_state), dim=1)            
                    rnnlm_state_prev = rnnlm_state                                       
                else:
                    output_in = z_list[-1]
                
                # get nbest local scores and their ids
                local_att_scores = F.log_softmax(self.output(output_in), dim=1).data
                if fstlm:
                    '''local_best_scores, local_best_ids = torch.topk(local_att_scores, kenlm_beam, dim=1)
                    kenlm_state, kenlm_scores = kenlm.predict(hyp['kenlm_prev'], local_best_ids[0])                
                    local_scores = local_att_scores[:, local_best_ids[0]] + recog_args.lm_weight * torch.from_numpy(kenlm_scores)
                    local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]'''
                    fstlm_state, local_lm_scores = fstlm.predict(hyp['fstlm_prev'], vy)
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores                    
                elif rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1)
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'])
                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                        + ctc_weight * to_cuda(self, torch.from_numpy(ctc_scores - hyp['ctc_score_prev']))
                    if rnnlm:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                    elif fstlm:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                    ##print('vy', vy)
                    ##print('local_att_scores', local_scores, local_scores.shape)
                    ##print('local_lm_scores', recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]])
                    local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    ##if not kenlm:
                    local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    # [:] is needed!
                    new_hyp['z_prev'] = z_list[:]
                    new_hyp['c_prev'] = c_list[:]
                    new_hyp['a_prev'] = att_w[:]
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if fstlm:
                        new_hyp['fstlm_prev'] = fstlm_state
                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))
            logging.debug(
                'best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remeined hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            for hyp in hyps:
                logging.debug(
                    'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

            logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]
        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))

        # remove sos
        return nbest_hyps

    def calculate_all_attentions(self, hpad, hlen, ys):
        '''Calculate all of attentions

        :return: numpy array format attentions
        '''
        hlen = list(map(int, hlen))
        hpad = mask_by_length(hpad, hlen, 0)
        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = Variable(ys[0].data.new([self.eos]))
        sos = Variable(ys[0].data.new([self.sos]))
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        pad_ys_in = pad_list(ys_in, self.eos)
        pad_ys_out = pad_list(ys_out, self.ignore_id)

        # get length info
        olength = pad_ys_out.size(1)

        # initialization
        c_list = [self.zero_state(hpad)]
        z_list = [self.zero_state(hpad)]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hpad))
            z_list.append(self.zero_state(hpad))
        att_w = None
        att_ws = []
        self.att.reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.embed(pad_ys_in)  # utt x olen x zdim
        rnnlm_state_prev = None
        
        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hpad, hlen, z_list[0], att_w)
            if i > 0:
                topv, topi = y_i.topk(1)
                topi = topi.squeeze(1)
                ey_top = self.embed(topi)  # utt x zdim
                ey = torch.cat((ey_top, att_c), dim=1)  # utt x (zdim + hdim)              
            else:
                topi = pad_ys_in[:, i]
                ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list[0], c_list[0] = self.decoder[0](ey, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            att_ws.append(att_w)
            
            if self.fusion == 'deep_fusion' and self.rnnlm is not None:
                rnnlm_state, lm_scores = self.rnnlm.predict(rnnlm_state_prev, topi)
                lm_state = rnnlm_state['h2']        
                gi = F.sigmoid(self.gate_linear(lm_state))
                output_in = torch.cat((z_list[-1], gi * lm_state), dim=1)            
                rnnlm_state_prev = rnnlm_state  
            elif self.fusion == 'cold_fusion' and self.rnnlm is not None:
                rnnlm_state, lm_scores = self.rnnlm.predict(rnnlm_state_prev, topi)
                lm_state = F.relu(self.lm_linear(lm_scores))       
                gi = F.sigmoid(self.gate_linear(torch.cat((lm_state, z_list[-1]), dim=1)))
                output_in = torch.cat((z_list[-1], gi * lm_state), dim=1)            
                rnnlm_state_prev = rnnlm_state                                       
            else:
                output_in = z_list[-1]
            y_i = self.output(output_in)                                     
            
        # convert to numpy array with the shape (B, Lmax, Tmax)
        if isinstance(self.att, e2e_attention.AttLoc2D):
            # att_ws => list of previous concate attentions
            att_ws = torch.stack([aw[:, -1] for aw in att_ws], dim=1).data.cpu().numpy()
        elif isinstance(self.att, (e2e_attention.AttCov, e2e_attention.AttCovLoc)):
            # att_ws => list of list of previous attentions
            att_ws = torch.stack([aw[-1] for aw in att_ws], dim=1).data.cpu().numpy()
        elif isinstance(self.att, e2e_attention.AttLocRec):
            # att_ws => list of tuple of attention and hidden states
            att_ws = torch.stack([aw[0] for aw in att_ws], dim=1).data.cpu().numpy()
        elif isinstance(self.att, (e2e_attention.AttMultiHeadDot, e2e_attention.AttMultiHeadAdd, e2e_attention.AttMultiHeadLoc, e2e_attention.AttMultiHeadMultiResLoc)):
            # att_ws => list of list of each head attetion
            n_heads = len(att_ws[0])
            att_ws_sorted_by_head = []
            for h in six.moves.range(n_heads):
                att_ws_head = torch.stack([aw[h] for aw in att_ws], dim=1)
                att_ws_sorted_by_head += [att_ws_head]
            att_ws = torch.stack(att_ws_sorted_by_head, dim=1).data.cpu().numpy()
        else:
            # att_ws => list of attetions
            att_ws = torch.stack(att_ws, dim=1).data.cpu().numpy()
        return att_ws

