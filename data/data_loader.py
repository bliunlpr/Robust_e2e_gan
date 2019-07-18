# encoding=utf8
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from data.audioparse import FbankFeatLabelParser


class SequentialDataset(Dataset, FbankFeatLabelParser):
    def __init__(self, args, data_dir, dict_file):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        utt_id /path/to/audio.wav
        ...
        :param args: Path to scp as describe above
        :param data_dir : Dictionary containing the delta_order, context_width, normalize_type and max_num_utt_cmvn
        :param dict_file: Dictionary containing the sample_rate, num_channel, window_size window_shift
        """
        self.args = args
        self.exp_path = args.exp_path
        
        self.feat_type = args.feat_type
        if self.feat_type.split('_')[0] == 'kaldi':
            self.speech_scp = os.path.join(data_dir, 'feats.scp')
            self.feat_len_scp = os.path.join(data_dir, 'kaldi_feat_len.scp')
        else:
            self.speech_scp = os.path.join(data_dir, 'wav.scp')
            self.feat_len_scp = os.path.join(data_dir, 'feat_len.scp')        
        with open(self.speech_scp, encoding='utf-8') as f:
            spe_ids = f.readlines()
        self.spe_ids = [x.strip().split(' ') for x in spe_ids]
        self.spe_size = len(self.spe_ids)
        
        if not os.path.exists(self.feat_len_scp):
            self.loading_feat_len(self.speech_scp, self.feat_len_scp)
        with open(self.feat_len_scp, encoding='utf-8') as f:
            feat_len_ids = f.readlines()
        self.feat_len_ids = {x.strip().split(' ')[0]: x.strip().split(' ')[1] for x in feat_len_ids}
        
        audio_lengths = [self.load_audio_feat_len(utt_id) for utt_id in self.spe_ids]
        hist, bin_edges = np.histogram(audio_lengths, bins="auto")
        audio_samples_indices = np.digitize(audio_lengths, bins=bin_edges)
        self.bins_to_samples = defaultdict(list)
        for idx, bin_id in enumerate(audio_samples_indices):
            self.bins_to_samples[bin_id].append(idx)
            
        self.delta_order         = args.delta_order
        self.left_context_width  = args.left_context_width
        self.right_context_width = args.right_context_width                        
        self.feat_size = 0
        for n in range(self.spe_size):
            wav_path = self.spe_ids[n][1]
            if self.feat_type.split('_')[0] == 'kaldi':
                in_feat = self.extract_kaldi_feat(wav_path, feat_type=self.feat_type)
            else:
                speech_wav = self.WaveData(wav_path)
                in_feat = self.extract_feat(speech_wav, feat_type=self.feat_type)
            in_feat = self.transform_feat(in_feat, None, self.delta_order, self.left_context_width, self.right_context_width)
            if in_feat is not None:
                self.feat_size = np.shape(in_feat)[1]
                break
        if self.feat_size <= 0:
            raise Exception('Wrong feat_size {}'.format(self.feat_size))
                
        self.normalize_type = args.normalize_type
        self.num_utt_cmvn   = args.num_utt_cmvn       
        if self.normalize_type == 1:
            self.cmvn = self.loading_cmvn()
        else:
            self.cmvn = None  
        
        self.utt2spk_file = os.path.join(data_dir, 'utt2spk')
        with open(self.utt2spk_file, encoding='utf-8') as f:
            utt2spk_ids = f.readlines()
        self.utt2spk_ids = {x.strip().split(' ')[0]: x.strip().split(' ')[1] for x in utt2spk_ids}
        
        self.model_unit = args.model_unit
        if self.model_unit == 'char':                 
            self.label_file = os.path.join(data_dir, 'text_char') 
        elif self.model_unit == 'word':    
            self.label_file = os.path.join(data_dir, 'text_word')
        if dict_file is not None:
            with open(dict_file, 'r', encoding='utf-8') as f:
                dictionary = f.readlines()
            char_list = [entry.split(' ')[0] for entry in dictionary]
            char_list.insert(0, '<blank>')             
            char_list.append('<eos>')
            self.char_list = char_list
        else:
            self.char_list = None        
        self.num_classes  = len(self.char_list)        
        
        super(SequentialDataset, self).__init__(self.label_file, self.char_list)
    
    def loading_feat_len(self, feats_scp, out_scp):
        print('load feat_len from {} to {}'.format(feats_scp, out_scp))
        fwrite = open(out_scp, 'w')
        with open(feats_scp, 'r') as fid:
            for line in fid:
                line_splits = line.strip().split()
                utt_id = line_splits[0] 
                wav_path = line_splits[1]
                try:
                    if self.feat_type.split('_')[0] == 'kaldi':
                        in_feat = self.extract_kaldi_feat(wav_path, feat_type=self.feat_type)
                    else:
                        speech_wav = self.WaveData(wav_path)
                        in_feat = self.extract_feat(speech_wav, feat_type=self.feat_type)                             
                    fwrite.write(utt_id + ' ' + str(in_feat.shape[0]) + '\n')  
                except:
                    print(line, 'error')                               
        fwrite.close()

    def loading_cmvn(self):
        if not os.path.isdir(self.exp_path):
            raise Exception(self.exp_path + ' isn.t a path!')
        cmvn_file = os.path.join(self.exp_path, 'cmvn.npy')
        if os.path.exists(cmvn_file):
            cmvn = np.load(cmvn_file)
            if cmvn.shape[1] == self.feat_size:
                print('load cmvn from {}'.format(cmvn_file))
            else:
                cmvn = self.compute_cmvn()
                np.save(cmvn_file, cmvn)
                print('original cmvn is wrong, so save new cmvn to {}'.format(cmvn_file))
        else:
            cmvn = self.compute_cmvn()
            np.save(cmvn_file, cmvn)
            print('save cmvn to {}'.format(cmvn_file))
        return cmvn
    
    def compute_cmvn(self):
        sum = np.zeros(shape=[1, self.feat_size], dtype=np.float32)
        sum_sq = np.zeros(shape=[1, self.feat_size], dtype=np.float32)
        cmvn = np.zeros(shape=[2, self.feat_size], dtype=np.float32)
        frame_count = 0
               
        cmvn_num = min(self.spe_size, self.num_utt_cmvn) 
        print(">> compute cmvn using {} utterance ".format(cmvn_num))        
        cmvn_rand_idx = np.random.permutation(self.spe_size)
        for n in tqdm(range(cmvn_num)):  
            audio_path = self.spe_ids[cmvn_rand_idx[n]][1]
            if self.feat_type.split('_')[0] == 'kaldi':
                in_feat = self.extract_kaldi_feat(audio_path, feat_type=self.feat_type)
                in_feat[in_feat <= 1e-7] = 1e-7
                in_feat = 10 * np.log10(in_feat)
            else:
                speech_wav = self.WaveData(audio_path)
                in_feat = self.extract_feat(speech_wav, feat_type=self.feat_type)
                in_feat[in_feat <= 1e-7] = 1e-7
                in_feat = 10 * np.log10(in_feat)
            feature_mat = self.transform_feat(in_feat, None, self.delta_order, left_context_width=0, right_context_width=0)
            if feature_mat is None:
                continue 
            sum_1utt = np.sum(feature_mat, axis=0)
            sum = np.add(sum, sum_1utt)
            feature_mat_square = np.square(feature_mat)
            sum_sq_1utt = np.sum(feature_mat_square, axis=0)
            sum_sq = np.add(sum_sq, sum_sq_1utt)
            frame_count += feature_mat.shape[0]            
        
        mean = sum / frame_count
        var = sum_sq / frame_count - np.square(mean)
        print (frame_count)
        print (mean)
        print (var)
        cmvn[0, :] = -mean
        cmvn[1, :] = 1 / np.sqrt(var)
        return cmvn
                        
    def __getitem__(self, index):
                           
        utt_key, utt_path = self.spe_ids[index]
        if len(utt_key.split('__')) == 2:
            clean_utt_key = utt_key.split('__')[0]
        if utt_key in self.utt2spk_ids:
            spk_id = self.utt2spk_ids[utt_key]
        elif clean_utt_key in self.utt2spk_ids:
            spk_id = self.utt2spk_ids[clean_utt_key]
            
        if self.feat_type.split('_')[0] == 'kaldi':
            in_feat = self.extract_kaldi_feat(utt_path, feat_type=self.feat_type)
            in_feat[in_feat <= 1e-7] = 1e-7
            log_in_feat = 10 * np.log10(in_feat)
        else:
            speech_wav = self.WaveData(utt_path)
            in_feat = self.extract_feat(speech_wav, feat_type=self.feat_type)
            in_feat[in_feat <= 1e-7] = 1e-7
            log_in_feat = 10 * np.log10(in_feat)
        in_feat = self.transform_feat(in_feat, None, self.delta_order, self.left_context_width, self.right_context_width) 
        log_in_feat = self.transform_feat(log_in_feat, self.cmvn, self.delta_order, self.left_context_width, self.right_context_width)  
        
        if utt_key in self.target_dict:          
            target_out = self.extract_label(utt_key)
        elif clean_utt_key in self.target_dict:          
            target_out = self.extract_label(clean_utt_key)
            
        in_feat = torch.FloatTensor(in_feat)
        log_in_feat = torch.FloatTensor(log_in_feat)
        target_out = torch.LongTensor(target_out)
        return utt_key, spk_id, in_feat, log_in_feat, target_out

    def __len__(self):
        return self.spe_size
        
    def load_audio_feat_len(self, path):
        utt_id, utt_path = path
        try:
            return int(self.feat_len_ids[utt_id])
        except:
            if self.feat_type.split('_')[0] == 'kaldi':
                in_feat = self.extract_kaldi_feat(utt_path, feat_type=self.feat_type)
            else:
                speech_wav = self.WaveData(utt_path)
                in_feat = self.extract_feat(speech_wav, feat_type=self.feat_type)
            return in_feat.shape[0]
        
    def get_feat_size(self):
        return self.feat_size
        
    def get_char_list(self):
        return self.char_list
    
    def get_num_classes(self):
        return self.num_classes
        
                
def _collate_fn(batch):
    batch = sorted(batch, key=lambda sample: sample[2].size(0), reverse=True)
    longest_sample = batch[0][2]
    freq_size = longest_sample.size(1)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(0)
    inputs = torch.zeros(minibatch_size, max_seqlength, freq_size)
    log_inputs = torch.zeros(minibatch_size, max_seqlength, freq_size)
    input_sizes = torch.IntTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    utt_ids = []
    spk_ids = []
    for x in range(minibatch_size):
        sample = batch[x]
        utt_id = sample[0]
        spk_id = sample[1]
        spect = sample[2]
        log_spect = sample[3]
        target = sample[4]
        utt_ids.append(utt_id)
        spk_ids.append(spk_id)
        seq_length = spect.size(0)
        inputs[x].narrow(0, 0, seq_length).copy_(spect)
        log_inputs[x].narrow(0, 0, seq_length).copy_(log_spect)
        input_sizes[x] = seq_length
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.LongTensor(targets)
    return utt_ids, spk_ids, inputs, log_inputs, targets, input_sizes, target_sizes


class SequentialDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(SequentialDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
        
        
class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        ids = []
        for bin, sample_idx in self.data_source.bins_to_samples.items():
            np.random.shuffle(sample_idx)
            ids.extend(sample_idx)
        self.bins = self.build_bins()

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        self.bins = self.build_bins()
        np.random.shuffle(self.bins)
        
    def build_bins(self):
        ids = []
        for bin, sample_idx in self.data_source.bins_to_samples.items():
            np.random.shuffle(sample_idx)
            ids.extend(sample_idx)
        bins = [ids[i:i + self.batch_size] for i in range(0, len(ids), self.batch_size)]
        return bins
