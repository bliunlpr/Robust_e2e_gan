import os
import gzip
import numpy as np
import struct
import random
import math
import scipy.signal
import librosa
import torch
import scipy.io as sio
from python_speech_features import fbank, delta
import torchaudio
from data import extract_fbanks_module, kaldi_io
import decimal

    
def load_audio(path):
    try:
        sound, _ = torchaudio.load(path)
        sound = sound.numpy()
        if len(sound.shape) > 1:
            if sound.shape[1] == 1:
                sound = sound.squeeze()
            else:
                sound = sound.mean(axis=1)  # multiple channels, average
        sound = sound / 65536.
        return sound
    except:
        print ('load_audio {} failed'.format(path))
        return None 


def load_mat(filename, key = 'feat'):
    if os.path.exists(filename):
        ##mat = sio.loadmat(filename, appendmat = True, mat_dtype = True, struct_as_record = True, verify_compressed_data_integrity=False).values()[0]
        mat = sio.loadmat(filename)
        for key in ['rmr','trmr','filter']:
            if key in mat:
                mat = mat[key]
                break
        if mat.shape[0] > 1:
            mat = mat.reshape(mat.shape[0])
        elif mat.shape[1] > 1:
            mat = mat.reshape(mat.shape[1])
        if np.isnan(mat).sum() > 0:
            return None
        return mat
    else:
        print ('load_mat {} failed'.format(filename))
        return None


def remove_sp_frame(feature_mat, vad_id):
    real_length = feature_mat.shape[0]
    start_time = int(int(vad_id[0]) * real_length / int(vad_id[2]))
    end_time = int(int(vad_id[1]) * real_length / int(vad_id[2]))            
    feature_mat = feature_mat[start_time:end_time, :]
    return feature_mat


class Targetcounter(object):
    def __init__(self, target_path, label_num):
        self.target_dict = read_target_file(target_path)
        self.label_num = label_num    
    def compute_target_count(self):
        encoded_targets = np.concatenate(
            [self.encode(targets)
             for targets in self.target_dict.values()])

        #  count the number of occurences of each target
        count = np.bincount(encoded_targets,
                        minlength=self.label_num)
        return count
            
    def encode(self, targets):
        """ encode a target sequence """
        encoded_targets = list([int(x) for x in list(targets)])
        return np.array(encoded_targets, dtype=np.int)
        
        
def read_target_file(model_unit, target_path, char_list, map_oov=1):
    """
    read the file containing the state alignments
    Args:
        target_path: path to the alignment file
    Returns:
        A dictionary containing
            - Key: Utterance ID
            - Value: The state alignments as a space seperated string
    """
    odim = len(char_list)
    labelcount = np.zeros(odim)
    transcript_num = 0 
    blank = 0       
    target_dict = {}
    space_target_dict = {}
    char_dict = dict(zip(char_list, [x for x in range(len(char_list))]))
    with open(target_path, 'r', encoding='utf-8') as fid:
        for line in fid:
            split_line = line.strip().replace('\t', ' ').split(' ')
            if model_unit == 'char':
                text = ''.join(split_line[1:])
                token = [char_dict[x] if x in char_dict else map_oov for x in text]
                space_token = [1] * len(text)
            elif model_unit == 'word':
                token = []
                space_token = []
                for word_num in range(1, len(split_line)):
                    word = split_line[word_num]
                    token_tmp = [char_dict[x] if x in char_dict else map_oov for x in word]                                     
                    token.extend(token_tmp) 
                    space_token_tmp = [0] * (len(word) - 1) + [1]  
                    space_token.extend(space_token_tmp)  
            if len(token) > 0:
                for x in token:
                    labelcount[x] += 1
                transcript_num += 1
            target_dict[split_line[0]] = token 
            space_target_dict[split_line[0]] = space_token            
    labelcount[odim - 1] = transcript_num  # count <eos>
    labelcount[labelcount == 0] = 1  # flooring
    labelcount[blank] = 0  # remove counts for blank
    labeldist = labelcount.astype(np.float32) / np.sum(labelcount)
    return target_dict, space_target_dict, labeldist


def splice(utt, left_context_width, right_context_width):
    """
    splice the utterance
    Args:
        utt: numpy matrix containing the utterance features to be spliced
        context_width: how many frames to the left and right should
            be concatenated
    Returns:
        a numpy array containing the spliced features, if the features are
        too short to splice None will be returned
    """
    # return None if utterance is too short
    if utt.shape[0] < 1 + left_context_width + right_context_width:
        return None

    #  create spliced utterance holder
    utt_spliced = np.zeros(
        shape=[utt.shape[0], utt.shape[1]*(1 + left_context_width + right_context_width)],
        dtype=np.float32)

    #  middle part is just the utterance
    utt_spliced[:, left_context_width*utt.shape[1]:
                (left_context_width+1)*utt.shape[1]] = utt

    for i in range(left_context_width):
        #  add left context
        utt_spliced[i+1:utt_spliced.shape[0],
                    (left_context_width-i-1)*utt.shape[1]:
                    (left_context_width-i)*utt.shape[1]] = utt[0:utt.shape[0]-i-1, :]
    
    for i in range(right_context_width):
        # add right context
        utt_spliced[0:utt_spliced.shape[0]-i-1,
                    (left_context_width+i+1)*utt.shape[1]:
                    (left_context_width+i+2)*utt.shape[1]] = utt[i+1:utt.shape[0], :]

    return utt_spliced


def add_delta(utt, delta_order):
    num_frames = utt.shape[0]
    feat_dim = utt.shape[1]

    utt_delta = np.zeros(
        shape=[num_frames, feat_dim * (1 + delta_order)],
        dtype=np.float32)

    #  first order part is just the utterance max_offset+1
    utt_delta[:, 0:feat_dim] = utt

    scales = [[1.0], [-0.2, -0.1, 0.0, 0.1, 0.2],
              [0.04, 0.04, 0.01, -0.04, -0.1, -0.04, 0.01, 0.04, 0.04]]

    delta_tmp = np.zeros(shape=[num_frames, feat_dim], dtype=np.float32)
    for i in range(1, delta_order + 1):
        max_offset = (len(scales[i]) - 1) / 2
        for j in range(-max_offset, 0):
            delta_tmp[-j:, :] = utt[0:(num_frames + j), :]
            for k in range(-j):
                delta_tmp[k, :] = utt[0, :]
            scale = scales[i][j + max_offset]
            if scale != 0.0:
                utt_delta[:, i * feat_dim:(i + 1) * feat_dim] += scale * delta_tmp

        scale = scales[i][max_offset]
        if scale != 0.0:
            utt_delta[:, i * feat_dim:(i + 1) * feat_dim] += scale * utt

        for j in range(1, max_offset + 1):
            delta_tmp[0:(num_frames - j), :] = utt[j:, :]
            for k in range(j):
                delta_tmp[-(k + 1), :] = utt[(num_frames - 1), :]
            scale = scales[i][j + max_offset]
            if scale != 0.0:
                utt_delta[:, i * feat_dim:(i + 1) * feat_dim] += scale * delta_tmp

    return utt_delta

        
class AudioParser(object):
    def WaveData(self, audio_path):
        y = load_audio(audio_path)
        return y

    def RMRData(self, rmr_path, mat_key = 'rir'):
        y = load_mat(rmr_path, mat_key)
        return y

    def MakeMixture(self, speech, noise, db):
        if speech is None or noise is None:
            #print("MakeMixture: speech is None or noise is None")
            return None
        if np.sum(np.square(noise)) < 1.0e-6:
            #print("MakeMixture: np.sum(np.square(noise)) < 1.0e-6")
            return None

        spelen = speech.shape[0]

        exnoise = noise
        while exnoise.shape[0] < spelen:
            exnoise = np.concatenate([exnoise, noise], 0)
        noise = exnoise
        noilen = noise.shape[0]

        elen = noilen - spelen - 1
        if elen > 1:
            s = np.round( random.randint(0, elen - 1) )
        else:
            s = 0
        e = s + spelen

        noise = noise[s:e]

        try:
            noi_pow = np.sum(np.square(noise))
            if noi_pow > 0:
               noi_scale = math.sqrt(np.sum(np.square(speech)) / ( noi_pow * ( 10 ** (db / 10.0) ) ) )
            else:
                 return None
        except:
            return None

        nnoise  = noise * noi_scale
        mixture = speech + nnoise
        mixture = mixture.astype('float32')
        return mixture

    def Make_Reverberation(self, speech, rmr, use_fast = False):

        if speech is None or rmr is None:
            #print("Make_Reverberation: speech is None or rmr is None")           
            return None

        speech_nsam = speech.shape[0]
        if use_fast:
            speech = convfft(rmr, speech)
        else:
            speech = np.convolve(rmr, speech, mode='full')

        speech = speech[:speech_nsam]
        speech = speech.astype('float32')

        return speech

    def Gain_Control(self, wave, Gain):

        if wave is None:
            #print("Gain_Control: wave is None")
            return None

        max_sample = np.max(np.fabs(wave))        
        if max_sample <= 0:
           #print("Gain_Control: np.fabs(wave) is 0")
           return None
           
        wave = wave / max_sample
        wave = wave * Gain
        wave = wave.astype('float32')

        return wave

    def Make_Noisy_Wave(self, speech, noise, spe_rmr, noi_rmr, SNR):
        if speech is None or noise is None:
            #print("Make_Noisy_Wave:speech is None or noise is None")
            return None

        if spe_rmr is not None:
            speech = self.Make_Reverberation(speech, spe_rmr)
        if noi_rmr is not None:
            noise = self.Make_Reverberation(noise, noi_rmr)

        noisy = self.MakeMixture(speech, noise, SNR)
        if noisy is not None:
            noisy = noisy.astype('float32')
            return noisy
        else:
            return None


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def rolling_window(a, window, step=1):
    # http://ellisvalentiner.com/post/2017-03-21-np-strides-trick
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]


def framesig(sig, frame_len, frame_step, winfunc=lambda x: numpy.ones((x,)), stride_trick=True):
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :param stride_trick: use stride trick to compute the rolling window and window multiplication faster
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig, zeros))
    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
            numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = np.array(indices, dtype=np.int32)
        frames = padsignal[indices]
        win = np.tile(winfunc(frame_len), (numframes, 1))

    return frames * win
    
    
def make_fft_feat(sound):
    nfft = 512 
    winlen = 0.032
    winstep = 0.016
    samplerate = 16000             
    winfunc = scipy.signal.hamming
    frames = framesig(sound, winlen*samplerate, winstep*samplerate, winfunc)
    complex_spec = np.fft.rfft(frames, nfft)
    utt_mat = np.absolute(complex_spec)         
    return utt_mat
    
                                   
class FbankFeatLabelParser(AudioParser):
    def __init__(self, label_file, char_list):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        self.label_num = len(char_list)
        if label_file is not None:
            self.target_dict, self.space_target_dicts, self.labeldist = read_target_file(self.model_unit, label_file, char_list)
        else:
            self.target_dict = self.labeldist = None

        super(FbankFeatLabelParser, self).__init__()
    
    def get_labeldist(self):
        return self.labeldist
        
    def extract_label(self, utt_key):
        if utt_key is not None and utt_key in self.target_dict:
            targets = self.target_dict[utt_key]
            encoded_target = np.array(targets)
            if encoded_target.shape[0] > 0:
                return encoded_target
            else:
                return None
        else:
            print(utt_key, 'label error')
            return None
    
    def extract_space_label(self, utt_key):
        if utt_key is not None and utt_key in self.space_target_dicts:
            targets = self.space_target_dicts[utt_key]
            encoded_target = np.array(targets)
            if encoded_target.shape[0] > 0:
                return encoded_target
            else:
                return None
        else:
            print(utt_key, 'space_label error')
            return None
                    
    def extract_kaldi_feat(self, feat_path, feat_type='kaldi_magspec'): 
        try:    
            utt_mat = kaldi_io.read_mat(feat_path)
            ##utt_mat = utt_mat[:, :256]
            if feat_type == 'kaldi_magspec':
                spect = utt_mat
            elif feat_type == 'kaldi_powspec':
                spect = np.square(utt_mat)
            return spect  
        except:
            print(feat_path, 'extract_kaldi_feat error')
            return None     
                
    def extract_feat(self, sound, feat_type='fbank'):
        if sound is None:
           print('extract_feat sound error: sound is None')
           return None
        if np.max(np.fabs(sound)) <= 1e-6:
           print('extract_feat sound error: sound is Small')
           return None        
        
        if feat_type == 'fft':
            utt_mat = make_fft_feat(sound)
        else:
            sound = sound.astype('float32')
            if feat_type == 'fbank':
                utt_mat = extract_fbanks_module.make_fbank(sound)
            elif feat_type == 'agc_fbank':
                utt_mat = extract_fbanks_module.make_agc_fbank(sound)
            elif feat_type == 'ns_fbank':
                utt_mat = extract_fbanks_module.make_nsx_fbank(sound)
            elif feat_type == 'ns_agc_fbank':
                utt_mat = extract_fbanks_module.make_nsx_agc_fbank(sound)
            else:
                raise Exception('feat_type {} is not supported!'.format(feat_type))

        return utt_mat
    
    def transform_feat(self, utt_mat, cmvn=None, delta_order=0, left_context_width=0, right_context_width=0):
        if utt_mat is None:
           return None
           
        if delta_order > 0:
            utt_mat = add_delta(utt_mat, delta_order)

        if cmvn is not None:
            utt_mat = (utt_mat + cmvn[0, :]) * cmvn[1, :]

        if left_context_width > 0 or right_context_width > 0:
            utt_mat = splice(utt_mat, left_context_width, right_context_width)
            
        return utt_mat
        