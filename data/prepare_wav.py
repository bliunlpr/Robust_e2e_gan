import os
import sys
from data import kaldi_io
import librosa
import numpy as np
import scipy.signal
import math
from multiprocessing import Pool
import scipy.io.wavfile as wav


def load_audio(path):
    rate, sig = wav.read(path)
    return sig


def make_wav(path_list, enhanced_wav_dir):
    for path in path_list:
        uttid, feat_path, angle_path = path
        feat_mat = kaldi_io.read_mat(feat_path)
        angle_mat = kaldi_io.read_mat(angle_path)
        image = np.sin(angle_mat) * feat_mat
        real = np.cos(angle_mat) * feat_mat
        result = 1j * image
        result += real 
        D = librosa.istft(result, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)
        wav_file = os.path.join(enhanced_wav_dir, uttid + '.wav')
        wav.write(wav_file, 16000, D)

        
def main():
    
    noisy_angle_scp = sys.argv[1]
    enhanced_feats_scp = sys.argv[2]
    enhanced_wav_dir = sys.argv[3]
    
    if not os.path.exists(enhanced_wav_dir):
        os.makedirs(enhanced_wav_dir)
        
    noisy_angle_dict = {}
    with open(noisy_angle_scp, 'r', encoding='utf-8') as fid:
        for line in fid:
            line = line.strip().replace('\n','')
            uttid, angle_path = line.split()
            noisy_angle_dict[uttid] = angle_path
    print('noisy_angle_dict', len(noisy_angle_dict)) 
    
    enhanced_list = []
    with open(enhanced_feats_scp, 'r', encoding='utf-8') as fid:
        for line in fid:
            line = line.strip().replace('\n','')
            uttid, feat_path = line.split()
            if uttid in noisy_angle_dict:
                angle_path = noisy_angle_dict[uttid]
                enhanced_list.append((uttid, feat_path, angle_path))
            else:
                print('{} not in noisy_angle_dict'.format(uttid)) 
    print('enhanced_list', len(enhanced_list)) 
    
    threads_num = 8 
    wav_num = len(enhanced_list)
    print('wav_num ', wav_num)
    print('Parent process %s.' % os.getpid())
    p = Pool()    
    start = 0
    for i in range(threads_num):
        end = start + int(wav_num / threads_num)
        if i == (threads_num - 1):
            end = wav_num
        wav_path_tmp_list = enhanced_list[start:end]
        start = end
        p.apply_async(make_wav, args=(wav_path_tmp_list, enhanced_wav_dir))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')          
      
if __name__ == '__main__':
    main()
