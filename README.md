# Jointly Adversarial Enhancement Training for Robust End-to-End Speech Recognition

This is an implementation of our InterSpeech2019 paper "Jointly Adversarial Enhancement Training for Robust End-to-End Speech Recognition" on Python 3, PyTorch. We propose a jointly adversarial enhancement training to boost robustness of end-to-end systems. Specifically, we use a jointly compositional scheme of maskbased enhancement network, attention-based encoder-decoder network and discriminant network during training. 

# Requirements
Python 3.5, PyTorch 0.4.0.

# Data
### AISHELL
You can download [AISHELL](http://www.aishelltech.com/kysjcp) to run the code.

### Your Own Dataset
You need build train, dev and test directory. Each has ```clean_feats.scp``` ```noisy_feats.scp``` and ```text```. You can run ```python3 data/prepare_feats.py data_dir feat_dir noise_repeat_num``` to generate the noisy data ```noisy_feats.scp```.

# Model

The system consists of a mask-based enhancement network, an attention-based encoderdecoder network, a fbank feature extraction network and a discriminant network. The enhancement network transforms the noisy STFT features to the enhanced STFT features. The fbank feature extraction network is used to extract the normalized log fbank features. The end-to-end ASR model estimates the posteriori probabilities for output labels. The discriminant network is used to distinguish between the enhanced features and clean ones.

<div align="center">
<img src="https://github.com/bliunlpr/Robust_e2e_gan/blob/master/fig/framework.Jpeg"  height="400" width="495">
</div>

# Training

### E2E ASR training
You can train the E2E ASR network using the clean speech data and multi-condition training strategy i.e., optimization with both the clean and noisy speech.

```
python3 asr_train.py --dataroot Your data directory(including train, dev and test dataset) 

```

### Enhancement Training
You can train the enhancement network by the mask loss function.

```
python3 enhance_base_train.py --dataroot Your data directory

```
or the mask fbank loss function.

```
python3 enhance_fbank_train.py --dataroot Your data directory

```
or the gan loss function.

```
python3 enhance_gan_train.py --dataroot Your data directory

```
# Decoding
We use the Kaldi WFST decoder for decoding in all the experiments.
```
sh kaldi/decode.sh  

```
