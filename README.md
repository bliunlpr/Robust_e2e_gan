# Boosting Noise Robustness of Acoustical Model via Deep Adversarial Training

This is an implementation of our ICASSP2018 Best Student Award paper "Boosting Noise Robustness of Acoustical Model via Deep Adversarial Training" 
on Python 3, PyTorch. We propose an adversarial training method to directly boost noise robustness of acoustic model. The joint 
optimization of generator, discriminator and AM concentrates the strengths of both GAN and AM for speech recognition. 

# Requirements
kaldi, Python 3.5, PyTorch 0.4.0.

# Data
### Chime 4
You can download [Chime 4](http://spandh.dcs.shef.ac.uk/chime_challenge/chime2016/) to run the code.

### Your Own Dataset
You need build train, dev and test directory. Each has ```feats.scp``` ```utt2spk``` ```spk2utt``` and ```text```. 

# Model

The model consists of a generator(G), a discriminator (D) and a classifier(C). The generator G performs the speech enhancement. It transforms the noisy speech signals into the enhanced version. The discriminator D aims to distinguish between the enhanced signals and clean ones. The classifier C classifies senones by features derivated from G. 

<div align="center">
<img src="https://github.com/bliunlpr/Robust_e2e_gan/blob/master/fig/framework.Jpeg"  height="400" width="495">
</div>
