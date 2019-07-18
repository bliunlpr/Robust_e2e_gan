import argparse
import os
from utils import utils
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # general configuration
        self.parser.add_argument('--works_dir', help='path to work', default='.')
        self.parser.add_argument('--dataroot', help='path (should have subfolders train, dev, test)')
        self.parser.add_argument('--dict_dir', default='/home/bliu/SRC/workspace/e2e/data/mix_aishell/lang_1char/', help='path to dict')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='vad', help='name of the experiment.')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')  
        self.parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')   
        self.parser.add_argument('--enhance_resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
        self.parser.add_argument('--asr_resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')  
        self.parser.add_argument('--joint_resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')            
        self.parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in data-loading')
        
        # input features
        self.parser.add_argument('--feat_type', type=str, default='kaldi_magspec', help='feat_type')   
        self.parser.add_argument('--left_context_width', type=int, default=0, help='input left_context_width-width')
        self.parser.add_argument('--right_context_width', type=int, default=0, help='input right_context_width')
        self.parser.add_argument('--delta_order', type=int, default=0, help='input delta-order')
        self.parser.add_argument('--normalize_type', type=int, default=1, help='normalize_type')        
        self.parser.add_argument('--num_utt_cmvn', type=int, help='the number of utterances for cmvn', default=20000)
        self.parser.add_argument('--num_utt_per_loading', type=int, help='the number of utterances one loading', default=200)
        self.parser.add_argument('--mix_noise', dest='mix_noise', action='store_true', help='mix_noise')        
        self.parser.add_argument('--lowSNR', type=float, default=5, help='lowSNR')
        self.parser.add_argument('--upSNR', type=float, default=30, help='upSNR') 
                        
        # encoder
        self.parser.add_argument('--etype', default='vggblstmp', type=str, choices=['blstm','blstmp','cnnblstmp','cnnblstm', 'vggblstmp','vggblstm'], 
                                            help='Type of encoder architecture')
        self.parser.add_argument('--elayers', default=4, type=int, help='Number of encoder layers')
        self.parser.add_argument('--eunits', '-u', default=320, type=int, help='Number of encoder hidden units')
        self.parser.add_argument('--eprojs', default=320, type=int, help='Number of encoder projection units')
        self.parser.add_argument('--subsample', default='1_1_1_1_1', type=str, help='Subsample input frames x_y means subsample every x frame at 1st layer y at 2nd.')
        self.parser.add_argument('--subsample-type', default='skip', type=str, choices=['skip','maxpooling'], help='subsample-type')
        
        # attention
        self.parser.add_argument('--atype', default='location', type=str, choices=['noatt', 'dot', 'add', 'location', 'coverage', 'coverage_location', 'location2d', 
                                 'location_recurrent', 'multi_head_dot', 'multi_head_add', 'multi_head_loc', 'multi_head_multi_res_loc'], 
                                 help='Type of attention architecture')
        self.parser.add_argument('--adim', default=320, type=int, help='Number of attention transformation dimensions')
        self.parser.add_argument('--aact-fuc', default='softmax', type=str, choices=['softmax','sigmoid','sigmoid_softmax'], help='out type')
        self.parser.add_argument('--awin', default=5, type=int, help='Window size for location2d attention')
        self.parser.add_argument('--aheads', default=4, type=int, help='Number of heads for multi head attention')
        self.parser.add_argument('--aconv-chans', default=10, type=int, help='Number of attention convolution channels(negative indicates no location-aware attention)')
        self.parser.add_argument('--aconv-filts', default=100, type=int, help='Number of attention convolution filters(negative indicates no location-aware attention)')
        
        # decoder
        self.parser.add_argument('--dtype', default='lstm', type=str, choices=['lstm'], help='Type of decoder network architecture')
        self.parser.add_argument('--dlayers', default=1, type=int, help='Number of decoder layers')
        self.parser.add_argument('--dunits', default=300, type=int, help='Number of decoder hidden units')
        self.parser.add_argument('--mtlalpha', default=0.5, type=float, help='Multitask learning coefficient, alpha: alpha*ctc_loss + (1-alpha)*att_loss ')
        self.parser.add_argument('--lsm-type', default='', type=str, choices=['', 'unigram'], help='Apply label smoothing with a specified distribution type')
        self.parser.add_argument('--lsm-weight', default=0.0, type=float, help='Label smoothing weight')
        self.parser.add_argument('--fusion', default='', type=str, help='Type of decoder fusion architecture')
        
        # enhance model
        self.parser.add_argument('--enhance_type', default='blstm', type=str, 
                                  choices=['blstm','unet_128', 'unet_256', 'blstmp','cnnblstmp','cnnblstm', 'vggblstmp','vggblstm'], 
                                  help='Type of enhance model architecture')
        self.parser.add_argument('--enhance_layers', default=3, type=int, help='Number of enhance model layers')
        self.parser.add_argument('--enhance_units', default=128, type=int, help='Number of enhance model hidden units')
        self.parser.add_argument('--enhance_projs', default=128, type=int, help='Number of enhance model projection units')
        self.parser.add_argument('--enhance_nonlinear_type', default='sigmoid', type=str, choices=['sigmoid','relu', 'relu6', 'softplus'], help='enhance_nonlinear_type')
        self.parser.add_argument('--enhance_loss_type', default='L2', type=str, choices=['L2','L1', 'smooth_L1'], help='enhance_loss_type')
        self.parser.add_argument('--enhance_opt_type', default='gan_fbank', type=str, choices=['gan_fft','gan_fbank'], help='enhance_opt_type')
        self.parser.add_argument('--enhance_dropout_rate', default=0.0, type=float, help='enhance_dropout_rate')
        self.parser.add_argument('--enhance_input_nc', default=1, type=int, help='enhance_input_nc')
        self.parser.add_argument('--enhance_output_nc', default=1, type=int, help='enhance_output_nc')
        self.parser.add_argument('--enhance_ngf', default=64, type=int, help='enhance_ngf')
        self.parser.add_argument('--enhance_norm', default='batch', type=str, help='enhance_norm')
        self.parser.add_argument('--L1_loss_lambda', default=1.0, type=float, help='L1_loss_lambda')
        
        # gan model
        self.parser.add_argument('--gan_loss_lambda', default=1.0, type=float, help='gan_loss_lambda')
        self.parser.add_argument('--netD_type', type=str, default='basic', help='selects model to use for netD [basic | n_layers | pixel]')
        self.parser.add_argument('--input_nc', type=int, default=1, help='# of input channels: 1 for grayscale') 
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--norm_D', type=str, default='batch', help='instance normalization or batch normalization [batch | norm | none]')        
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        
        # fbank model
        self.parser.add_argument('--fbank_dim', type=int, default=40, help='num_features of a frame')
        self.parser.add_argument('--fbank-opti-type', type=str, default='frozen', choices=['frozen', 'train'], help='fbank-opti-type')
        
        # model (parameter) related
        self.parser.add_argument('--dropout-rate', default=0.0, type=float, help='Dropout rate')
        self.parser.add_argument('--sche-samp-rate', default=0.0, type=float, help='scheduled sampling rate')
        self.parser.add_argument('--sche-samp-final-rate', default=0.6, type=float, help='scheduled sampling final rate')
        self.parser.add_argument('--sche-samp-start-epoch', default=5, type=int, help='scheduled sampling start epoch')
        self.parser.add_argument('--sche-samp-final_epoch', default=15, type=int, help='scheduled sampling start epoch')
        
         # rnnlm related         
        self.parser.add_argument('--model-unit', type=str, default='char', choices=['char', 'word', 'syllable'], help='model_unit')
        self.parser.add_argument('--space-loss-weight', default=0.1, type=float, help='space_loss_weight.')
        self.parser.add_argument('--lmtype', type=str, default=None, help='RNNLM model file to read')
        self.parser.add_argument('--rnnlm', type=str, default=None, help='RNNLM model file to read')
        self.parser.add_argument('--kenlm', type=str, default=None, help='KENLM model file to read')
        self.parser.add_argument('--word-rnnlm', type=str, default=None, help='Word RNNLM model file to read')
        self.parser.add_argument('--word-dict', type=str, default=None, help='Word list to read')
        self.parser.add_argument('--lm-weight', default=0.1, type=float, help='RNNLM weight.')
        
        # FSLSTMLM training configuration
        self.parser.add_argument('--fast_cell_size', type=int, default=400, help='fast_cell_size')
        self.parser.add_argument('--slow_cell_size', type=int, default=400, help='slow_cell_size')
        self.parser.add_argument('--fast_layers', type=int, default=2, help='fast_layers')
        self.parser.add_argument('--zoneout_keep_c', type=float, default=0.5, help='zoneout_c')
        self.parser.add_argument('--zoneout_keep_h', type=float, default=0.9, help='zoneout_h')
    
        # minibatch related
        self.parser.add_argument('--batch-size', '-b', default=30, type=int, help='Batch size')
        self.parser.add_argument('--maxlen-in', default=800, type=int, metavar='ML', help='Batch size is reduced if the input sequence length > ML')
        self.parser.add_argument('--maxlen-out', default=150, type=int, metavar='ML', help='Batch size is reduced if the output sequence length > ML')        
        self.parser.add_argument('--verbose', default=1, type=int, help='Verbose option')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        if self.opt.mtlalpha == 1.0:
            self.opt.mtl_mode = 'ctc'
        elif self.opt.mtlalpha == 0.0:
            self.opt.mtl_mode = 'att'
        else:
            self.opt.mtl_mode = 'mtl'

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        exp_path = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        utils.mkdirs(exp_path)
        self.opt.exp_path = exp_path
        file_name = os.path.join(exp_path, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt

