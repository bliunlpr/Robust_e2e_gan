from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # optimization related
        self.parser.add_argument('--opt_type', default='adadelta', type=str, choices=['adadelta', 'adam'], help='Optimizer')
        self.parser.add_argument('--lr', type=float, default=0.005, help='learning rate, default=0.0002')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
        self.parser.add_argument('--eps', default=1e-8, type=float, help='Epsilon constant for optimizer')
        self.parser.add_argument('--eps-decay', default=0.01, type=float, help='Decaying ratio of epsilon')
        self.parser.add_argument('--criterion', default='acc', type=str, choices=['loss', 'acc'], help='Criterion to perform epsilon decay')
        self.parser.add_argument('--threshold', default=1e-4, type=float, help='Threshold to stop iteration')
        self.parser.add_argument('--start_epoch', default=0, type=int, help='manual iters number (useful on restarts)') 
        self.parser.add_argument('--iters', default=0, type=int, help='manual iters number (useful on restarts)')   
        self.parser.add_argument('--epochs', '-e', default=30, type=int, help='Number of maximum epochs')
        self.parser.add_argument('--shuffle_epoch', default=-1, type=int, help='Number of shuffle epochs')        
        self.parser.add_argument('--grad-clip', default=5, type=float, help='Gradient norm threshold to clip')
        self.parser.add_argument('--num-save-attention', default=3, type=int, help='Number of samples of attention to be saved')   
        self.parser.add_argument('--num-saved-specgram', default=3, type=int, help='Number of samples of specgram to be saved')               
         
        # debug related   
        self.parser.add_argument('--validate_freq', type=int, default=8000, help='how many batches to validate the trained model')   
        self.parser.add_argument('--print_freq', type=int, default=500, help='how many batches to print the trained model')         
        self.parser.add_argument('--best_acc', default=0, type=float, help='best_acc')
        self.parser.add_argument('--best_loss', default=float('inf'), type=float, help='best_loss')
