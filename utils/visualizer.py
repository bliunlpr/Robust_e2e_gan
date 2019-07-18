import os
import time
import ntpath
import logging
from collections import OrderedDict
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from . import utils


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
class LogFormatter():
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)
        
class Visualizer():
    def __init__(self, opt):        
        self.opt = opt
        self.name = opt.name
        self.log_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.img_dir = os.path.join(self.log_dir, 'att_ws')
        utils.mkdirs([self.log_dir, self.img_dir])
        self.logger = self.create_output_dir(opt)  

        self.errors_meter = OrderedDict()
        self.plot_reports = {}
        self._grid = True  
        self._marker = 'x'
              
    def get_logger(self):
        return self.logger
                
    def create_output_dir(self, opt):
        filepath = os.path.join(self.log_dir, 'main.log')
    
        # Safety check
        if os.path.exists(filepath) and opt.resume == "":
            logging.warning("Experiment already exists!")
    
        # Create logger
        log_formatter = LogFormatter()
    
        # create file handler and set level to debug
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)
    
        # create console handler and set level to info
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_formatter)
    
        # create logger and set level to debug
        logger = logging.getLogger()
        logger.handlers = []
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
        # quite down visdom
        logging.getLogger("requests").setLevel(logging.CRITICAL)
        logging.getLogger("urllib3").setLevel(logging.CRITICAL)
    
        # reset logger elapsed time
        def reset_time():
            log_formatter.start_time = time.time()
        logger.reset_time = reset_time
    
        logger.info(opt)        
        return logger
    
    def reset(self):
        for k, meter in self.errors_meter.items():
            meter.reset()
                
    # errors: same format as |errors| of plotCurrentErrors
    def set_current_errors(self, errors):        
        for k, v in errors.items():
            if k not in self.errors_meter:
                self.errors_meter[k] = utils.AverageMeter()
            self.errors_meter[k].update(v) 
    
    # errors: same format as |errors| of plotCurrentErrors
    def get_current_errors(self, error):
        if error in self.errors_meter: 
            meter = self.errors_meter[error]   
            avg = meter.avg
            return avg   
        else:
            return None   
                        
    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, iters):
        message = '(epoch: %d, iters: %d ) ' % (epoch, iters)
        for k, meter in self.errors_meter.items():
            val = meter.val
            avg = meter.avg            
            if avg != 0:
                message += '%s: %.6f (%.6f) ' % (k, val, avg)
        self.logger.info(message)
    
    # errors: same format as |errors| of plotCurrentErrors
    def print_epoch_errors(self, epoch, iters):
        message = '(epoch: %d, iters: %d ) ' % (epoch, iters)
        for k, meter in self.errors_meter.items():
            avg = meter.avg            
            if avg != 0:
                message += '%s: %.3f ' % (k, avg)
        self.logger.info(message)
                
    # save image to the disk
    def add_plot_report(self, report_keys, file_name):
        plot_report_keys = OrderedDict()
        for k in report_keys:
            plot_report_keys[k] = []
        self.plot_reports[file_name] = plot_report_keys
        return plot_report_keys
    
    # save image to the disk
    def set_plot_report(self, plot_report, file_name):
        plot_report_keys = OrderedDict()
        for k, v in plot_report.items():
            plot_report_keys[k] = plot_report[k]
        self.plot_reports[file_name] = plot_report_keys
            
    # save image to the disk
    def plot_epoch_errors(self, epoch, iters, file_name):
        stats_cpu = {}
        for name, meter in self.errors_meter.items():            
            stats_cpu[name] = float(meter.avg)  # copy to CPU

        stats_cpu['epoch'] = epoch
        stats_cpu['iteration'] = iters
        
        if file_name in self.plot_reports:
            plot_report = self.plot_reports[file_name] 
            f = plt.figure()
            a = f.add_subplot(111)
            a.set_xlabel('iteration')
            if self._grid:
                a.grid()
            for k, v in plot_report.items():
                if k in stats_cpu:                     
                    plot_report[k].append((iters, stats_cpu[k]))
                    xy = plot_report[k]
                    if len(xy) == 0:
                        continue
                    xy = np.array(xy)
                    a.plot(xy[:, 0], xy[:, 1], marker=self._marker, label=k)
            if a.has_data():
                l = a.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                f.savefig(os.path.join(self.log_dir, file_name),
                          bbox_extra_artists=(l,), bbox_inches='tight')

            plt.close() 
            return plot_report

    def plot_attention(self, att_w, dec_len, enc_len, file_name):               
        if len(att_w.shape) == 3:
            att_w = att_w[:, :dec_len, :enc_len]
        else:
            att_w = att_w[:dec_len, :enc_len]
        try:
            self._plot_and_save_attention(att_w, file_name)
        except:
            print('plot_attention error')

    def _plot_and_save_attention(self, att_w, file_name):
        # dynamically import matplotlib due to not found error
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        if len(att_w.shape) == 3:
            for h, aw in enumerate(att_w, 1):
                plt.subplot(1, len(att_w), h)
                plt.imshow(aw, aspect="auto")
                plt.xlabel("Encoder Index")
                plt.ylabel("Decoder Index")
        else:
            plt.imshow(att_w, aspect="auto")
            plt.xlabel("Encoder Index")
            plt.ylabel("Decoder Index")
        plt.tight_layout()
        plt.savefig(os.path.join(self.img_dir, file_name))
        plt.close()  
        
    def plot_specgram(self, clean_input, mix_input, enhanced_out, input_size, file_name): 
        try:     
            clean_input = clean_input[:input_size, :].T
            mix_input = mix_input[:input_size, :].T
            enhanced_out = enhanced_out[:input_size, :].T
            import matplotlib.pyplot as plt
            plt.switch_backend('agg')
            plt.subplot(311)
            plt.imshow(clean_input, aspect="auto")
            plt.title('clean specgram')
            ax = plt.gca()
            ax.invert_yaxis()
            plt.subplot(312)
            plt.imshow(mix_input, aspect="auto")
            ax = plt.gca()
            ax.invert_yaxis()
            plt.title('mix specgram')
            plt.subplot(313)
            plt.imshow(enhanced_out, aspect="auto")
            plt.title('enhance specgram')
            ax = plt.gca()
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(self.img_dir, file_name))
            plt.close()    
        except:
            print('plot_specgram error')               