from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        
        # task related
        self.parser.add_argument('--recog-dir', type=str, help='Filename of recognition feature data (Kaldi scp)')
        self.parser.add_argument('--enhance-dir', type=str, help='Filename of enhance feature data (Kaldi scp)')
        self.parser.add_argument('--enhance-out-dir', type=str, help='Filename of enhance feature data (Kaldi scp)')
        self.parser.add_argument('--recog-label', type=str, help='Filename of recognition label data (json)')
        self.parser.add_argument('--recog-json', type=str, help='Filename of recognition data (json)')
        self.parser.add_argument('--result-label', type=str, help='Filename of result label data (json)')
        self.parser.add_argument('--nj', type=int, default=1, help='nj')
       
        # search related
        self.parser.add_argument('--nbest', type=int, default=1, help='Output N-best hypotheses')
        self.parser.add_argument('--beam-size', type=int, default=1, help='Beam size')
        self.parser.add_argument('--penalty', default=0.0, type=float, help='Incertion penalty')
        self.parser.add_argument('--maxlenratio', default=0.0, type=float,
                            help="""Input length ratio to obtain max output length.
                            If maxlenratio=0.0 (default), it uses a end-detect function
                            to automatically find maximum hypothesis lengths""")
        self.parser.add_argument('--minlenratio', default=0.0, type=float, help='Input length ratio to obtain min output length')
        self.parser.add_argument('--ctc-weight', default=0.0, type=float, help='CTC weight in joint decoding')
        
        self.parser.add_argument('--fstlm-path', type=str, help='fstlm_path')
        self.parser.add_argument('--nn-char-map-file', type=str, help='nn-char-map_file')       
        
        