#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


# general configuration
backend=chainer
stage=0        # start from 0 if you need to start from data preparation
gpu=            # will be deprecated, please use ngpu
ngpu=0          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=${resume:=none}    # Resume the training from snapshot

# feature configuration
do_delta=false # true when using CNN

# network archtecture
# encoder related
etype=vggblstmp     # encoder architecture type
elayers=8
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
subsample_type="skip"
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
aact_func=softmax
aconv_chans=10
aconv_filts=100
lsm_type="none"
lsm_weight=0.0
dropout_rate=0.0
# hybrid CTC/attention
mtlalpha=0.5

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=30

# rnnlm related
model_unit=char
batchsize_lm=64
dropout_lm=0.5
input_unit_lm=256
hidden_unit_lm=650
lm_weight=0.2
fusion=${fusion:=none}
feat_type=kaldi_magspec

# decoding parameter
lmtype=rnnlm
beam_size=12
nbest=12
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'


# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;
. ./cmd.sh
. ./path.sh
# check gpu option usage
if [ ! -z $gpu ]; then
    echo "WARNING: --gpu option will be deprecated."
    echo "WARNING: please use --ngpu option."
    if [ $gpu -eq -1 ]; then
        ngpu=0
    else
        ngpu=1
    fi
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',     --resume $resume \
set -e
set -u
set -o pipefail

enhanceroot="/home/bliu/SRC/workspace/e2e/data/snie_enhance/"
tag=""
dictroot="/home/bliu/SRC/workspace/e2e/data/mix_aishell/lang_1char/"
train_set=train
train_dev=dev
recog_set="mix clean enhance enhance1"
dict=${dictroot}/${train_set}_units.txt
nlsyms=${dictroot}/non_lang_syms.txt

expdir=checkpoints/asr_mix_fbank80_vggblstmp4_drop0.2/
name=asr_mix_fbank80_vggblstmp4_drop0.2
lmexpdir=checkpoints/train_rnnlm_2layer_256_650_drop0.2_bs64
if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=4
    for rtask in ${recog_set}; do
    ##(
        decode_dir=decode_${tag}_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_${lmtype}${lm_weight}
        feat_recog_dir=${enhanceroot}/${rtask}/
        utils/fix_data_dir.sh $feat_recog_dir
        # split data
        ##splitjson.py --parts ${nj} ${feat_recog_dir}/data.json 
        sdata=${feat_recog_dir}/split$nj

        mkdir -p ${expdir}/${decode_dir}/log/

        [[ -d $sdata && ${feat_recog_dir}/feats.scp -ot $sdata ]] || utils/split_data.sh ${feat_recog_dir} $nj || exit 1;
        echo $nj > ${expdir}/num_jobs

        #### use CPU for decoding  ##& ##${decode_cmd} JOB=1 ${expdir}/${decode_dir}/log/decode.JOB.log \

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            python3 asr_recog.py \
            --dataroot ${enhanceroot} \
            --name $name \
            --nj $nj \
            --gpu_ids 0 \
            --dict_dir ${dictroot} \
            --feat_type ${feat_type} \
            --nbest $nbest \
            --resume ${expdir}/model.acc.best \
            --recog-dir ${sdata}/JOB/ \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --lmtype ${lmtype} \
            --verbose ${verbose} \
            --normalize_type 0 \
            --rnnlm ${lmexpdir}/rnnlm.model.best \
            --kenlm ${dictroot}/text.arpa \
            --lm-weight ${lm_weight} 
          
        score_sclite.sh --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
        
        kenlm_path="/home/bliu/mywork/workspace/e2e/src/kenlm/build/text_character.arpa"
        rescore_sclite.sh --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${expdir}/${decode_dir}_rescore ${dict} ${kenlm_path}
    ##) &
    done
    ##wait
    echo "Finished"
fi