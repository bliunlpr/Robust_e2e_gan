#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


# general configuration
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

dataroot="/home/bliu/SRC/workspace/e2e/data/clean_aishell/"
##dictroot="/home/bliu/mywork/workspace/e2e/data/lang_1char/"
dictroot="/home/bliu/SRC/workspace/e2e/data/lang_syllable/"
train_set=train
train_dev=dev
##recog_set="test_mix test_clean"
##recog_set="test_clean_small"
recog_set="test"

# you can skip this and remove --rnnlm option in the recognition (stage 5)
dict=${dictroot}/${train_set}_units.txt
embed_init_file=${dictroot}/char_embed_vec
echo "dictionary: ${dict}"
nlsyms=${dictroot}/non_lang_syms.txt
lmexpdir=checkpoints/train_${lmtype}_2layer_${input_unit_lm}_${hidden_unit_lm}_drop${dropout_lm}_bs${batchsize_lm}
mkdir -p ${lmexpdir}
if [ ${stage} -le 3 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=${lmexpdir}/local/lm_train
    mkdir -p ${lmdatadir}
    text2token.py -s 1 -n 1 -l ${nlsyms} ${dataroot}/train/text_syllable | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/train_trans.txt
    ##text2token.py -s 1 -n 1 -l ${nlsyms} /home/bliu/mywork/workspace/e2e/data/lang_1char/all_text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
    ##    > ${lmdatadir}/train_trans.txt
    cat ${lmdatadir}/train_trans.txt | tr '\n' ' ' > ${lmdatadir}/train.txt
    text2token.py -s 1 -n 1 -l ${nlsyms} ${dataroot}/${train_dev}/text_syllable | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/valid.txt
    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
    fi
    ${cuda_cmd} ${lmexpdir}/train.log \
        python3 lm_train.py \
        --ngpu 1 \
        --input-unit ${input_unit_lm} \
        --lm-type ${lmtype} \
        --unit ${hidden_unit_lm} \
        --dropout-rate ${dropout_lm} \
        --embed-init-file ${embed_init_file} \
        --verbose 1 \
        --batchsize ${batchsize_lm} \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --dict ${dict}
fi


name=aishell_${model_unit}_${etype}_e${elayers}_subsample${subsample}_${subsample_type}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_${aact_func}_aconvc${aconv_chans}_aconvf${aconv_filts}_lsm_type${lsm_type}_lsm_weight${lsm_weight}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}_dropout${dropout_rate}_fusion${fusion}

##name=aishell_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}  --resume $resume \

lmexpdir=checkpoints/train_fsrnnlm_2layer_256_650_drop0.5_bs64
    
if [ ${stage} -le 4 ]; then
    echo "stage 4: Network Training"
    python3 asr_train.py \
    --dataroot $dataroot \
    --name $name \
    --model-unit $model_unit \
    --resume $resume \
    --dropout-rate ${dropout_rate} \
    --etype ${etype} \
    --elayers ${elayers} \
    --eunits ${eunits} \
    --eprojs ${eprojs} \
    --subsample ${subsample} \
    --subsample-type ${subsample_type} \
    --dlayers ${dlayers} \
    --dunits ${dunits} \
    --atype ${atype} \
    --aact-fuc ${aact_func} \
    --aconv-chans ${aconv_chans} \
    --aconv-filts ${aconv_filts} \
    --mtlalpha ${mtlalpha} \
    --batch-size ${batchsize} \
    --maxlen-in ${maxlen_in} \
    --maxlen-out ${maxlen_out} \
    --opt_type ${opt} \
    --verbose ${verbose} \
    --lmtype ${lmtype} \
    --rnnlm ${lmexpdir}/rnnlm.model.best \
    --fusion ${fusion} \
    --epochs ${epochs}
fi

expdir=checkpoints/asr_clean_syllable_fbank80_drop0.2/
name=asr_clean_syllable_fbank80_drop0.2
lmexpdir=checkpoints/train_rnnlm_2layer_256_650_drop0.2_bs64
fst_path="/home/bliu/mywork/workspace/e2e/data/lang_word/LG_pushed_withsyms.fst"
nn_char_map_file="/home/bliu/mywork/workspace/e2e/data/lang_word/net_chars.txt"
if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=4
    for rtask in ${recog_set}; do
    ##(
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_${lmtype}${lm_weight}
        feat_recog_dir=${dataroot}/${rtask}/
        utils/fix_data_dir.sh $feat_recog_dir
        # split data
        ##splitjson.py --parts ${nj} ${feat_recog_dir}/data.json  --kenlm ${dictroot}/text.arpa \
        sdata=${feat_recog_dir}/split$nj

        mkdir -p ${expdir}/${decode_dir}/log/

        [[ -d $sdata && ${feat_recog_dir}/feats.scp -ot $sdata ]] || utils/split_data.sh ${feat_recog_dir} $nj || exit 1;
        echo $nj > ${expdir}/num_jobs

        #### use CPU for decoding  ##& ##${decode_cmd} JOB=1 ${expdir}/${decode_dir}/log/decode.JOB.log \

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            python3 asr_recog.py \
            --dataroot ${dataroot} \
            --name $name \
            --model-unit $model_unit \
            --nj $nj \
            --gpu_ids 0 \
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
            --fstlm-path ${fst_path} \
            --nn-char-map-file ${nn_char_map_file} \
            --lm-weight ${lm_weight} 
          
        score_sclite.sh --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
        
        ##kenlm_path="/home/bliu/mywork/workspace/e2e/src/kenlm/build/text_character.arpa"
        ##rescore_sclite.sh --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${expdir}/${decode_dir}_rescore ${dict} ${kenlm_path}
    ##) &
    done
    ##wait
    echo "Finished"
fi