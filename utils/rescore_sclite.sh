#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nlsyms=""
wer=false
bpe=""
bpemodel=""
remove_blank=true
filter=""
rescore_weight=15.

. utils/parse_options.sh

if [ $# != 4 ]; then
    echo "Usage: $0 <data-dir> <dict>";
    exit 1;
fi

score_dir=$1
dir=$2
dic=$3
lm_path=$4

mkdir -p ${dir}
cp ${score_dir}/data.json ${dir}/
##concatjson.py ${dir}/data.*.json > ${dir}/data.json
python3 utils/kenlm_rescore.py ${dir}/data.json $lm_path $rescore_weight ${dir}/data_rescore.json
json2trn.py ${dir}/data_rescore.json ${dic} ${dir}/ref.trn ${dir}/hyp.trn

if $remove_blank; then
    sed -i.bak2 -r 's/<blank> //g' ${dir}/hyp.trn
fi
if [ ! -z ${nlsyms} ]; then
    cp ${dir}/ref.trn ${dir}/ref.trn.org
    cp ${dir}/hyp.trn ${dir}/hyp.trn.org
    filt.py -v $nlsyms ${dir}/ref.trn.org > ${dir}/ref.trn
    filt.py -v $nlsyms ${dir}/hyp.trn.org > ${dir}/hyp.trn
fi
if [ ! -z ${filter} ]; then
    sed -i.bak3 -f ${filter} ${dir}/hyp.trn
    sed -i.bak3 -f ${filter} ${dir}/ref.trn
fi
    
sclite -r ${dir}/ref.trn trn -h ${dir}/hyp.trn trn -i rm -o all stdout > ${dir}/result.txt

echo "write a CER (or TER) result in ${dir}/result.txt"
grep -e Avg -e SPKR -m 2 ${dir}/result.txt

if ${wer}; then
    if [ ! -z $bpe ]; then
	spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref.trn | sed -e "s/â–? /g" > ${dir}/ref.wrd.trn
	spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/â–? /g" > ${dir}/hyp.wrd.trn
    else
	sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/ref.trn > ${dir}/ref.wrd.trn
	sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
    fi
    sclite -r ${dir}/ref.wrd.trn trn -h ${dir}/hyp.wrd.trn trn -i rm -o all stdout > ${dir}/result.wrd.txt
	
    echo "write a WER result in ${dir}/result.wrd.txt"
    grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.txt
fi
