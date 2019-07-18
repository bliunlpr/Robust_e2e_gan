# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import json
import kenlm

if __name__ == '__main__':
    in_json_file = sys.argv[1]
    lm_path = sys.argv[2]
    rescore_weight = float(sys.argv[3])
    out_json_file = sys.argv[4]
    print(sys.argv)
    
    model = kenlm.LanguageModel(lm_path)
    print('{0}-gram model'.format(model.order))

    new_json = {}
    with open(in_json_file, 'r', encoding='utf-8') as f:
        j = json.load(f)

    for x in j['utts']:
        try:
            rec_text = j['utts'][x]['output'][0]['rec_text'].replace('<eos>', '') 
            text = j['utts'][x]['output'][0]['text']
            ##if rec_text != text:   
            new_json[x] = j['utts'][x]  
            new_scores = []    
            for num in range(12):
                rec_token = j['utts'][x]['rec_token[{0:05d}]'.format(num)].replace('<eos>', '') 
                score = float(j['utts'][x]['score[{0:05d}]'.format(num)])   
                kenlm_score = float(model.score(rec_token) / len(rec_token))
                new_score = rescore_weight * kenlm_score + score
                new_json[x]['kenlm_score' + '[' + '{:05d}'.format(num) + ']'] = kenlm_score
                new_scores.append(new_score) 
                
            max_score = max(new_scores)
            max_score_idx = new_scores.index(max(new_scores))
            j['utts'][x]['output'][0]['rec_text'] = j['utts'][x]['rec_text[{0:05d}]'.format(max_score_idx)]
            j['utts'][x]['output'][0]['rec_token'] = j['utts'][x]['rec_token[{0:05d}]'.format(max_score_idx)]
            j['utts'][x]['output'][0]['rec_tokenid'] = j['utts'][x]['rec_tokenid[{0:05d}]'.format(max_score_idx)]
        except:
            new_json[x] = j['utts'][x] 
        
    with open(out_json_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps({'utts': new_json}, indent=4, sort_keys=True, ensure_ascii=False))
         