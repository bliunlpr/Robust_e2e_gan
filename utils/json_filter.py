# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import json
import argparse

if __name__ == '__main__':
    in_json_file = sys.argv[1]
    out_json_file = sys.argv[2]
    
    new_json = {}
    with open(in_json_file, 'r') as f:
        j = json.load(f)

    for x in j['utts']:
        rec_text = j['utts'][x]['output'][0]['rec_text'].replace('<eos>', '') 
        text = j['utts'][x]['output'][0]['text']
        if rec_text != text:   
           new_json[x] = j['utts'][x]  
    with open(out_json_file, 'w') as f:
        f.write(json.dumps({'utts': new_json}, indent=4, sort_keys=True, ensure_ascii=False))
         
