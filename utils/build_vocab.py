"""Build vocabulary from manifest files.

Each item in vocabulary file is a character.
"""
import sys
import argparse
import functools
import codecs
import json
from collections import Counter
import os.path

def count_manifest(counter, manifest_path):
    with open(manifest_path) as f:
        for line in f:
            line_splits = line.strip().split()
            utt_id = line_splits[0]
            transcript = ''.join(line_splits[1:])
            for char in transcript:
                counter.update(char)


def main():
    text = sys.argv[1]
    count_threshold = int(sys.argv[2])
    vocab_path = sys.argv[3]
    
    counter = Counter()
    count_manifest(counter, text)
    
    count_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    print (len(count_sorted))
    num = 1
    with open(vocab_path, 'w') as fout:
        fout.write('<unk> 1' + '\n')
        for char, count in count_sorted:
            if count < count_threshold: break
            num += 1
            fout.write(char + ' ' + str(num) + '\n')
    print (num)

if __name__ == '__main__':
    main()

