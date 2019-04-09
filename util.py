#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
General utils
'''

import pickle
import subprocess

def fread(filename, mode='r'):
    f = open(filename)
    return open(filename, mode).read().strip()

def write_pickle(obj, filename, *args):
    f = open(filename, 'wb')
    pickle.dump(obj, f, *args)

def read_pickle(filename):
    f = open(filename, 'rb')
    return pickle.load(f)

def write_file(log, filename, mode = 'a+'):
    f = open(filename, mode)
    f.write(log + '\n')
    f.close()

def get_bleu_moses(hypotheses, reference):
    """Get BLEU score with moses bleu score."""
    with open('tmp_hypotheses.txt', 'w') as f:
        for hypothesis in hypotheses:
            f.write(' '.join(hypothesis) + '\n')

    with open('tmp_reference.txt', 'w') as f:
        for ref in reference:
            f.write(' '.join(ref) + '\n')

    hypothesis_pipe = '\n'.join([' '.join(hyp) for hyp in hypotheses])

    pipe = subprocess.Popen(
        ["perl", 'bleu-1.04.pl', '-lc', 'tmp_reference.txt'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    pipe.stdin.write(hypothesis_pipe)
    pipe.stdin.close()
    return pipe.stdout.read()

# References:
# 1) Secondary symbols for telugu: http://www.learningtelugu.org/vowels-consonants-and-combinations.html
# 2) Unicodes: https://www.ssec.wisc.edu/~tomw/java/unicode.html
# TODO: Add secondary symbols for all the indian languages using Reference-2

telugu_secondary_symbols = [u'\u0c00', u'\u0c01', u'\u0c02', u'\u0c03', u'\u0c3E', u'\u0c3F', u'\u0c40', u'\u0c41', u'\u0c42', u'\u0c43', u'\u0c44', u'\u0c46', u'\u0c47', u'\u0c48', u'\u0c4A', u'\u0c4B', u'\u0c4C', u'\u0c4D', u'\u0c55', u'\u0c56', u'\u0c62', u'\u0c63']
hindi_secondary_symbols = [u'\u0900', u'\u0901', u'\u0902', u'\u0903', u'\u093A', u'\u093B', u'\u093C', u'\u093D', u'\u093E', u'\u093F', u'\u0940', u'\u0941', u'\u0942', u'\u0943', u'\u0944', u'\u0945', u'\u0946', u'\u0947', u'\u0948', u'\u0949', u'\u094A', u'\u094B', u'\u094C', u'\u094D', u'\u094E', u'\u094F', u'\u0953', u'\u0954', u'\u0955', u'\u0956', u'\u0957', u'\u0962', u'\u0963']

secondary_symbols = telugu_secondary_symbols + hindi_secondary_symbols

def combine_secondary_symbols(word):
    if isinstance(word, str):
        actual = list(word.decode('utf-8'))
    else:
        assert isinstance(word, list)
        actual = word
    res = []
    for i in actual:
        if i in secondary_symbols:
            if len(res) == 0:
                res.append(i)
            else:
                res[-1] += i
        else:
            res.append(i)
    return res

def print_attributes(instances, model):
    f = open('./data/training_metadata.txt', 'w')
    for instance in instances:
        attributes = instance[0].__dict__
        head = '>>> ' + instance[0].__class__.__name__ + ' : '+ instance[1]
        head = head.lstrip().upper()
        f.write(head + '\n')
        for i in attributes:
            if isinstance(attributes[i], int) or isinstance(attributes[i], str) or isinstance(attributes[i], float) or isinstance(attributes[i], bool):
                f.write('   >>> ' + i + ' : ' + str(attributes[i]) + '\n')
        f.write('\n\n')
    f.write('>>> MODEL\n')
    f.write(str(model) + '\n')
    f.close()