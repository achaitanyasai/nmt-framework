#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Translate the given data
'''

import argparse
import logging
import math
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

import data_iterator_optimized as data_iterator
import util
from models import seq2seq_attn
from modules import Beam, Loss, Optimizer, Trainer, Translator

use_cuda = torch.cuda.is_available()

#The below line is for reproducibility of results, refer:
# https://github.com/pytorch/pytorch/issues/114 

torch.backends.cudnn.deterministic = True

random.seed(3435)
np.random.seed(3435)
torch.manual_seed(3435)
torch.cuda.manual_seed_all(3435) #For multiple gpu's
torch.cuda.manual_seed(3435)

level = logging.INFO
logging.basicConfig(level=level, format='')

def parse_arguments():
    '''
    Parsing arguments
    '''
    logging.info('Parsing arguments')
    parser = argparse.ArgumentParser()
    data = parser.add_argument_group('data sets; model loading and saving')
    data.add_argument('--input', type=str, required=True, metavar='PATH', nargs=1,
                        help="path to test file")
    data.add_argument('--output', type=str, required=True, metavar='PATH',
                        help="path to output")
    data.add_argument('--language_objects_train', type=str, metavar='PATH', nargs=2, 
                        default = ['./data/language_object.train.src.pkl', './data/language_object.train.trg.pkl'],
                        help="paths to training pickle files for data_iterator.Lang objects(source and target).")
    data.add_argument('--language_objects_test', type=str, metavar='PATH', nargs=1, 
                        default = ['./data/language_object.test.src.pkl'],
                        help="paths to test set pickle file for data_iterator.Lang object (source).")
    data.add_argument('--model', type=str, default='./data/model.pt', metavar='PATH', required=True,
                        help="path to trained model")
    data.add_argument('--beam_size', type=int, default=5, metavar='INT',
                        help="beam size (default: %(default)s)")
    data.add_argument('--maxlen', type=int, default=50, metavar='INT',
                        help="maximum length of sentence (default: %(default)s)")
    data.add_argument('--batch_size', type=int, default=5, metavar='INT',
                        help="number of samples per batch to decode (default: %(default)s)")
    data.add_argument('--minlen', type=int, default=0, metavar='INT',
                        help="minimum length of sentence (default: %(default)s)")
    # Alpha and Beta values for Google Length + Coverage penalty
    # Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
    data.add_argument('--alpha', type=float, default=0.0, metavar='FLOAT',
                        help="Google NMT length penalty parameter (default: %(default)s)")
    data.add_argument('--beta', type=float, default=-0.0, metavar='FLOAT',
                        help="Coverage penalty parameter (default: %(default)s)")
    data.add_argument('--n_best', type=int, default=1, metavar='INT',
                        help="number of best prediction sentences (default: %(default)s)")
    data.add_argument('--dump_beam', type=str, default='./data/beam.log', metavar='PATH',
                        help="file to dump beam output")
    data.add_argument('--replace_unk', action="store_true", required=True,
                        help="Replace the generated UNK tokens with the source token that had highest attention weight. (default: %(default)s)")
    data.add_argument('--char', action='store_true',
                        help="use character cnn on words of encoder")
    args = parser.parse_args()
    return args

def get_Lang_object(dataset_paths, pkl_file_paths):
    '''
    Loading/creating the data_iterator.Lang objects.
    If newly creating, the pickle files will be saved in ./data/ folder.
    '''
    def get_single_object(dataset_path, pkl_file_path, _type, chars):
        if pkl_file_path == None:
            return None
        if os.path.isfile(pkl_file_path):
            logging.info('%s lang object found, loading' % _type)
            data = util.read_pickle(pkl_file_path)
        else:
            logging.info('%s lang object not found, creating' % _type)
            data = data_iterator.Lang(dataset_path, args.maxlen, 30, 10000000, langtype=_type, chars=False, verbose=True, data_type='test')
        return data
    source_data = get_single_object(dataset_paths[0], pkl_file_paths[0], 'source', chars = args.char)
    target_data = None
    if len(dataset_paths) == 2:
        target_data = get_single_object(dataset_paths[1], pkl_file_paths[1], 'target', chars = False)
    return source_data, target_data

def get_data(args):
    '''
    load data_iterator.Lang and build data_iterator.dataIterator objects
    '''
    source_data, target_data = get_Lang_object([None, None], args.language_objects_train)
    source_data_test, target_data_test = get_Lang_object(args.input, args.language_objects_test)
    data_iter = data_iterator.dataIterator(source_data, target_data)
    data_iter_test = data_iterator.dataIterator(source_data_test, None, shuffle = False)

    return source_data, target_data, source_data_test, None, data_iter, data_iter_test

def load_model(args):
    print args.model
    model = torch.load(args.model)
    model.eval()
    return model

if __name__ == '__main__':
    args = parse_arguments()
    
    #Load data
    source_data, target_data, test_source_data, test_target_data, train_data_iterator, test_data_iterator = get_data(args)

    # print source_data.word2idx['may']
    # exit(-1)
    #Load model
    model = load_model(args)

    #Build scorer
    scorer = Beam.GNMTGlobalScorer(args.alpha, args.beta)

    #Build translator
    
    translator = Translator.Translator(model, source_data, target_data, test_data_iterator,
                                           beam_size=args.beam_size,
                                           n_best=args.n_best,
                                           global_scorer=scorer,
                                           max_length=args.maxlen,
                                           copy_attn=False,
                                           cuda=True,
                                           beam_trace=args.dump_beam != "",
                                           min_length=args.minlen)
    
    builder = Translator.TranslationBuilder(source_data, target_data, test_data_iterator,
                                                    args.n_best, args.replace_unk)

    pred_score_total, pred_words_total = 0, 0
    cur_batch = 0

    f = open(args.output, 'w')

    while cur_batch < test_data_iterator.n_samples:
        batch = test_data_iterator.next_batch(args.batch_size, args.maxlen, source_language_model = source_data, target_language_model = target_data)
        batch_data = translator.translate_batch(batch)
        
        translations = builder.from_batch(batch_data, batch)
        
        for trans in translations:
            pred_score_total += trans.pred_scores[0]
            pred_words_total += len(trans.pred_sents[0])
            n_best_preds = [' '.join(pred) for pred in trans.pred_sents[: args.n_best]]
            f.write('\n'.join(n_best_preds))
            f.write('\n')
            f.flush()
        cur_batch += batch.batch_size
        sys.stdout.write('%d/%d       \r' % (cur_batch, test_data_iterator.n_samples))
        sys.stdout.flush()

    f.close()
