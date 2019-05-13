#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Neural Machine Translation base module
'''
# Start from line 308 and then follow DFS
# manner to understand the code structure.
# Import comet_ml in the top of your file
# from comet_ml import Experiment

# Create an experiment
# experiment = Experiment(api_key="G1Hu2kJ89qGE5sZ1cBNlrq6Hj",
#                         project_name="general", workspace="achaitanyasai",
#                         disabled=True)

# experiment.log_asset(file_path='./models/seq2seq_attn.py', file_like_object=None, file_name='seq2seq_attn.py', overwrite=False)
# experiment.log_asset(file_path='./modules/Trainer.py', file_like_object=None, file_name='trainer.py', overwrite=False)
# experiment.log_asset(file_path='./data_iterator.py', file_like_object=None, file_name='data_iterator.py', overwrite=False)
# experiment.add_tag('Expt-8')
# experiment.add_tag('vector_space=2')
# experiment.add_tag('UNK percentage=5')
# experiment.add_tag('Baseline')

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
from models import seq2seq_attn, seq2seq_attn_char_cnn
from modules import Loss, Optimizer, Trainer, Beam, Translator
import subprocess
import logging.config
import warnings
import datetime

use_cuda = torch.cuda.is_available()

#The below line is for reproducibility of results, refer:
# https://github.com/pytorch/pytorch/issues/114 

torch.backends.cudnn.deterministic = True

random.seed(3435)
np.random.seed(3435)
torch.manual_seed(3435)
torch.cuda.manual_seed_all(3435)
torch.cuda.manual_seed(3435)

logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

def parse_arguments():
    '''
    Parsing arguments
    '''
    logger.info('Parsing arguments')
    parser = argparse.ArgumentParser()
    data = parser.add_argument_group('data sets; model loading and saving')
    data.add_argument('--datasets', type=str, required=True, metavar='PATH', nargs=1,
                        help="parallel training corpus (source and target)")
    data.add_argument('--datasets_valid', type=str, required=True, metavar='PATH', nargs=1,
                        help="parallel validation corpus (source and target)")
    data.add_argument('--language_objects', type=str, metavar='PATH', nargs=2, 
                        default = ['./data/language_object.train.src.pkl', './data/language_object.train.trg.pkl'],
                        help="paths to pickle files for data_iterator.Lang objects (source and target).")
    data.add_argument('--language_objects_valid', type=str, metavar='PATH', nargs=2, 
                        default = ['./data/language_object.valid.src.pkl', './data/language_object.valid.trg.pkl'],
                        help="paths to pickle files for data_iterator.Lang objects (source and target).")
    data.add_argument('--saveTo', type=str, default='./data/model.pt', metavar='PATH', required=True,
                        help="location of model to save (default: %(default)s)")
    data.add_argument('--saveFreq', type=int, default=30000, metavar='INT',
                        help="save frequency (default: %(default)s)")
    data.add_argument('--overwrite', action='store_true',
                        help="write all models to same file")
    data.add_argument('--char', action='store_true',
                        help="use character cnn on words of encoder")
    
    network = parser.add_argument_group('network parameters')
    network.add_argument('--dim_word_src', type=int, default=500, metavar='INT',
                        help="source embedding layer size (default: %(default)s)")
    network.add_argument('--dim_word_trg', type=int, default=500, metavar='INT',
                        help="target embedding layer size (default: %(default)s)")
    network.add_argument('--dim', type=int, default=500, metavar='INT',
                        help="hidden layer size (default: %(default)s)")
    network.add_argument('--bidirectional', action="store_true",
                        help="bidirectional encoder (default: %(default)s)")
    network.add_argument('--enc_depth', type=int, default=1, metavar='INT',
                        help="number of encoder layers (default: %(default)s)")
    network.add_argument('--dec_depth', type=int, default=1, metavar='INT',
                        help="number of decoder layers (default: %(default)s)") #TODO: decoder with layers more than one(not sure how it works)
    network.add_argument('--dropout', type=float, default=0.2, metavar="FLOAT",
                        help="dropout (0: no dropout) (default: %(default)s)") #TODO: Seperate dropout for enc, dec, layers
    network.add_argument('--encoder', type=str, default='gru', choices=['gru', 'lstm'],
                        help='encoder recurrent layer (default: %(default)s)')
    network.add_argument('--decoder', type=str, default='gru', choices=['gru', 'lstm'],
                        help='first decoder recurrent layer (default: %(default)s)')
    
    training = parser.add_argument_group('training parameters')
    training.add_argument('--src_maxlen', type=int, default=50, metavar='INT',
                        help="maximum source sequence length (default: %(default)s)")
    training.add_argument('--tgt_maxlen', type=int, default=50, metavar='INT',
                        help="maximum target sequence length (default: %(default)s)")

    training.add_argument('--src_max_word_len', type=int, required=True, metavar='INT',
                        help="maximum source word length")
    training.add_argument('--tgt_max_word_len', type=int, required=True, metavar='INT',
                        help="maximum source word length")
    training.add_argument('--src_max_vocab_size', type=int, required=True, metavar='INT',
                        help="maximum source vocab size")
    training.add_argument('--tgt_max_vocab_size', type=int, required=True, metavar='INT',
                        help="maximum target vocab size")

    training.add_argument('--optimizer', type=str, default="sgd", 
                        choices=['adam', 'adadelta', 'rmsprop', 'sgd', 'sgdmomentum'],
                        help="optimizer (default: %(default)s)")
    training.add_argument('--lrate', type=float, default=1.0, metavar='FLOAT',
                        help="learning rate (default: %(default)s)")
    training.add_argument('--max_grad_norm', type=float, default=5, metavar='FLOAT',
                        help="TODO (default: %(default)s)")
    training.add_argument('--lrate_decay', type=float, default=0.5, metavar='FLOAT',
                        help="TODO (default: %(default)s)")
    training.add_argument('--start_decay_at', type=float, default=16, metavar='FLOAT',
                        help="TODO (default: %(default)s)")
    training.add_argument('--adam_beta1', type=float, default=0.9, metavar='FLOAT',
                        help="TODO (default: %(default)s)")
    training.add_argument('--adam_beta2', type=float, default=0.999, metavar='FLOAT',
                        help="TODO (default: %(default)s)")
    training.add_argument('--adagrad_accumulator_init', type=float, default=0, metavar='FLOAT',
                        help="TODO (default: %(default)s)")
    training.add_argument('--decay_method', type=str, default="", 
                        choices=['noam'], help="lr decay method (default: %(default)s)")
    training.add_argument('-warmup_steps', type=int, default=4000,
                       help="""Number of warmup steps for custom decay.""")
    training.add_argument('--batch_size', type=int, default=64, metavar='INT',
                        help="minibatch size (default: %(default)s)")
    training.add_argument('--max_epochs', type=int, default=20, metavar='INT',
                        help="maximum number of epochs (default: %(default)s)")
    training.add_argument('--max_number_of_sentences_allowed', type=int, default=1000000000000000, metavar='INT',
                        help="maximum number of sentences (default: %(default)s)")
    
    training.add_argument('--norm_method', type=str, default="sent", 
                        choices=['sent', 'word'], help="normalization method (default: %(default)s)")
    training.add_argument('--no_shuffle', action="store_false", dest="shuffle_each_epoch",
                        help="disable shuffling of training data (for each epoch)")
    training.add_argument('--objective', choices=['CE', 'MRT', 'RAML'], default='CE', #TODO: MRT, RAML
                        help='training objective. CE: cross-entropy minimization (default); MRT: Minimum Risk Training (https://www.aclweb.org/anthology/P/P16/P16-1159.pdf) \
                        RAML: Reward Augmented Maximum Likelihood (https://papers.nips.cc/paper/6547-reward-augmented-maximum-likelihood-for-neural-structured-prediction.pdf)')
    
    validation = parser.add_argument_group('validation parameters')
    validation.add_argument('--valid_datasets', type=str, default=None, metavar='PATH', nargs=2,
                        help="parallel validation corpus (source and target) (default: %(default)s)")
    validation.add_argument('--valid_batch_size', type=int, default=80, metavar='INT',
                        help="validation minibatch size (default: %(default)s)")
    validation.add_argument('--validFreq', type=int, default=10000, metavar='INT',
                        help="validation frequency (default: %(default)s)")
    validation.add_argument('--patience', type=int, default=10, metavar='INT',
                        help="early stopping patience (default: %(default)s)")
    
    display = parser.add_argument_group('display parameters')
    display.add_argument('--dispFreq', type=int, default=1, metavar='INT',
                        help="display loss after INT updates (default: %(default)s)")
    display.add_argument('--evaluateFreq', type=int, default=10, metavar='INT',
                        help="evaluate model after INT epochs (default: %(default)s)")
    display.add_argument('--sampleFreq', type=int, default=1000, metavar='INT',
                        help="display some samples after INT updates (default: %(default)s)")
    
    args = parser.parse_args()

    return args

def get_Lang_object(dataset_paths, pkl_file_paths, data_type):
    '''
    Loading/creating the data_iterator.Lang objects.
    If newly creating, the pickle files will be saved in ./data/ folder.
    '''
    if os.path.isfile(pkl_file_paths[0]) and os.path.isfile(pkl_file_paths[1]):
        '''
        Loading data_iterator.Lang objects for source and target languages.
        '''
        logger.info('Lang objects found, loading')
        source_data = util.read_pickle(pkl_file_paths[0])
        target_data = util.read_pickle(pkl_file_paths[1])
    else:
        '''
        Creating data_iterator.Lang objects for source and target languages.
        '''
        logger.info('Lang objects not found, creating')
        source_data = data_iterator.Lang(dataset_paths[0], args.src_maxlen, args.src_max_word_len, args.src_max_vocab_size, langtype='source', chars=False, verbose=True, data_type=data_type, ignore_too_many_unknowns = False, max_number_of_sentences_allowed=args.max_number_of_sentences_allowed)
        target_data = data_iterator.Lang(dataset_paths[1], args.tgt_maxlen, args.tgt_max_word_len, args.tgt_max_vocab_size, langtype='target', chars=False, verbose=True, data_type=data_type, ignore_too_many_unknowns = False, max_number_of_sentences_allowed=args.max_number_of_sentences_allowed)
        logger.info('Saving source lang at %s' % (pkl_file_paths[0]))
        util.write_pickle(source_data, pkl_file_paths[0], pickle.HIGHEST_PROTOCOL)
        logger.info('Saving target lang at %s' % (pkl_file_paths[1]))
        util.write_pickle(target_data, pkl_file_paths[1], pickle.HIGHEST_PROTOCOL)
    return source_data, target_data

def get_data(args):
    '''
    load data_iterator.Lang and build data_iterator.dataIterator objects
    '''
    source_data, target_data = get_Lang_object(args.datasets[0].split(), args.language_objects, 'train')
    source_data_valid, target_data_valid = get_Lang_object(args.datasets_valid[0].split(), args.language_objects_valid, 'valid')
    
    data_iter = data_iterator.dataIterator(source_data, target_data, shuffle=True)
    data_iter_valid = data_iterator.dataIterator(source_data_valid, target_data_valid, shuffle = False)
    
    return source_data, target_data, source_data_valid, target_data_valid, data_iter, data_iter_valid
    

def get_model(args, source_data, target_data):
    '''
    Build model
    '''
    #TODO: Make this user friendly.
    logger.info('Building encoder')    
    encoder = seq2seq_attn.CustomEncoder(
        'LSTM',
        bidirectional = True,
        num_layers = args.enc_depth, 
        hidden_size = 500,
        vocab_size = source_data.n_words,
        embedding_dim = 100,
        pad_token = source_data.word2idx['PAD'], 
        ngrams_vocab_size = source_data.n_hashes,
        dropout = 0.3,
        ngram_pad_token = 0).cuda()

    logger.info('Building decoder')    
    decoder = seq2seq_attn.decoder(
        'LSTM', 
        bidirectional_encoder = True,
        num_layers = args.dec_depth,
        hidden_size = 500,
        vocab_size = target_data.n_words,
        ngrams_vocab_size = target_data.n_hashes,
        embedding_dim = 500,
        pad_token = target_data.word2idx['PAD'],
        attn_type = 'general',
        dropout = 0.3,
        ngram_pad_token = 0).cuda()
    
    model = seq2seq_attn.Seq2SeqAttention(encoder, decoder).cuda()
    logger.info('Initializing model parameters')
    #Refer: https://discuss.pytorch.org/t/initializing-embeddings-for-nmt-matters-a-lot/10517
    for p in model.parameters():
        p.data.uniform_(-0.1, 0.1)
    
    with open('../dataset/adagram_vectors/emb1.txt') as f:
        print 'Reading emb1.txt'
        o = 0
        for j, line in enumerate(f.readlines()):
            if j == 0:
                continue
            cur = line.strip().split(' ')
            word = cur[0]
            wv = map(float, cur[1::])
            if word in source_data.word2idx:
                o += 1
                if 0 == wv[0] and 0 == wv[1] and 0 == wv[2] and 0 == wv[3]:
                    continue
                idx = source_data.word2idx[word]
                model.encoder.embeddings1.weight.data[idx] = torch.tensor(wv).cuda()
    with open('../dataset/adagram_vectors/emb2.txt') as f:
        print 'Reading emb2.txt'
        o1 = 0
        for j, line in enumerate(f.readlines()):
            if j == 0:
                continue
            cur = line.strip().split(' ')
            word = cur[0]
            wv = map(float, cur[1::])
            if word in source_data.word2idx:
                o1 += 1
                idx = source_data.word2idx[word]
                if 0 == wv[0] and 0 == wv[1] and 0 == wv[2] and 0 == wv[3]:
                    continue
                model.encoder.embeddings2.weight.data[idx] = torch.tensor(wv).cuda()
    with open('../dataset/adagram_vectors/emb3.txt') as f:
        print 'Reading emb3.txt'
        o2 = 0
        for j, line in enumerate(f.readlines()):
            if j == 0:
                continue
            cur = line.strip().split(' ')
            word = cur[0]
            wv = map(float, cur[1::])
            if word in source_data.word2idx:
                o2 += 1
                if 0 == wv[0] and 0 == wv[1] and 0 == wv[2] and 0 == wv[3]:
                    continue
                idx = source_data.word2idx[word]
                model.encoder.embeddings3.weight.data[idx] = torch.tensor(wv).cuda()
    
    print(o)
    print(o1)
    print(o2)
    
    logger.info(model)
    return None, None, model

def load_model():
    model = torch.load('./data/model.pt')
    model.eval()
    logger.info(model)
    return None, None, model

def get_optimizer(args, model, verbose=True):
    '''
    Build optimizer
    '''
    optimizer = Optimizer.Optimizer(
        args.optimizer,
        args.lrate,
        args.max_grad_norm,
        args.lrate_decay,
        args.start_decay_at,
        args.adam_beta1,
        args.adam_beta2,
        args.adagrad_accumulator_init,
        args.decay_method,
        args.warmup_steps,
        args.dim,
        verbose
    )
    optimizer.set_parameters(model.parameters())
    return optimizer

def get_criterion(args, target_data):
    '''
    Build error criterion.
    Only supports NLLLoss()
    '''
    criterion = Loss.NMTLoss(target_data.word2idx, target_data.word2idx['PAD'])
    return criterion.cuda()

def predict_and_get_bleu_score(args, model, source_data, target_data, source_data_valid, 
                               target_data_valid, train_data_iterator, valid_data_iterator):
    
    scorer = Beam.GNMTGlobalScorer(0.0, -0.0)
    
    translator = Translator.Translator(model, source_data, target_data, valid_data_iterator,
                                       beam_size=5,
                                       n_best=1,
                                       global_scorer=scorer,
                                       max_length=50,
                                       copy_attn=False,
                                       cuda=True,
                                       beam_trace=False,
                                       min_length=0)
    
    builder = Translator.TranslationBuilder(source_data, target_data, valid_data_iterator,
                                            1, True)

    pred_score_total, pred_words_total = 0, 0
    cur_batch = 0

    f = open('./data/predicted.txt', 'w')

    while cur_batch < valid_data_iterator.n_samples:
        batch = valid_data_iterator.next_batch(10, 50, source_language_model = source_data, target_language_model = target_data)
        batch_data = translator.translate_batch(batch)
        
        translations = builder.from_batch(batch_data, batch)
        
        for trans in translations:
            pred_score_total += trans.pred_scores[0]
            pred_words_total += len(trans.pred_sents[0])
            n_best_preds = [' '.join(pred) for pred in trans.pred_sents[: 1]]
            f.write('\n'.join(n_best_preds))
            f.write('\n')
            f.flush()
        cur_batch += batch.batch_size
        sys.stdout.write('%d/%d       \r' % (cur_batch, valid_data_iterator.n_samples))
        sys.stdout.flush()

    f.close()
    cmd = './scripts/bleu-1.04.pl %s < ./data/predicted.txt' % (args.datasets_valid[0].split()[1])
    p = subprocess.Popen(cmd, bufsize = 4096, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
    out, error = p.communicate()
    return float(out.split(',')[0].replace('BLEU = ', ''))

def train_model(args, model, train_criterion, valid_criterion, optimizer,
                source_data, target_data, source_data_valid, target_data_valid, 
                train_data_iterator, valid_data_iterator):
    
    
    #normalization method: whether to normalize the loss per sentences or per words.
    norm_method = args.norm_method

    #Instantiate the trainer module. 
    prev = 100000000000000000000
    best_bleu_sofar = 0
    trainer = Trainer.Trainer(model, train_criterion, valid_criterion, optimizer, norm_method=norm_method)
    
    instances = [
        (trainer, ''), (trainer.optimizer, ''), (trainer.train_loss, 'Train loss'),
        (trainer.valid_loss, 'Valid loss'), (source_data, 'Source training data'),
        (target_data, 'Target training data'), (train_data_iterator, 'Training iterator'),
        (source_data_valid, 'Source validation data'), (target_data_valid, 'Target validation data'),
        (valid_data_iterator, 'Validation iterator'), (model.encoder, 'Model encoder'),
        (model.decoder, 'Model decoder')
    ]
    logger.info('Writing training metadata to ./data/training_metadata.txt')
    util.print_attributes(instances, model)

    logger.info('Training')
    start_time = time.time()

    for epoch in xrange(1, args.max_epochs + 1):
        # Train the model on training set
        train_data_iterator.reset()
        train_stats = trainer.train(args, train_data_iterator, source_data, target_data, epoch)
        msg = ('=' * 80) + '\nEpoch: %d, Train Loss: %.3f, Train Accuracy: %.3f, Train Perplexity: %.3f'
        msg = msg % (epoch, train_stats._loss(), train_stats.accuracy(), train_stats.perplexity())
        logger.info(msg)
        # experiment.log_metric("train loss", train_stats._loss(), step=epoch)
        # experiment.log_metric("train accuracy", train_stats.accuracy(), step=epoch)

        # Validate the model on validation set
        valid_data_iterator.reset()
        valid_stats = trainer.validate(args, valid_data_iterator, source_data, target_data)
        msg = 'Epoch: %d, Valid Loss: %.3f, Valid Accuracy: %.3f, Valid Perplexity: %.3f'
        msg = msg % (epoch,valid_stats._loss(),valid_stats.accuracy(),valid_stats.perplexity())
        # experiment.log_metric("valid loss", valid_stats._loss(), step=epoch)
        # experiment.log_metric("valid accuracy", valid_stats.accuracy(), step=epoch)
        
        logger.info(msg)
        valid_data_iterator.reset()
        model.eval()
        bleu_score = predict_and_get_bleu_score(args, model, source_data, target_data, source_data_valid, 
                                   target_data_valid, train_data_iterator, valid_data_iterator)
        
        msg = 'Epoch: %d, Valid BLEU Score: %.4f, Best BLEU Score: %.4f'
        msg = msg % (epoch, bleu_score, max(bleu_score, best_bleu_sofar))
        # experiment.log_metric("valid bleu score", bleu_score, step=epoch)
        logger.info(msg)

        if(bleu_score > best_bleu_sofar):
            logger.info('Saving model')            
            torch.save(model, args.saveTo)
        torch.save(model, './data/model_epoch_%d_valid_ppl_%.4f_bleu_%.4f_.pt' % (epoch, valid_stats.perplexity(), bleu_score))
        best_bleu_sofar = max(best_bleu_sofar, bleu_score)
        trainer.epoch_step(-bleu_score, epoch) #Just a work around to replace bleuscore inplace of ppl.
        model.train()
        elapsed = time.time() - start_time
        forcast = int(((args.max_epochs - epoch) * elapsed) / epoch)
        m, s = divmod(forcast, 60)
        h, m = divmod(m, 60)
        msg = '\nETA  : %d hours, %d mins\n' + ('=' * 80)
        msg = msg % (h, m)
        logger.info(msg)

if __name__ == '__main__':
    #parse command line arguments
    args = parse_arguments()

    #Load data
    source_data, target_data, source_data_valid, target_data_valid, train_data_iterator, valid_data_iterator = get_data(args)
    
    #Build/Load model
    encoder, decoder, model = get_model(args, source_data, target_data)
    #encoder, decoder, model = load_model()

    #Build criterion
    train_criterion = get_criterion(args, target_data)
    valid_criterion = get_criterion(args, target_data)

    #Build optimizer
    optimizer = get_optimizer(args, model)
    #Train model
    train_model(args, model, train_criterion, valid_criterion, optimizer,
                source_data, target_data, source_data_valid, target_data_valid, 
                train_data_iterator, valid_data_iterator)
    
