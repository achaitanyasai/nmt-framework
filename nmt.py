#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Neural Machine Translation base module
'''
import json

from comet_ml import Experiment
import argparse
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

from data_iterator import *
import data_iterator
import util
from models import seq2seq_attn, seq2seq_attn_baseline, seq2seq_attn_multivec, seq2seq_attn_baseline_word_attn, seq2seq_attn_multivec_word_attn
from modules import Loss, Optimizer, Trainer, Beam, Translator
from modules import  utils as module_utils
import subprocess
import warnings
import datetime
# from structs import *
import neptune

#The below line is for reproducibility of results, refer:
# https://github.com/pytorch/pytorch/issues/114 

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

random.seed(346)
np.random.seed(346)
torch.manual_seed(346)
torch.cuda.manual_seed_all(346)
torch.cuda.manual_seed(346)

def parse_arguments():
    logger.info("Parsing arguments")
    parser = argparse.ArgumentParser()

    data = parser.add_argument_group("Data sets; model loading and saving; Hyper parameters related to data")
    data.add_argument("--train_dataset", type=str, required=True, metavar='PATH', help='Path to training corpus in csv file')
    data.add_argument("--valid_dataset", type=str, required=True, metavar='PATH', help='Path to validation corpus in csv file')
    data.add_argument("--test_dataset", type=str, required=True, metavar='PATH', help='Path to test corpus in csv file')
    data.add_argument("--adagram_embeddings_dir", type=str, required=True, metavar='PATH', help='Path to adagram initialized embeddings')
    data.add_argument("--testset_target", type=str, required=True, metavar='PATH', help='Path to target language testset')
    data.add_argument('--save_to', type=str, required=True, metavar='PATH', help='Location of model to save')
    data.add_argument('--save_freq', type=int, required=True, metavar='INT', help='Saving frequency (epochs or steps)')
    data.add_argument('--source_max_len', type=int, required=True, metavar='INT', help='Maximum length of the source sentence. Only applied to training set')
    data.add_argument('--target_max_len', type=int, required=True, metavar='INT', help='Maximum length of the target sentence. Only applied to training set')
    data.add_argument('--source_max_vocab_size', type=int, required=True, metavar='INT', help='Maximum vocabulary size of source language. Only applied to training set')
    data.add_argument('--target_max_vocab_size', type=int, required=True, metavar='INT', help='Maximum vocabulary size of target language. Only applied to training set')
    data.add_argument('--ignore_too_many_unknowns', type=int, required=True, metavar='INT', help='Ignore too many unknowns. Only applies to training set')
    data.add_argument('--path_to_logs', type=str, required=True, metavar='PATH', help='PATH to logs')

    network = parser.add_argument_group('Network hyper parameters')
    network.add_argument('--model_type', type=str, required=True, metavar='STR', help='Model type i.e., baseline/multivec/etc.')
    network.add_argument('--bidirectional', type=int, required=True, metavar='INT', help='Bidirectional RNN encoder')
    network.add_argument('--encoder_num_layers', type=int, required=True, metavar='INT', help='Number of layers in RNN Encoder')
    network.add_argument('--encoder_hidden_dim', type=int, required=True, metavar='INT', help='Hidden dimension in RNN Encoder')
    network.add_argument('--encoder_embedding_dim', type=int, required=True, metavar='INT', help='Embedding dimension for RNN Encoder')
    network.add_argument('--encoder_dropout', type=float, required=True, metavar='FLOAT', help='Dropout between layers in RNN Encoder')

    network.add_argument('--decoder_num_layers', type=int, required=True, metavar='INT', help='Number of layers in RNN Decoder')
    network.add_argument('--decoder_hidden_dim', type=int, required=True, metavar='INT', help='Hidden dimension in RNN Decoder')
    network.add_argument('--decoder_embedding_dim', type=int, required=True, metavar='INT', help='Embedding dimension for RNN Decoder')
    network.add_argument('--decoder_dropout', type=float, required=True, metavar='FLOAT', help='Dropout between layers in RNN Decoder')
    # TODO: Add more attention options: mlp, dot, etc. (Bahdanau, Luong)
    network.add_argument('--decoder_attention_type', type=str, required=True, choices=['general'], help='Attention type between encoder outputs and decoder hidden state')

    training = parser.add_argument_group('Training hyper parameters')
    training.add_argument('--use_epochs', action='store_true', help='Use epochs instead of steps/updates')
    training.add_argument('--gradient_checks', action='store_true', help='Check gradients for all the weights during each update during training')
    training.add_argument('--steps', type=int, required=True, metavar='INT', help='Number of steps. If --use_epochs is used, then the value should be number of epochs')
    training.add_argument('--batch_size', type=int, required=True, metavar='INT', help='Batch size')
    training.add_argument('--valid_steps', type=int, required=True, metavar='INT', help='Model will be evaluated on validation set after every valid_steps')
    training.add_argument('--patience_steps', type=int, required=True, metavar='INT', help='If the validation loss does not improve after these many steps, the training will stop')
    training.add_argument('--norm_method', type=str, required=True, choices=['sents', 'tokens'], help='Normalization method')
    training.add_argument('--expt_name', type=str, required=True, help='Experiment name for comet.ml')


    opt = parser.add_argument_group('Optimizer hyper parameters')
    opt.add_argument('--optimizer', type=str, required=True, choices=['sgd', 'adam'], help='Optimizer')
    opt.add_argument('--lrate', type=float, required=True, metavar='FLOAT', help='Learning rate')
    opt.add_argument('--max_grad_norm', type=float, required=True, metavar='FLOAT', help='Max Gradient Norm')
    opt.add_argument('--start_decay_at', type=int, required=True, metavar='INT', help='Number of steps after which learning rate starts decaying')
    opt.add_argument('--decay_lrate_steps', type=float, required=True, metavar='FLOAT', help='Number of steps between two learning rate decays')
    opt.add_argument('--adam_beta1', type=float, required=True, metavar='FLOAT', help='Adam beta1')
    opt.add_argument('--adam_beta2', type=float, required=True, metavar='FLOAT', help='Adam beta2')
    opt.add_argument('--lrate_decay', type=float, required=True, metavar='FLOAT', help='learning rate decay factor')
    opt.add_argument('--warmup_steps', type=float, required=True, metavar='FLOAT', help='warmup steps for adam optimizer') # TODO: Need to learn about warmup steps

    args = parser.parse_args()

    args.bidirectional = args.bidirectional == 1
    args.ignore_too_many_unknowns = args.ignore_too_many_unknowns == 1

    #TODO: Restructure these checks.
    assert args.expt_name != ''
    assert args.expt_name is not None

    logger.error('Change store-true arguments to integers. arg-switch in guild.yml is not working')
    # TODO: Verify arguments.

    return args

def getIterators(args):
    logger.info("Loading train iterator")
    train_iterator = data_iterator.DataIterator(fields=None, fname=args.train_dataset, shuffle=True,
                                                data_type='train', src_max_len=args.source_max_len, tgt_max_len=args.target_max_len,
                                                src_max_vocab_size=args.source_max_vocab_size, tgt_max_vocab_size=args.target_max_vocab_size,
                                                ignore_too_many_unknowns=args.ignore_too_many_unknowns,
                                                break_on_stop_iteration=args.use_epochs)

    logger.info("Loading valid iterator")
    valid_iterator = data_iterator.DataIterator(fields=(train_iterator.sourceField, train_iterator.targetField),
                                                fname=args.valid_dataset, shuffle=False,
                                                data_type='valid')

    logger.info("Loading test iterator")
    test_iterator = data_iterator.DataIterator(fields=(train_iterator.sourceField, train_iterator.targetField),
                                                fname=args.test_dataset, shuffle=False,
                                                data_type='test')
    return TrainValidTestIterator(train_iterator, valid_iterator, test_iterator)

def initialize_model_parameters(path, model, source_word2idx):
    for p in model.parameters():
        p.data.uniform_(-0.1, 0.1)
    try:
        for emb_idx in range(1, 4):
            logger.info('Reading %s/emb%d.txt' % (path, emb_idx))
            is_single = False
            with open('%s/emb%d.txt' % (path, emb_idx)) as f:
                o = 0
                for j, line in enumerate(f.readlines()):
                    if j == 0:
                        continue
                    cur = line.strip().split(' ')
                    word = cur[0].lower()
                    wv = list(map(float, cur[1::]))
                    if word in source_word2idx:
                        o += 1
                        if 0 == wv[0] and 0 == wv[1] and 0 == wv[2] and 0 == wv[3]:
                            continue
                        idx = source_word2idx[word]
                        try:
                            if emb_idx == 1:
                                model.encoder.embeddings1.weight.data[idx] = torch.tensor(wv).cuda()
                            elif emb_idx == 2:
                                model.encoder.embeddings2.weight.data[idx] = torch.tensor(wv).cuda()
                            elif emb_idx == 3:
                                model.encoder.embeddings3.weight.data[idx] = torch.tensor(wv).cuda()
                            else:
                                assert False
                        except AttributeError:
                            model.encoder.embeddings.weight.data[idx] = torch.tensor(wv).cuda()
                            is_single = True
                logger.info('Number of words found in monolingual corpus for emb%d: %d out of %d' % (emb_idx, o, len(source_word2idx)))
                if is_single:
                    break
    except IOError as e:
        pass

        # Uncomment below lines to freeze embeddings. It's discouraged to freeze though.
        # logger.warn('Freezing embeddings')
        # model.encoder.embeddings1.weight.requires_grad = False
        # model.encoder.embeddings2.weight.requires_grad = False
        # model.encoder.embeddings3.weight.requires_grad = False


def buildModel(args, train_iterator):
    __model = None
    if args.model_type == 'seq2seq_baseline':
        __model = seq2seq_attn_baseline
    elif args.model_type == 'seq2seq_baseline_word_attn':
        __model = seq2seq_attn_baseline_word_attn
    elif args.model_type == 'seq2seq_multivec':
        __model = seq2seq_attn_multivec
    elif args.model_type == 'seq2seq_multivec_word_attn':
        __model = seq2seq_attn_multivec_word_attn
    else:
        assert False
    assert __model is not None
    logger.info('Building Encoder')
    encoder = __model.encoder(
        'LSTM',
        bidirectional=args.bidirectional,
        num_layers=args.encoder_num_layers,
        hidden_size=args.encoder_hidden_dim,
        vocab_size=train_iterator.sourceField.nWords,
        embedding_dim=args.encoder_embedding_dim,
        pad_token=train_iterator.sourceField.word2idx['PAD'],
        dropout=args.encoder_dropout
    ).cuda()

    logger.info('Building Decoder')
    decoder = __model.decoder(
        'LSTM',
        bidirectional_encoder=args.bidirectional,
        num_layers=args.decoder_num_layers,
        hidden_size=args.decoder_hidden_dim,
        vocab_size=train_iterator.targetField.nWords,
        embedding_dim=args.decoder_embedding_dim,
        pad_token=train_iterator.targetField.word2idx['PAD'],
        attn_type=args.decoder_attention_type,
        dropout=args.decoder_dropout
    ).cuda()

    logger.info('Building Model')
    model = __model.Seq2SeqAttention(encoder, decoder).cuda()

    logger.info('Initializing Model Parameters')
    initialize_model_parameters(args.adagram_embeddings_dir, model, train_iterator.sourceField.word2idx)
    logger.info(model)
    return model

def loadModel(args):
    logger.info('Loading model')
    pass

def getOptimizer(args):
    logger.info('Building optimizer')
    optimizer = Optimizer.Optimizer(
        args.optimizer,
        args.lrate,
        args.max_grad_norm,
        args.lrate_decay,
        args.start_decay_at,
        args.decay_lrate_steps,
        args.adam_beta1,
        args.adam_beta2,
        args.patience_steps
    )

    optimizer.set_parameters(args.model.parameters())
    return optimizer

def getCriterions(args):
    train_criterion = Loss.NMTLoss(target_vocabulary_len=args.iterators.train_iterator.targetField.nWords, target_padding_idx=1,
                              reduction='sum', perform_dimension_checks=True).cuda()
    valid_criterion = Loss.NMTLoss(target_vocabulary_len=args.iterators.train_iterator.targetField.nWords, target_padding_idx=1,
                              reduction='sum', perform_dimension_checks=True).cuda()
    return Criterions(train_criterion, valid_criterion)

def test(args):
    pass

intervals = (
    ('w', 604800),  # 60 * 60 * 24 * 7
    ('d', 86400),    # 60 * 60 * 24
    ('h', 3600),    # 60 * 60
    ('m', 60),
    ('s', 1),
    )

def display_time(seconds, granularity=2):
    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{}{}".format(int(value), name))
    return ', '.join(result[:granularity])

def train(args):

    trainer = Trainer.Trainer(args)
    best_loss = 1e9
    best_accuracy = -1
    best_loss_real = 1e9
    best_accuracy_real = -1
    start_time = time.time()
    logger.info('Training with validation_steps: %d, patience: %d, model: %s' % (args.valid_steps, args.patience_steps, args.model_type))
    processed_sentences = 0
    sentences_start_time = time.time()
    try:
      for steps in range(args.steps):
        batch = args.iterators.train_iterator.next_batch(args.batch_size)
        batch.transpose()
        processed_sentences += batch.batch_size
        stats = trainer.propagate(batch, steps)
        # Previously, it's: trainer.lr_step(valid_stats._loss(), steps)
        # trainer.lr_step(stats._loss(), steps)
        end_time = time.time()
        elapsed = end_time - start_time
        remaining_time = ((args.steps * elapsed) / float(steps + 1)) - elapsed
        if steps % 10 == 0:
            speed = round(processed_sentences / (time.time() - sentences_start_time))
            sentences_start_time = time.time()
            processed_sentences = 0

            logger.info('Step: %d, train_loss: %.6f, train_accuracy: %.6f, ETA: %s, Speed: %d sents/sec' % (steps, stats._loss(), stats.accuracy(), display_time(remaining_time, 5), speed))
            # neptune.log_metric('Training Loss', stats._loss())
            # neptune.log_metric('Training Accuracy', stats.accuracy())

            experiment.log_metric('Training Loss', stats._loss(), step=steps)
            experiment.log_metric('Training Accuracy', stats.accuracy(), step=steps)
            if steps >= args.steps - 20:
                torch.save(args.model.state_dict(), "./data/model_every_10steps.pt")

        if steps >= args.valid_steps and steps % args.valid_steps == 0:
            args.iterators.valid_iterator.reset()
            valid_stats = trainer.validate(args.iterators.valid_iterator, b_size=20)

            args.iterators.valid_iterator.reset()
            if steps >= args.steps - 1500:
                valid_stats1 = trainer.validate_fixed(args.iterators.valid_iterator, b_size=20)
            else:
                valid_stats1 = module_utils.Statistics()

            trainer.lr_step(valid_stats._loss(), steps)
            logger.info('\n' + '=' * 20 + '\nStep: %d, valid_loss: %.6f | %.6f, valid_accuracy: %.6f | %.6f, LR: %.9f, patience: %d\n' % (
            steps, valid_stats._loss(), valid_stats1._loss(), valid_stats.accuracy(), valid_stats1.accuracy(), args.optimizer.lrate, args.optimizer.patience) + '=' * 20)

            cur_loss = valid_stats._loss()
            cur_acc = valid_stats.accuracy()

            cur_loss_real = valid_stats1._loss()
            cur_acc_real = valid_stats1.accuracy()
            # neptune.log_metric('Validation Loss', valid_stats._loss())
            # neptune.log_metric('Validation Accuracy', valid_stats.accuracy())

            experiment.log_metric('Validation Loss', valid_stats._loss(), step=steps)
            experiment.log_metric('Validation Accuracy', valid_stats.accuracy(), step=steps)

            experiment.log_metric('Validation Loss_real', valid_stats1._loss(), step=steps)
            experiment.log_metric('Validation Accuracy_real', valid_stats1.accuracy(), step=steps)
            if best_loss > cur_loss or best_accuracy < cur_acc:
                logger.info('Saving model to %s' % args.save_to)
                torch.save(args.model.state_dict(), args.save_to)

            if best_loss_real > cur_loss_real or best_accuracy_real < cur_acc_real:
                logger.info('Saving model to ./data/model_real.pt')
                torch.save(args.model.state_dict(), './data/model_real.pt')

            best_loss = min(best_loss, cur_loss)
            best_accuracy = max(best_accuracy, cur_acc)

            best_loss_real = min(best_loss_real, cur_loss_real)
            best_accuracy_real = max(best_accuracy_real, cur_acc_real)

        if steps >= 2000 and steps % 2000 == 0:
            logger.info('Sleeping for 2 minutes')
            time.sleep(120)
        if args.optimizer.lrate <= 1e-7:
            torch.save(args.model.state_dict(), "./data/model_every_10steps.pt")
            break
    except KeyboardInterrupt:
      pass

    # Loading the best parameters
    args.model.load_state_dict(torch.load('./data/model.pt'))
    args.model.eval()

    args.iterators.valid_iterator.reset()
    valid_stats = trainer.validate(args.iterators.valid_iterator, b_size=10)

    args.iterators.valid_iterator.reset()
    valid_stats1 = trainer.validate_fixed(args.iterators.valid_iterator, b_size=10)

    logger.info(
        '\n' + '=' * 20 + '\n[BEST] valid_loss: %.6f | %.6f, valid_accuracy: %.6f | %.6f, LR: %.6f, patience: %d\n' % (
            valid_stats._loss(), valid_stats1._loss(), valid_stats.accuracy(), valid_stats1.accuracy(),
            args.optimizer.lrate, args.optimizer.patience) + '=' * 20)

    args.iterators.test_iterator.reset()
    test_stats = trainer.validate(args.iterators.test_iterator, b_size=10)

    args.iterators.test_iterator.reset()
    test_stats1 = trainer.validate_fixed(args.iterators.test_iterator, b_size=10)
    logger.info(
        '\n' + '#' * 70 + '\n[BEST] test_loss: %.6f | %.6f, test_accuracy: %.6f | %.6f\n' % (
            test_stats._loss(), test_stats1._loss(), test_stats.accuracy(), test_stats1.accuracy()))



def translate(args, iterators):
    # args.model.load_state_dict(torch.load('/home/chaitanya/PycharmProjects/venv/.guild/runs/55ea009d2bc84dc18706c99d8027a20e/data/model.pt'))
    args.model.eval()

    scorer = Beam.GNMTGlobalScorer(0.0, 0.0)

    # Build translator

    translator = Translator.Translator(args.model, iterators.train_iterator.sourceField,
                                       iterators.train_iterator.targetField, iterators.test_iterator,
                                       beam_size=5,
                                       n_best=1,
                                       global_scorer=scorer,
                                       max_length=50,
                                       copy_attn=False,
                                       cuda=True,
                                       beam_trace=False,
                                       min_length=0)

    iterators.test_iterator.reset()
    builder = Translator.TranslationBuilder(iterators.train_iterator.sourceField, iterators.train_iterator.targetField,
                                            iterators.test_iterator,
                                            1, True)

    pred_score_total, pred_words_total = 0, 0
    cur_batch = 0

    f = open('./data/predicted.txt', 'w')
    translations_with_attn = []
    assert iterators.test_iterator.nSentencesPruned == iterators.test_iterator.nSentences
    while cur_batch < iterators.test_iterator.nSentences:
        batch = iterators.test_iterator.next_batch(
            1)  # , args.maxlen, source_language_model = source_data, target_language_model = target_data)
        batch_data = translator.translate_batch(batch)

        translations = builder.from_batch(batch_data, batch)

        for trans in translations:
            pred_score_total += trans.pred_scores[0]
            pred_words_total += len(trans.pred_sents[0])
            n_best_preds = [' '.join(pred) for pred in trans.pred_sents[: 1]]
            # translations_with_attn.append(
            #     {
            #         'attentions':trans.attns[0].data.cpu().numpy().tolist(),
            #         'source sentence': batch.src_raw,
            #         'translation': ' '.join(n_best_preds),
            #         'src': batch.src.data.cpu().numpy().tolist(),
            #     }
            # )
            f.write('\n'.join(n_best_preds))
            f.write('\n')
            f.flush()
        cur_batch += batch.batch_size
        sys.stdout.write('%d/%d       \r' % (cur_batch, iterators.test_iterator.nSentences))
        sys.stdout.flush()
        with open('./data/predictions_with_attention_wts.txt', 'w') as fp:
            json.dump(translations_with_attn, fp)

    f.close()

if __name__ == '__main__':
    # neptune.init('achaitanyasai/Machine-Translation-Test', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiIxMmU4NTM4Yy04OGVkLTQxZjktOTAzNy0wMWJlNTkwZGU4MWQifQ==')

    args = parse_arguments()
    disabled = False
    if args.expt_name == 'test' or args.expt_name == 'dev':
        disabled = True

    experiment = Experiment(api_key="G1Hu2kJ89qGE5sZ1cBNlrq6Hj",
                            project_name="english-hindi", workspace="achaitanyasai",
                            disabled=disabled)

    # experiment.log_asset(file_data='./models/seq2seq_attn.py', file_name='seq2seq_attn.py', overwrite=False)
    # experiment.log_asset(file_data='./modules/Trainer.py', file_name='trainer.py', overwrite=False)
    # experiment.log_asset(file_data='./data_iterator.py', file_name='data_iterator.py', overwrite=False)
    # experiment.add_tag('Expt-8')
    # experiment.add_tag('vector_space=2')
    # experiment.add_tag('UNK percentage=5')

    # experiment.add_tag('Bidirectional')
    experiment.add_tag('init: uniform')
    experiment.add_tag('%s' % args.model_type)
    experiment.add_tag('patience-10')
    experiment.set_name("%s" % args.expt_name)
    #experiment.add_tag('attn: bahdanau-additive')
    #experiment.add_tag('re-train with much better adagram vectors')
    #experiment.add_tag('Word level attn + word penalty')
    # experiment.add_tag('TEST')

    # Adding the Run parameters
    run_id = os.environ.get('RUN_ID')
    run_dir = os.environ.get('RUN_DIR')
    logger.info('Run Id: %s' % (run_id))
    logger.info('Run Dir: %s' % (run_dir))

    args.run_id = run_id
    args.run_dir = run_dir

    PARAMS = vars(args)
    # neptune.create_experiment(name='test', params=PARAMS)
    # neptune.append_tag('first-example-2')
    # neptune.send_artifact(args.path_to_logs)

    experiment.log_parameters(PARAMS)

    iterators = getIterators(args)

    model = buildModel(args, iterators.train_iterator)
    # model = loadModel(args)
    args.model = model
    args.iterators = iterators

    criterions = getCriterions(args)
    args.train_loss = criterions.train_criterion
    args.valid_loss = criterions.valid_criterion

    optimizer = getOptimizer(args)
    args.optimizer = optimizer

    if args.expt_name != 'test':
        train(args)

    saved_models = [
        './data/model_every_10steps.pt',
        './data/model_real.pt',
        './data/model.pt'
    ]
    max_bleu = 0
    for model_file in saved_models:
        args.model.load_state_dict(torch.load(model_file))
        args.model.eval()

        args.iterators.test_iterator.reset()
        translate(args, iterators)

        if args.expt_name != 'test':
            s = os.popen('perl %s/.guild/sourcecode/scripts/bleu-1.04.pl %s < ./data/predicted.txt' % (run_dir, args.testset_target)).read().strip()
        else:
            s = os.popen('perl ./scripts/bleu-1.04.pl %s < ./data/predicted.txt' % (args.testset_target)).read().strip()

        x = args.testset_target.strip('/')
        x = x.split('/')[-1]
        y = model_file.split('/')[-1]
        f = open('./data/bleu_%s_%s.txt' % (x, y), 'w')
        f.write(s)
        f.close()

        bleu = 0.0
        try:
            bleu = float(s.split(',')[0].replace('BLEU = ', ''))
            max_bleu = max(max_bleu, bleu)
        except Exception:
            pass

        logger.info('[%s] BLEU score: %.3f' % (y, bleu))
    logger.info('#' * 70)

    experiment.log_metric('Bleu score', max_bleu)
    experiment.log_asset('./data/training_logs.txt')
    experiment.log_asset_folder('%s/.guild/sourcecode/models' % (run_dir))
    experiment.log_asset_folder('%s/.guild/sourcecode/modules' % (run_dir))

    # Please explicitly delete the objects in which you have __del__ method implemented.
    del args
    iterators.__del__()
