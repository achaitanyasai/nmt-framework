import math
import random
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import data_iterator_optimized
import nmt
from models import seq2seq_attn, utils
from modules import Loss, Optimizer, Trainer, utils, Translator, Beam
import sys

class argsWrapper(object):
    def __init__(self):
        self.optimizer = 'sgd'
        self.lrate = 1
        self.max_grad_norm = 5.0
        self.lrate_decay = 0.5
        self.start_decay_at = 50
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adagrad_accumulator_init = 0
        self.decay_method = ""
        self.warmup_steps = 4000
        self.dim = 10
        self.dispFreq = 1000000
        self.norm_method = 'sentences'

class testTranslator(unittest.TestCase):
    def setUp(self):
        self.src_lang = data_iterator_optimized.Lang('./tests/test_data.src', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='source', chars=False, verbose=False, ignore_too_many_unknowns=True)
        self.tgt_lang = data_iterator_optimized.Lang('./tests/test_data.tgt', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='target', chars=False, verbose=False, ignore_too_many_unknowns=True)

        self.src_lang_val = data_iterator_optimized.Lang('./tests/test_data.val.src', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='source', chars=False, verbose=False, ignore_too_many_unknowns=True)
        self.tgt_lang_val = data_iterator_optimized.Lang('./tests/test_data.val.tgt', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='target', chars=False, verbose=False, ignore_too_many_unknowns=True)

        self.iterator = data_iterator_optimized.dataIterator(self.src_lang, self.tgt_lang, shuffle=False)
        self.val_iterator = data_iterator_optimized.dataIterator(self.src_lang_val, self.tgt_lang_val, shuffle=False)
        
        self.encoder = seq2seq_attn.CustomEncoder('LSTM', True, 2, 10, self.src_lang.n_words, 10, self.src_lang.word2idx['PAD'], self.src_lang.n_hashes, 0.0).cuda()        
        self.decoder = seq2seq_attn.decoder('LSTM', True, 2, 10, self.tgt_lang.n_words, self.tgt_lang.n_hashes, 10, self.tgt_lang.word2idx['PAD'], 'general', dropout=0.0).cuda()
        self.model = seq2seq_attn.Seq2SeqAttention(self.encoder, self.decoder)
        
        self.args = argsWrapper()
        
        self.optimizer = nmt.get_optimizer(self.args, self.model, verbose=False)
        self.train_loss = nmt.get_criterion(self.args, self.tgt_lang)
        self.valid_loss = nmt.get_criterion(self.args, self.tgt_lang)

        self.Trainer = Trainer.Trainer(
            self.model, self.train_loss, self.valid_loss, self.optimizer, norm_method='tokens'
        )

        self.Translator = Translator.Translator(
            self.model, self.src_lang, self.tgt_lang, None,
            None, n_best=1, max_length=50, global_scorer=None,
            copy_attn=False, cuda=True, beam_trace=False, min_length=0
        )

        self.Translator1 = Translator.Translator(
            self.model, self.tgt_lang, self.src_lang, None,
            None, n_best=1, max_length=50, global_scorer=None,
            copy_attn=False, cuda=True, beam_trace=False, min_length=0
        )

        self.scorer = Beam.GNMTGlobalScorer(0.0, -0.0)
        
        self.translator = Translator.Translator(
                            self.model, self.src_lang, self.tgt_lang, self.iterator, beam_size=5,
                            n_best=1, global_scorer=self.scorer, max_length=50, copy_attn=False,
                            cuda=True, beam_trace=False, min_length=0)
        
        self.builder = Translator.TranslationBuilder(self.src_lang, self.tgt_lang, self.iterator, 1, True)

    def test_translate(self):
        for i in range(100):
            self.iterator.reset()
            self.args.batch_size = 2
            train_stats = self.Trainer.train(self.args, self.iterator, self.src_lang, self.tgt_lang, i)
        # self.assertGreaterEqual(train_stats.accuracy(), 98.0) #98 is lower limit. It should be 100.0.
        self.assertEqual(train_stats.accuracy(), 100.0)
        pred_score_total, pred_words_total = 0, 0
        cur_batch = 0

        self.iterator.reset()

        
        target_preds = ['abcd', 'pbbbd f']
        while cur_batch < self.iterator.n_samples:
            batch = self.iterator.next_batch(1, 50, source_language_model = self.src_lang, target_language_model = self.tgt_lang)
            batch_data = self.translator.translate_batch(batch)
            
            translations = self.builder.from_batch(batch_data, batch)
            for trans in translations:
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                n_best_preds = [' '.join(pred) for pred in trans.pred_sents[: 1]]
            correct = target_preds[cur_batch]
            self.assertEqual(n_best_preds[0], correct)
            cur_batch += batch.batch_size

class testTranslator_ToyCopy(unittest.TestCase):
    def setUp(self):
        self.src_lang = data_iterator_optimized.Lang('./tests/toy_copy/train.src', 20, max_word_len_allowed=20, max_vocab_size=10000, langtype='source', chars=False, verbose=False, ignore_too_many_unknowns=True)
        self.tgt_lang = data_iterator_optimized.Lang('./tests/toy_copy/train.tgt', 20, max_word_len_allowed=20, max_vocab_size=10000, langtype='target', chars=False, verbose=False, ignore_too_many_unknowns=True)

        self.src_lang_val = data_iterator_optimized.Lang('./tests/toy_copy/valid.src', 20, max_word_len_allowed=20, max_vocab_size=10000, langtype='source', chars=False, verbose=False, ignore_too_many_unknowns=True)
        self.tgt_lang_val = data_iterator_optimized.Lang('./tests/toy_copy/valid.tgt', 20, max_word_len_allowed=20, max_vocab_size=10000, langtype='target', chars=False, verbose=False, ignore_too_many_unknowns=True)

        self.iterator = data_iterator_optimized.dataIterator(self.src_lang, self.tgt_lang, shuffle=False)
        self.val_iterator = data_iterator_optimized.dataIterator(self.src_lang_val, self.tgt_lang_val, shuffle=False)
        
        b = self.iterator.next_batch(40, 33)
        
        self.encoder = seq2seq_attn.CustomEncoder('LSTM', True, 2, 50, self.src_lang.n_words, 50, self.src_lang.word2idx['PAD'], self.src_lang.n_hashes, 0.2).cuda()        
        self.decoder = seq2seq_attn.decoder('LSTM', True, 2, 50, self.tgt_lang.n_words, self.tgt_lang.n_hashes, 50, self.tgt_lang.word2idx['PAD'], 'general', dropout=0.2).cuda()
        self.model = seq2seq_attn.Seq2SeqAttention(self.encoder, self.decoder)
        
        self.args = argsWrapper()
        
        self.optimizer = nmt.get_optimizer(self.args, self.model, verbose=False)
        self.train_loss = nmt.get_criterion(self.args, self.tgt_lang)
        self.valid_loss = nmt.get_criterion(self.args, self.tgt_lang)

        self.Trainer = Trainer.Trainer(
            self.model, self.train_loss, self.valid_loss, self.optimizer, norm_method='tokens'
        )

        self.Translator = Translator.Translator(
            self.model, self.src_lang, self.tgt_lang, None,
            None, n_best=1, max_length=50, global_scorer=None,
            copy_attn=False, cuda=True, beam_trace=False, min_length=0
        )

        self.Translator1 = Translator.Translator(
            self.model, self.tgt_lang, self.src_lang, None,
            None, n_best=1, max_length=50, global_scorer=None,
            copy_attn=False, cuda=True, beam_trace=False, min_length=0
        )

        self.scorer = Beam.GNMTGlobalScorer(0.0, -0.0)
        
        self.translator = Translator.Translator(
                            self.model, self.src_lang, self.tgt_lang, self.iterator, beam_size=5,
                            n_best=1, global_scorer=self.scorer, max_length=50, copy_attn=False,
                            cuda=True, beam_trace=False, min_length=0)
        
        self.builder = Translator.TranslationBuilder(self.src_lang, self.tgt_lang, self.iterator, 1, True)

    def test_translate(self):
        print
        for i in range(10):
            self.iterator.reset()
            self.args.batch_size = 40
            train_stats = self.Trainer.train(self.args, self.iterator, self.src_lang, self.tgt_lang, i)
            msg = ('=' * 80) + '\nEpoch: %d, Train Loss: %.3f, Train Accuracy: %.3f, Train Perplexity: %.3f'
            msg = msg % (i + 1, train_stats._loss(), train_stats.accuracy(), train_stats.perplexity())
            print(msg)
        self.assertGreaterEqual(train_stats.accuracy(), 99.0)        
        # self.assertLessEqual(train_stats._loss(), 0.005)
        
        pred_score_total, pred_words_total = 0, 0
        cur_batch = 0

        self.val_iterator.reset()
        total = 0
        correct = 0
        while cur_batch < self.val_iterator.n_samples:
            batch = self.val_iterator.next_batch(1, 50, source_language_model = self.src_lang, target_language_model = self.tgt_lang)
            batch_data = self.translator.translate_batch(batch)
            
            translations = self.builder.from_batch(batch_data, batch)
            for trans in translations:
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                n_best_preds = [' '.join(pred) for pred in trans.pred_sents[: 1]]
                total += 1
                if n_best_preds[0] == ' '.join(batch.tgt_raw[0]):
                    correct += 1
            cur_batch += batch.batch_size
        self.assertGreaterEqual(float(correct) / total, 0.95)


class testTranslator_ToyReverse(unittest.TestCase):
    '''
    For some reason, the model is not convering on reverse dataset.
    Need to think.

    UPD: It's working :) The reason why it didn't work is that,
    if length of sentence is more than 10, I am truncating. Due to that, words that appear at
    the end of the sentence are missing leading to not so trivial data.
    Now the max length was increased to 50. In addition to that, the embedding and hidden size was 
    also increased to increase the power of the model.
    '''
    def setUp(self):
        self.src_lang = data_iterator_optimized.Lang('./tests/toy_reverse/train.src', 50, max_word_len_allowed=20, max_vocab_size=10000, langtype='source', chars=False, verbose=False, ignore_too_many_unknowns=True, max_number_of_sentences_allowed=1000000)
        self.tgt_lang = data_iterator_optimized.Lang('./tests/toy_reverse/train.tgt', 50, max_word_len_allowed=20, max_vocab_size=10000, langtype='target', chars=False, verbose=False, ignore_too_many_unknowns=True, max_number_of_sentences_allowed=1000000)

        self.src_lang_val = data_iterator_optimized.Lang('./tests/toy_reverse/valid.src', 50, max_word_len_allowed=20, max_vocab_size=10000, langtype='source', chars=False, verbose=False, ignore_too_many_unknowns=True)
        self.tgt_lang_val = data_iterator_optimized.Lang('./tests/toy_reverse/valid.tgt', 50, max_word_len_allowed=20, max_vocab_size=10000, langtype='target', chars=False, verbose=False, ignore_too_many_unknowns=True)

        self.iterator = data_iterator_optimized.dataIterator(self.src_lang, self.tgt_lang, shuffle=False)
        self.val_iterator = data_iterator_optimized.dataIterator(self.src_lang_val, self.tgt_lang_val, shuffle=False)
        
        self.encoder = seq2seq_attn.CustomEncoder('LSTM', True, 2, 50, self.src_lang.n_words, 50, self.src_lang.word2idx['PAD'], self.src_lang.n_hashes, 0.0).cuda()        
        self.decoder = seq2seq_attn.decoder('LSTM', True, 2, 50, self.tgt_lang.n_words, self.tgt_lang.n_hashes, 50, self.tgt_lang.word2idx['PAD'], 'general', dropout=0.0).cuda()
        self.model = seq2seq_attn.Seq2SeqAttention(self.encoder, self.decoder)
        
        self.args = argsWrapper()
        
        self.optimizer = nmt.get_optimizer(self.args, self.model, verbose=False)
        self.train_loss = nmt.get_criterion(self.args, self.tgt_lang)
        self.valid_loss = nmt.get_criterion(self.args, self.tgt_lang)

        self.Trainer = Trainer.Trainer(
            self.model, self.train_loss, self.valid_loss, self.optimizer, norm_method='tokens'
        )

        self.Translator = Translator.Translator(
            self.model, self.src_lang, self.tgt_lang, None,
            None, n_best=1, max_length=50, global_scorer=None,
            copy_attn=False, cuda=True, beam_trace=False, min_length=0
        )

        self.Translator1 = Translator.Translator(
            self.model, self.tgt_lang, self.src_lang, None,
            None, n_best=1, max_length=50, global_scorer=None,
            copy_attn=False, cuda=True, beam_trace=False, min_length=0
        )

        self.scorer = Beam.GNMTGlobalScorer(0.0, -0.0)
        
        self.translator = Translator.Translator(
                            self.model, self.src_lang, self.tgt_lang, self.iterator, beam_size=5,
                            n_best=1, global_scorer=self.scorer, max_length=50, copy_attn=False,
                            cuda=True, beam_trace=False, min_length=0)
        
        self.builder = Translator.TranslationBuilder(self.src_lang, self.tgt_lang, self.iterator, 1, True)

    def test_translate(self):
        print
        for i in range(10):
            self.iterator.reset()
            self.args.batch_size = 40
            train_stats = self.Trainer.train(self.args, self.iterator, self.src_lang, self.tgt_lang, i)
            msg = ('=' * 80) + '\nEpoch: %d, Train Loss: %.3f, Train Accuracy: %.3f, Train Perplexity: %.3f'
            msg = msg % (i + 1, train_stats._loss(), train_stats.accuracy(), train_stats.perplexity())
            print(msg)
        self.assertGreaterEqual(train_stats.accuracy(), 99.0)        
        # self.assertLessEqual(train_stats._loss(), 0.005)
        
        pred_score_total, pred_words_total = 0, 0
        cur_batch = 0

        self.val_iterator.reset()
        total = 0
        correct = 0
        while cur_batch < self.val_iterator.n_samples:
            batch = self.val_iterator.next_batch(1, 50, source_language_model = self.src_lang, target_language_model = self.tgt_lang)
            batch_data = self.translator.translate_batch(batch)
            
            translations = self.builder.from_batch(batch_data, batch)
            for trans in translations:
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                n_best_preds = [' '.join(pred) for pred in trans.pred_sents[: 1]]
                total += 1
                if n_best_preds[0] == ' '.join(batch.tgt_raw[0]):
                    correct += 1
            cur_batch += batch.batch_size
        self.assertGreaterEqual(float(correct) / total, 0.95)