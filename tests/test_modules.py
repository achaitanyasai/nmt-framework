import math
import random
import unittest

import nmt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import data_iterator_optimized
from models import seq2seq_attn, utils
from modules import Loss, Optimizer, Trainer, utils, Translator


class argsWrapper(object):
    def __init__(self):
        self.optimizer = 'sgd'
        self.lrate = 1.0
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

class testTrainer(unittest.TestCase):
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

    def test_train(self):
        for i in range(100):
            self.iterator.reset()
            self.args.batch_size = 2
            train_stats = self.Trainer.train(self.args, self.iterator, self.src_lang, self.tgt_lang, i)
        # self.assertGreaterEqual(train_stats.accuracy(), 98.0) #98 is lower limit. It should be 100.0.
        self.assertEqual(train_stats.accuracy(), 100.0)

    def test_validate(self):
        n = random.randint(1, 5)
        for i in range(n):
            self.iterator.reset()
            self.args.batch_size = 2
            self.Trainer.train(self.args, self.iterator, self.src_lang, self.tgt_lang, i)

        self.val_iterator.reset()
        valid_stats1 = self.Trainer.validate(self.args, self.val_iterator, self.src_lang, self.tgt_lang, 1)
        self.val_iterator.reset()
        valid_stats2 = self.Trainer.validate(self.args, self.val_iterator, self.src_lang, self.tgt_lang, 2)
        self.val_iterator.reset()
        valid_stats3 = self.Trainer.validate(self.args, self.val_iterator, self.src_lang, self.tgt_lang, 5)
        
        self.assertEqual(valid_stats1.accuracy(), valid_stats2.accuracy())
        self.assertEqual(round(valid_stats1._loss(), 5), round(valid_stats2._loss(), 5))
        self.assertEqual(valid_stats3.accuracy(), valid_stats2.accuracy())
        self.assertEqual(round(valid_stats3._loss(), 5), round(valid_stats2._loss(), 5))
    
    def test_validate_fixed(self):
        n = random.randint(1, 5)
        for i in range(n):
            self.iterator.reset()
            self.args.batch_size = 2
            self.Trainer.train(self.args, self.iterator, self.src_lang, self.tgt_lang, i)

        self.val_iterator.reset()
        valid_stats_old = self.Trainer.validate(self.args, self.val_iterator, self.src_lang, self.tgt_lang, 5)

        self.val_iterator.reset()
        valid_stats1 = self.Trainer.validate_fixed(self.args, self.val_iterator, self.src_lang, self.tgt_lang, 5)
        #FIXME: uncomment below line
        # self.assertEqual(round(valid_stats1._loss(), 5), round(valid_stats_old._loss(), 5))

    
    def test_loss_and_accuracy_with_pad(self):
        _loss = nmt.get_criterion(self.args, self.tgt_lang)
        src = Variable(torch.FloatTensor(
            [
                [2, 3],
                [3, 1],
                [4, 1]
            ]
        )).cuda()
        tgt = Variable(torch.LongTensor(
            [
                [2, 2],
                [4, 3],
                [3, 1],
            ]
        )).cuda()
        predicted = Variable(torch.FloatTensor(
            [
                [
                    [0.15, 0.238, 0.026, 0.290, 0.322, 0.085, 0.021], 
                    [0.006, 0.15, 0.090, 0.622, 0.026, 0.085, 0.021],
                ],
                [
                    [0.006, 0.15, 0.090, 0.622, 0.026, 0.085, 0.021], 
                    [0.001, 0.002, 15.00, 0.001, 0.789, 0.4, 0.0001],
                ]
            ]
        )).cuda()

        predicted = F.log_softmax(predicted, dim=2)
        
        batch = data_iterator_optimized.BatchData(src, tgt, None, None, None, None, None, None)
        loss, stats = _loss.compute_loss(batch, None, predicted, None, None, None, None)
        
        self.assertEqual(stats.accuracy(), 100.0)
        self.assertEqual(stats.n_words, 3)
        self.assertEqual(stats._loss() * 3, loss.item())
        required = predicted[0][0][4] + predicted[0][1][3] + predicted[1][0][3]
        self.assertEqual(-required.item(), loss.item())
    
    def test_loss_and_accuracy_without_pad(self):
        _loss = nmt.get_criterion(self.args, self.tgt_lang)
        src = Variable(torch.FloatTensor(
            [
                [2, 3],
                [3, 1],
                [4, 1]
            ]
        )).cuda()
        tgt = Variable(torch.LongTensor(
            [
                [2, 2],
                [4, 3],
                [3, 5],
            ]
        )).cuda()
        predicted = Variable(torch.FloatTensor(
            [
                [
                    [0.15, 0.238, 0.026, 0.290, 0.322, 0.085, 0.021], 
                    [0.006, 0.15, 0.090, 0.622, 0.026, 0.085, 0.021],
                ],
                [
                    [0.006, 0.15, 0.090, 0.622, 0.026, 0.085, 0.021], 
                    [0.001, 0.002, 15.00, 0.001, 0.789, 0.4, 0.0001],
                ]
            ]
        )).cuda()

        predicted = F.log_softmax(predicted, dim=2)
        
        batch = data_iterator_optimized.BatchData(src, tgt, None, None, None, None, None, None)
        loss, stats = _loss.compute_loss(batch, None, predicted, None, None, None, None)
        
        self.assertEqual(stats.accuracy(), 75.0)
        self.assertEqual(stats.n_words, 4)
        self.assertEqual(stats._loss() * 4, loss.item())
        required = predicted[0][0][4] + predicted[0][1][3] + predicted[1][0][3] + predicted[1][1][5]
        self.assertEqual(-required.item(), loss.item())
    
    # def test_get_hashes(self):
    #     self.iterator.reset()
    #     batch = self.iterator.next_batch(2, None, self.src_lang, self.tgt_lang)
    #     cur_src = batch.src.transpose(0, 1)
    #     res = []
    #     res1 = []
    #     for i in cur_src:
    #         w = i.unsqueeze(0)
    #         ret = utils.get_hashes(w, self.src_lang).transpose(0, 1)
    #         ret1 = self.Translator1.get_char_features(w).transpose(0, 1)
    #         res.append(ret)
    #         res1.append(ret1)
    #     res = torch.cat(res, 0)
    #     res1 = torch.cat(res1, 0)

    #     self.assertTrue((res1.data.cpu().numpy() == batch.src_hashes.data.cpu().numpy()).all())

    #     self.assertEqual(res.shape, batch.src_hashes.shape)
    #     self.assertTrue((res.data.cpu().numpy() == batch.src_hashes.data.cpu().numpy()).all())

    #     cur_tgt = batch.tgt.transpose(0, 1)
    #     res = []
    #     res1 = []
    #     for i in cur_tgt:
    #         w = i.unsqueeze(0)
    #         ret = utils.get_hashes(w, self.tgt_lang).transpose(0, 1)
    #         ret1 = self.Translator.get_char_features(w).transpose(0, 1)
    #         res.append(ret)
    #         res1.append(ret1)
    #     res = torch.cat(res, 0)
    #     res1 = torch.cat(res1, 0)

    #     self.assertTrue((res1.data.cpu().numpy() == batch.tgt_hashes.data.cpu().numpy()).all())

    #     self.assertEqual(res.shape, batch.tgt_hashes.shape)
    #     self.assertTrue((res.data.cpu().numpy() == batch.tgt_hashes.data.cpu().numpy()).all())


class testTranslator(unittest.TestCase):
    def setUp(self):
        pass
