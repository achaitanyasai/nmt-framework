import os
import unittest
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from modules import Trainer, Loss
from models import utils

import data_iterator
from models import seq2seq_attn_baseline, seq2seq_attn_multivec, seq2seq_attn_baseline_word_attn, seq2seq_attn_multivec_word_attn
torch.backends.cudnn.deterministic = True

random.seed(3435)
np.random.seed(3435)
torch.manual_seed(3435)
torch.cuda.manual_seed_all(3435)
torch.cuda.manual_seed(3435)


class Args(object):
    class Iterators(object):
        def __init__(self, train_iterator, valid_iterator, test_iterator):
            self.train_iterator = train_iterator
            self.valid_iterator = valid_iterator
            self.test_iterator = test_iterator

    def __init__(self, model, train_iterator, gradient_checks,
                 norm_method='sents', train_loss=None, valid_loss=None,
                 optimizer=None):
        self.model = model
        self.iterators = self.Iterators(train_iterator, None, None)
        self.gradient_checks = gradient_checks
        self.norm_method = norm_method
        self.train_loss = train_loss
        self.optimizer = optimizer
        self.valid_loss = valid_loss

class Testseq2seq_baseline(unittest.TestCase):
    def test_seq2seq_attn_copy(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_copy/train.csv', shuffle=True,
                                                    data_type='train',
                                                    src_max_len=50, tgt_max_len=50, src_max_vocab_size=100000,
                                                    tgt_max_vocab_size=1000000, ignore_too_many_unknowns=True,
                                                    break_on_stop_iteration=False)

        test_iterator = data_iterator.DataIterator(fields=(train_iterator.sourceField, train_iterator.targetField),
                                                   fname='./tests/toy_copy/valid.csv', shuffle=False,
                                                   data_type='valid')

        encoder = seq2seq_attn_baseline.encoder(rnn_type='LSTM', bidirectional=True, num_layers=2, hidden_size=200, vocab_size=10000, embedding_dim=200, pad_token=1, dropout=0.0)
        decoder = seq2seq_attn_baseline.decoder(rnn_type='LSTM', bidirectional_encoder=True, num_layers=2, hidden_size=200, vocab_size=train_iterator.targetField.nWords,
                                                embedding_dim=200, pad_token=train_iterator.targetField.word2idx['PAD'], attn_type='general', dropout=0.0)

        model = seq2seq_attn_baseline.Seq2SeqAttention(encoder=encoder, decoder=decoder).cuda()
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1)

        _loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        valid_loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        _optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        args = Args(model, train_iterator, gradient_checks=False,
                    norm_method='sents', train_loss=_loss, valid_loss=valid_loss,
                    optimizer=_optimizer)
        trainer = Trainer.Trainer(args)
        b_size = 8
        for steps in range(10000):
            data_iterator.logger.info('Next batch asking')
            batch = train_iterator.next_batch(b_size)
            batch.transpose()
            data_iterator.logger.info('Next batch got')
            data_iterator.logger.info('propagating')
            stats = trainer.propagate(batch, steps)
            data_iterator.logger.info('propagating done')
            data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
            if stats._loss() <= 1e-4:
                break
        self.assertLessEqual(stats._loss(), 1e-4)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

        test_iterator.reset()
        stats = trainer.validate(test_iterator, 16)
        data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
        self.assertLessEqual(stats._loss(), 1e-3)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

    def test_seq2seq_attn_reverse(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_reverse/train.csv', shuffle=True,
                                                    data_type='train',
                                                    src_max_len=50, tgt_max_len=50, src_max_vocab_size=100000,
                                                    tgt_max_vocab_size=1000000, ignore_too_many_unknowns=True,
                                                    break_on_stop_iteration=False)

        test_iterator = data_iterator.DataIterator(fields=(train_iterator.sourceField, train_iterator.targetField),
                                                   fname='./tests/toy_reverse/valid.csv', shuffle=False,
                                                   data_type='valid')

        encoder = seq2seq_attn_baseline.encoder(rnn_type='LSTM', bidirectional=True, num_layers=2, hidden_size=200, vocab_size=10000, embedding_dim=200, pad_token=1, dropout=0.0)
        decoder = seq2seq_attn_baseline.decoder(rnn_type='LSTM', bidirectional_encoder=True, num_layers=2, hidden_size=200, vocab_size=train_iterator.targetField.nWords,
                                                embedding_dim=200, pad_token=train_iterator.targetField.word2idx['PAD'], attn_type='general', dropout=0.0)

        model = seq2seq_attn_baseline.Seq2SeqAttention(encoder=encoder, decoder=decoder).cuda()
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1)

        _loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        valid_loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        _optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        args = Args(model, train_iterator, gradient_checks=False,
                    norm_method='sents', train_loss=_loss, valid_loss=valid_loss,
                    optimizer=_optimizer)
        trainer = Trainer.Trainer(args)
        b_size = 8
        for steps in range(10000):
            batch = train_iterator.next_batch(b_size)
            batch.transpose()
            stats = trainer.propagate(batch, steps)
            data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
            if stats._loss() <= 1e-4:
                break
        self.assertLessEqual(stats._loss(), 1e-4)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

        test_iterator.reset()
        stats = trainer.validate(test_iterator, 16)
        data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
        self.assertLessEqual(stats._loss(), 1e-3)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

        test_iterator.reset()
        stats = trainer.validate_fixed(test_iterator, 16)
        data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
        self.assertLessEqual(stats._loss(), 1e-3)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)


class Testseq2seq_baseline_wordattn(unittest.TestCase):
    def test_seq2seq_attn_copy(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_copy/train.csv', shuffle=True,
                                                    data_type='train',
                                                    src_max_len=50, tgt_max_len=50, src_max_vocab_size=100000,
                                                    tgt_max_vocab_size=1000000, ignore_too_many_unknowns=True,
                                                    break_on_stop_iteration=False)

        test_iterator = data_iterator.DataIterator(fields=(train_iterator.sourceField, train_iterator.targetField),
                                                   fname='./tests/toy_copy/valid.csv', shuffle=False,
                                                   data_type='valid')

        encoder = seq2seq_attn_baseline_word_attn.encoder(rnn_type='LSTM', bidirectional=True, num_layers=2, hidden_size=200, vocab_size=10000, embedding_dim=200, pad_token=1, dropout=0.0)
        decoder = seq2seq_attn_baseline_word_attn.decoder(rnn_type='LSTM', bidirectional_encoder=True, num_layers=2, hidden_size=200, vocab_size=train_iterator.targetField.nWords,
                                                embedding_dim=200, pad_token=train_iterator.targetField.word2idx['PAD'], attn_type='general', dropout=0.0)

        model = seq2seq_attn_baseline_word_attn.Seq2SeqAttention(encoder=encoder, decoder=decoder).cuda()
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1)

        _loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        valid_loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        _optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        args = Args(model, train_iterator, gradient_checks=False,
                    norm_method='sents', train_loss=_loss, valid_loss=valid_loss,
                    optimizer=_optimizer)
        trainer = Trainer.Trainer(args)
        b_size = 8
        for steps in range(10000):
            batch = train_iterator.next_batch(b_size)
            batch.transpose()
            stats = trainer.propagate(batch, steps)
            data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
            if stats._loss() <= 1e-4:
                break
        self.assertLessEqual(stats._loss(), 1e-4)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

        test_iterator.reset()
        stats = trainer.validate(test_iterator, 16)
        data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
        self.assertLessEqual(stats._loss(), 1e-3)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

        test_iterator.reset()
        stats = trainer.validate_fixed(test_iterator, 16)
        data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
        self.assertLessEqual(stats._loss(), 1e-3)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

    def test_seq2seq_attn_reverse(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_reverse/train.csv', shuffle=True,
                                                    data_type='train',
                                                    src_max_len=50, tgt_max_len=50, src_max_vocab_size=100000,
                                                    tgt_max_vocab_size=1000000, ignore_too_many_unknowns=True,
                                                    break_on_stop_iteration=False)

        test_iterator = data_iterator.DataIterator(fields=(train_iterator.sourceField, train_iterator.targetField),
                                                   fname='./tests/toy_reverse/valid.csv', shuffle=False,
                                                   data_type='valid')

        encoder = seq2seq_attn_baseline.encoder(rnn_type='LSTM', bidirectional=True, num_layers=2, hidden_size=200, vocab_size=10000, embedding_dim=200, pad_token=1, dropout=0.0)
        decoder = seq2seq_attn_baseline.decoder(rnn_type='LSTM', bidirectional_encoder=True, num_layers=2, hidden_size=200, vocab_size=train_iterator.targetField.nWords,
                                                embedding_dim=200, pad_token=train_iterator.targetField.word2idx['PAD'], attn_type='general', dropout=0.0)

        model = seq2seq_attn_baseline.Seq2SeqAttention(encoder=encoder, decoder=decoder).cuda()
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1)

        _loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        valid_loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        _optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        args = Args(model, train_iterator, gradient_checks=False,
                    norm_method='sents', train_loss=_loss, valid_loss=valid_loss,
                    optimizer=_optimizer)
        trainer = Trainer.Trainer(args)
        b_size = 8
        for steps in range(10000):
            batch = train_iterator.next_batch(b_size)
            batch.transpose()
            stats = trainer.propagate(batch, steps)
            data_iterator.logger.info(str(steps) + '/' + str(10000) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
            if stats._loss() <= 1e-4:
                break
        self.assertLessEqual(stats._loss(), 1e-4)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

        test_iterator.reset()
        stats = trainer.validate(test_iterator, 16)
        data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
        self.assertLessEqual(stats._loss(), 1e-3)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

        test_iterator.reset()
        stats = trainer.validate_fixed(test_iterator, 16)
        data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
        self.assertLessEqual(stats._loss(), 1e-3)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

class Testseq2seq_multivec(unittest.TestCase):
    def test_seq2seq_attn_copy(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_copy/train.csv', shuffle=True,
                                                    data_type='train',
                                                    src_max_len=50, tgt_max_len=50, src_max_vocab_size=100000,
                                                    tgt_max_vocab_size=1000000, ignore_too_many_unknowns=True,
                                                    break_on_stop_iteration=False)

        test_iterator = data_iterator.DataIterator(fields=(train_iterator.sourceField, train_iterator.targetField),
                                                   fname='./tests/toy_copy/valid.csv', shuffle=False,
                                                   data_type='valid')

        encoder = seq2seq_attn_multivec.encoder(rnn_type='LSTM', bidirectional=True, num_layers=2, hidden_size=200, vocab_size=10000, embedding_dim=200, pad_token=1, dropout=0.0)
        decoder = seq2seq_attn_multivec.decoder(rnn_type='LSTM', bidirectional_encoder=True, num_layers=2, hidden_size=200, vocab_size=train_iterator.targetField.nWords,
                                                embedding_dim=200, pad_token=train_iterator.targetField.word2idx['PAD'], attn_type='general', dropout=0.0)

        model = seq2seq_attn_multivec.Seq2SeqAttention(encoder=encoder, decoder=decoder).cuda()
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1)

        _loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        valid_loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        _optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        args = Args(model, train_iterator, gradient_checks=False,
                    norm_method='sents', train_loss=_loss, valid_loss=valid_loss,
                    optimizer=_optimizer)
        trainer = Trainer.Trainer(args)
        b_size = 8
        for steps in range(10000):
            batch = train_iterator.next_batch(b_size)
            batch.transpose()
            stats = trainer.propagate(batch, steps)
            data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
            if stats._loss() <= 1e-4:
                break
        self.assertLessEqual(stats._loss(), 1e-4)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

        test_iterator.reset()
        stats = trainer.validate(test_iterator, 16)
        data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
        self.assertLessEqual(stats._loss(), 1e-3)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

        test_iterator.reset()
        stats = trainer.validate_fixed(test_iterator, 16)
        data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
        self.assertLessEqual(stats._loss(), 1e-3)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

    def test_seq2seq_attn_reverse(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_reverse/train.csv', shuffle=True,
                                                    data_type='train',
                                                    src_max_len=50, tgt_max_len=50, src_max_vocab_size=100000,
                                                    tgt_max_vocab_size=1000000, ignore_too_many_unknowns=True,
                                                    break_on_stop_iteration=False)

        test_iterator = data_iterator.DataIterator(fields=(train_iterator.sourceField, train_iterator.targetField),
                                                   fname='./tests/toy_reverse/valid.csv', shuffle=False,
                                                   data_type='valid')

        encoder = seq2seq_attn_multivec.encoder(rnn_type='LSTM', bidirectional=True, num_layers=2, hidden_size=200, vocab_size=10000, embedding_dim=200, pad_token=1, dropout=0.0)
        decoder = seq2seq_attn_multivec.decoder(rnn_type='LSTM', bidirectional_encoder=True, num_layers=2, hidden_size=200, vocab_size=train_iterator.targetField.nWords,
                                                embedding_dim=200, pad_token=train_iterator.targetField.word2idx['PAD'], attn_type='general', dropout=0.0)

        model = seq2seq_attn_multivec.Seq2SeqAttention(encoder=encoder, decoder=decoder).cuda()
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1)

        _loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        valid_loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        _optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        args = Args(model, train_iterator, gradient_checks=False,
                    norm_method='sents', train_loss=_loss, valid_loss=valid_loss,
                    optimizer=_optimizer)
        trainer = Trainer.Trainer(args)
        b_size = 8
        for steps in range(10000):
            batch = train_iterator.next_batch(b_size)
            batch.transpose()
            stats = trainer.propagate(batch, steps)
            data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
            if stats._loss() <= 1e-4:
                break
        self.assertLessEqual(stats._loss(), 1e-4)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

        test_iterator.reset()
        stats = trainer.validate(test_iterator, 16)
        data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
        self.assertLessEqual(stats._loss(), 1e-3)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

        test_iterator.reset()
        stats = trainer.validate_fixed(test_iterator, 16)
        data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
        self.assertLessEqual(stats._loss(), 1e-3)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

class Testseq2seq_multivec_wordattn(unittest.TestCase):
    def test_seq2seq_attn_copy(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_copy/train.csv', shuffle=True,
                                                    data_type='train',
                                                    src_max_len=50, tgt_max_len=50, src_max_vocab_size=100000,
                                                    tgt_max_vocab_size=1000000, ignore_too_many_unknowns=True,
                                                    break_on_stop_iteration=False)

        test_iterator = data_iterator.DataIterator(fields=(train_iterator.sourceField, train_iterator.targetField),
                                                   fname='./tests/toy_copy/valid.csv', shuffle=False,
                                                   data_type='valid')

        encoder = seq2seq_attn_multivec_word_attn.encoder(rnn_type='LSTM', bidirectional=True, num_layers=2, hidden_size=500, vocab_size=10000, embedding_dim=500, pad_token=1, dropout=0.0)
        decoder = seq2seq_attn_multivec_word_attn.decoder(rnn_type='LSTM', bidirectional_encoder=True, num_layers=2, hidden_size=500, vocab_size=train_iterator.targetField.nWords,
                                                embedding_dim=500, pad_token=train_iterator.targetField.word2idx['PAD'], attn_type='general', dropout=0.0)

        model = seq2seq_attn_multivec_word_attn.Seq2SeqAttention(encoder=encoder, decoder=decoder).cuda()
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1)

        _loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        valid_loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        _optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        args = Args(model, train_iterator, gradient_checks=False,
                    norm_method='sents', train_loss=_loss, valid_loss=valid_loss,
                    optimizer=_optimizer)
        trainer = Trainer.Trainer(args)
        b_size = 64
        for steps in range(10000):
            batch = train_iterator.next_batch(b_size)
            batch.transpose()
            stats = trainer.propagate(batch, steps)
            data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
            if stats._loss() <= 1e-4:
                break
        self.assertLessEqual(stats._loss(), 1e-4)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

        test_iterator.reset()
        stats = trainer.validate(test_iterator, 16)
        data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
        self.assertLessEqual(stats._loss(), 1e-3)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

        test_iterator.reset()
        stats = trainer.validate_fixed(test_iterator, 16)
        data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
        self.assertLessEqual(stats._loss(), 1e-3)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

    def test_seq2seq_attn_reverse789(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_reverse/train.csv', shuffle=True,
                                                    data_type='train',
                                                    src_max_len=50, tgt_max_len=50, src_max_vocab_size=100000,
                                                    tgt_max_vocab_size=1000000, ignore_too_many_unknowns=True,
                                                    break_on_stop_iteration=False)

        test_iterator = data_iterator.DataIterator(fields=(train_iterator.sourceField, train_iterator.targetField),
                                                   fname='./tests/toy_reverse/valid.csv', shuffle=False,
                                                   data_type='valid')

        encoder = seq2seq_attn_multivec_word_attn.encoder(rnn_type='LSTM', bidirectional=True, num_layers=2, hidden_size=200, vocab_size=10000, embedding_dim=200, pad_token=1, dropout=0.0)
        decoder = seq2seq_attn_multivec_word_attn.decoder(rnn_type='LSTM', bidirectional_encoder=True, num_layers=2, hidden_size=200, vocab_size=train_iterator.targetField.nWords,
                                                embedding_dim=200, pad_token=train_iterator.targetField.word2idx['PAD'], attn_type='general', dropout=0.0)

        model = seq2seq_attn_multivec_word_attn.Seq2SeqAttention(encoder=encoder, decoder=decoder).cuda()
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1)

        _loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        valid_loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        _optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        args = Args(model, train_iterator, gradient_checks=False,
                    norm_method='sents', train_loss=_loss, valid_loss=valid_loss,
                    optimizer=_optimizer)
        trainer = Trainer.Trainer(args)
        b_size = 8
        for steps in range(10000):
            batch = train_iterator.next_batch(b_size)
            batch.transpose()
            stats = trainer.propagate(batch, steps)
            data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
            if stats._loss() <= 1e-4:
                break
        self.assertLessEqual(stats._loss(), 1e-4)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

        test_iterator.reset()
        stats = trainer.validate(test_iterator, 16)
        data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
        self.assertLessEqual(stats._loss(), 1e-3)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

        test_iterator.reset()
        stats = trainer.validate_fixed(test_iterator, 16)
        data_iterator.logger.info(str(steps) + ' ' + str(stats._loss()) + ' ' + str(stats.accuracy()))
        self.assertLessEqual(stats._loss(), 1e-3)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.embeddings = nn.Embedding(10, 3)
        self.sm = torch.nn.Softmax(dim=-1)

    def reverse(self, inputs, lengths, batch_first=False):
        # FIXME: fix requires_grad in reverse
        if batch_first:
            inputs = inputs.transpose(0, 1)
        if inputs.size(1) != len(lengths):
            raise ValueError('inputs incompatible with lengths.')
        reversed_inputs = inputs.clone().detach()  # .requires_grad_(False) # inputs.new_tensor()
        for i, length in enumerate(lengths):
            time_ind = torch.LongTensor(list(reversed(range(length))))
            reversed_inputs[:length, i] = inputs[:, i][time_ind]
        if batch_first:
            reversed_inputs = reversed_inputs.transpose(0, 1)
        return reversed_inputs

    def forward(self, a, b, lens, target):
        a = self.reverse(a, lens, batch_first=True).squeeze(0)
        a = self.embeddings(a).unsqueeze(0)
        b = self.embeddings(b).unsqueeze(2)
        zz = torch.zeros((1, 3))

        # target = self.embeddings(target)
        mask = utils.sequence_mask(lens, 3)
        mask = mask.unsqueeze(1)  # Make it broadcastable.
        mask = 1 - mask.type(torch.float32)
        mask = mask.type(torch.bool)
        w = torch.bmm(a, b).transpose(1, 2)  # .squeeze(2)
        w.data.masked_fill_(mask, -float('inf'))
        w = self.sm(w)
        a = torch.bmm(w, a).squeeze(1)
        for i in range(3):
            zz[0][i] = a[0][i]
        return zz, target

class Test_utility_functions(unittest.TestCase):

    def test_reverse(self):
        model = encoder()
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        target = torch.rand(1, 3)

        for it in range(100):
            lens = torch.LongTensor([1])
            a = torch.LongTensor([[1, 2, 3]])
            b = torch.LongTensor([4])

            a, _ = model(a, b, lens, None)
            loss = criterion(a, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            data_iterator.logger.info('Epoch: %d, Loss: %.20f' % (it + 1, loss.item()))

        self.assertAlmostEqual(0, loss.item(), delta=1e-9)
