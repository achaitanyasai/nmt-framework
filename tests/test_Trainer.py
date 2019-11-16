import os
import unittest
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from modules import Trainer, Loss

import data_iterator
from models import toy_seq2seq, toy_seq2seq_same_encoder_decoder, seq2seq_attn_baseline
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

class TestTrainer(unittest.TestCase):
    def test_propogate_overfit(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_copy/train.csv', shuffle=True,
                                                    data_type='train',
                                                    src_max_len=5, tgt_max_len=5, src_max_vocab_size=100000,
                                                    tgt_max_vocab_size=1000000, ignore_too_many_unknowns=True,
                                                    break_on_stop_iteration=False)

        encoder = toy_seq2seq.Encoder(bidirectional=True, num_layers=2, hidden_size=100, vocab_size=10000, embedding_dim=100, pad_token=1, dropout=0.0)
        decoder = toy_seq2seq.Decoder(hidden_size=100, output_size=train_iterator.targetField.nWords, num_layers=2, dropout=0.0)
        model = toy_seq2seq.Model(encoder=encoder, decoder=decoder).cuda()
        _loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        _optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        args = Args(model, train_iterator, gradient_checks=False,
                    norm_method='sents', train_loss=_loss,
                    optimizer=_optimizer)
        trainer = Trainer.Trainer(args)
        b_size = 16
        batch = train_iterator.next_batch(b_size)
        batch.transpose()
        for steps in range(1000):
            stats = trainer.propagate(batch, steps)
        self.assertLessEqual(stats._loss(), 1e-4)
        self.assertGreaterEqual(stats.accuracy(), 100 - 1e-4)

    def test_propogate_with_shuffle(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_copy/train.csv', shuffle=True,
                                                    data_type='train',
                                                    src_max_len=10, tgt_max_len=10, src_max_vocab_size=100000,
                                                    tgt_max_vocab_size=1000000, ignore_too_many_unknowns=True,
                                                    break_on_stop_iteration=False)

        encoder = toy_seq2seq.Encoder(bidirectional=True, num_layers=2, hidden_size=100, vocab_size=10000, embedding_dim=100, pad_token=1, dropout=0.0)
        decoder = toy_seq2seq.Decoder(hidden_size=100, output_size=train_iterator.targetField.nWords, num_layers=2, dropout=0.0)
        model = toy_seq2seq.Model(encoder=encoder, decoder=decoder).cuda()
        _loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        _optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        args = Args(model, train_iterator, gradient_checks=False,
                    norm_method='sents', train_loss=_loss,
                    optimizer=_optimizer)
        trainer = Trainer.Trainer(args)
        b_size = 64
        for steps in range(5000):
            batch = train_iterator.next_batch(b_size)
            batch.transpose()
            stats = trainer.propagate(batch, steps)
        self.assertLessEqual(stats._loss(), 0.01)
        self.assertGreaterEqual(stats.accuracy(), 99.0)

    def test_propogate_without_shuffle(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_copy/train.csv', shuffle=False,
                                                    data_type='train',
                                                    src_max_len=10, tgt_max_len=10, src_max_vocab_size=100000,
                                                    tgt_max_vocab_size=1000000, ignore_too_many_unknowns=True,
                                                    break_on_stop_iteration=False)

        encoder = toy_seq2seq.Encoder(bidirectional=True, num_layers=2, hidden_size=100, vocab_size=10000, embedding_dim=100, pad_token=1, dropout=0.0)
        decoder = toy_seq2seq.Decoder(hidden_size=100, output_size=train_iterator.targetField.nWords, num_layers=2, dropout=0.0)
        model = toy_seq2seq.Model(encoder=encoder, decoder=decoder).cuda()
        _loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        _optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        args = Args(model, train_iterator, gradient_checks=False,
                    norm_method='sents', train_loss=_loss,
                    optimizer=_optimizer)
        trainer = Trainer.Trainer(args)
        b_size = 64
        for steps in range(5000):
            batch = train_iterator.next_batch(b_size)
            batch.transpose()
            stats = trainer.propagate(batch, steps)
        self.assertLessEqual(stats._loss(), 0.01)
        self.assertGreaterEqual(stats.accuracy(), 99.0)

    def test_propogate_with_shuffle_random_batch(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_copy/train.csv', shuffle=True,
                                                    data_type='train',
                                                    src_max_len=10, tgt_max_len=10, src_max_vocab_size=100000,
                                                    tgt_max_vocab_size=1000000, ignore_too_many_unknowns=True,
                                                    break_on_stop_iteration=False)

        encoder = toy_seq2seq.Encoder(bidirectional=True, num_layers=2, hidden_size=100, vocab_size=10000, embedding_dim=100, pad_token=1, dropout=0.0)
        decoder = toy_seq2seq.Decoder(hidden_size=100, output_size=train_iterator.targetField.nWords, num_layers=2, dropout=0.0)
        model = toy_seq2seq.Model(encoder=encoder, decoder=decoder).cuda()
        _loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        _optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        args = Args(model, train_iterator, gradient_checks=False,
                    norm_method='sents', train_loss=_loss,
                    optimizer=_optimizer)
        trainer = Trainer.Trainer(args)
        b_size = random.randint(5, 64)
        for steps in range(5000):
            batch = train_iterator.next_batch(b_size)
            batch.transpose()
            stats = trainer.propagate(batch, steps)
        self.assertLessEqual(stats._loss(), 0.01)
        self.assertGreaterEqual(stats.accuracy(), 99.0)

class TestTrainer1(unittest.TestCase):
    def test_validate(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_copy/train.csv', shuffle=True,
                                                    data_type='train',
                                                    src_max_len=30, tgt_max_len=30, src_max_vocab_size=100000,
                                                    tgt_max_vocab_size=1000000, ignore_too_many_unknowns=True,
                                                    break_on_stop_iteration=False, tie_embeddings=True)

        valid_iterator = data_iterator.DataIterator(fields=(train_iterator.sourceField, train_iterator.targetField),
                                                    fname='./tests/toy_copy/valid.csv', shuffle=False,
                                                    data_type='valid',
                                                    src_max_len=None, tgt_max_len=None, src_max_vocab_size=None,
                                                    tgt_max_vocab_size=None, ignore_too_many_unknowns=None,
                                                    break_on_stop_iteration=True)

        encoder = toy_seq2seq_same_encoder_decoder.Encoder(bidirectional=True, num_layers=2, hidden_size=150, embedding_dim=150, pad_token=train_iterator.sourceField.word2idx['PAD'], dropout=0.0)
        decoder = toy_seq2seq_same_encoder_decoder.Decoder(hidden_size=150, output_size=train_iterator.sourceField.nWords, num_layers=2, dropout=0.0)

        model = toy_seq2seq_same_encoder_decoder.Model(encoder=encoder, decoder=decoder, vocab_size=train_iterator.sourceField.nWords, embedding_dim=150, pad_token=encoder.pad_token).cuda()

        train_loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        valid_loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                                  reduction='sum', perform_dimension_checks=True).cuda()
        _optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        _optimizer.param_groups[0]['lr'] = 0.001

        args = Args(model, train_iterator, gradient_checks=False,
                    norm_method='sents', train_loss=train_loss,
                    valid_loss=valid_loss, optimizer=_optimizer)
        trainer = Trainer.Trainer(args)
        b_size = 128
        for steps in range(10000):
            batch = train_iterator.next_batch(b_size)
            batch.transpose()
            stats = trainer.propagate(batch, steps)
        self.assertLessEqual(stats._loss(), 0.02)
        self.assertGreaterEqual(stats.accuracy(), 99.0)
        valid_stats = trainer.validate(valid_iterator, b_size=20)
        self.assertGreaterEqual(valid_stats.accuracy(), 90.0)
        # self.assertLessEqual(stats._loss(), 0.5)

class TestTrainerHindiToEnglish(unittest.TestCase):
    def test_validate(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='/home/chaitanya/Datasets/hindi-english/train.hn-en.csv', shuffle=True,
                                                    data_type='train',
                                                    src_max_len=50, tgt_max_len=50, src_max_vocab_size=36500,
                                                    tgt_max_vocab_size=27500, ignore_too_many_unknowns=False,
                                                    break_on_stop_iteration=False, tie_embeddings=False)

        valid_iterator = data_iterator.DataIterator(fields=(train_iterator.sourceField, train_iterator.targetField),
                                                    fname='/home/chaitanya/Datasets/hindi-english/valid.hn-en.csv', shuffle=False,
                                                    data_type='valid',
                                                    src_max_len=None, tgt_max_len=None, src_max_vocab_size=None,
                                                    tgt_max_vocab_size=None, ignore_too_many_unknowns=None,
                                                    break_on_stop_iteration=True)

        # encoder = toy_seq2seq.Encoder(bidirectional=True, num_layers=2, hidden_size=500, vocab_size=train_iterator.sourceField.nWords,
        #                               embedding_dim=500, pad_token=train_iterator.sourceField.word2idx['PAD'], dropout=0.4)
        # decoder = toy_seq2seq.Decoder(hidden_size=500, output_size=train_iterator.targetField.nWords, num_layers=2,
        #                               dropout=0.4)
        # model = toy_seq2seq.Model(encoder=encoder, decoder=decoder).cuda()

        encoder = seq2seq_attn_baseline.encoder('LSTM', bidirectional=True,num_layers=2,hidden_size=500, vocab_size=train_iterator.sourceField.nWords, embedding_dim=500,
                                                pad_token=train_iterator.sourceField.word2idx['PAD'], dropout=0.4)
        decoder = seq2seq_attn_baseline.decoder('LSTM', bidirectional_encoder=True, num_layers=2, hidden_size=500, vocab_size=train_iterator.targetField.nWords,
                                                embedding_dim=500, pad_token=train_iterator.targetField.word2idx['PAD'], attn_type='general', dropout=0.4)
        model = seq2seq_attn_baseline.Seq2SeqAttention(encoder=encoder, decoder=decoder).cuda()
        print('Model built')

        train_loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=train_iterator.targetField.word2idx['PAD'],
                             reduction='sum', perform_dimension_checks=False).cuda()
        valid_loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=train_iterator.targetField.word2idx['PAD'],
                                  reduction='sum', perform_dimension_checks=False).cuda()


        _optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        args = Args(model, train_iterator, gradient_checks=False,
                    norm_method='sents', train_loss=train_loss,
                    valid_loss=valid_loss, optimizer=_optimizer)
        trainer = Trainer.Trainer(args)
        b_size = 80

        # Remove return if you want to train and test on real world example.
        # Note that this will take several hours to complete.
        return

        for steps in range(50000):
            batch = train_iterator.next_batch(b_size)
            batch.transpose()
            stats = trainer.propagate(batch, steps)
            if steps % 10 == 0:
                print('Step: %d, train_loss: %.6f, train_accuracy: %.6f' % (steps, stats._loss(), stats.accuracy()))
            if steps % 100 == 0:
                valid_iterator.reset()
                valid_stats = trainer.validate(valid_iterator, b_size=20)
                print('='*20 + '\nStep: %d, valid_loss: %.6f, valid_accuracy: %.6f\n' % (steps, valid_stats._loss(), valid_stats.accuracy()) + '='*20)
        self.assertLessEqual(stats._loss(), 0.02)
        self.assertGreaterEqual(stats.accuracy(), 99.0)
        valid_stats = trainer.validate(valid_iterator, b_size=20)
        self.assertGreaterEqual(valid_stats.accuracy(), 90.0)
        # self.assertLessEqual(stats._loss(), 0.5)

class TestTrainer2(unittest.TestCase):
    def test_validate_different_batch_sizes(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_copy/train.csv', shuffle=True,
                                                    data_type='train',
                                                    src_max_len=30, tgt_max_len=30, src_max_vocab_size=100000,
                                                    tgt_max_vocab_size=1000000, ignore_too_many_unknowns=True,
                                                    break_on_stop_iteration=False, tie_embeddings=True)
        os.system('cp ./tests/toy_copy/valid.csv /tmp/valid.csv')

        valid_iterator = data_iterator.DataIterator(fields=(train_iterator.sourceField, train_iterator.targetField),
                                                    fname='/tmp/valid.csv', shuffle=False,
                                                    data_type='valid',
                                                    src_max_len=None, tgt_max_len=None, src_max_vocab_size=None,
                                                    tgt_max_vocab_size=None, ignore_too_many_unknowns=None,
                                                    break_on_stop_iteration=True)

        encoder = toy_seq2seq_same_encoder_decoder.Encoder(bidirectional=True, num_layers=2, hidden_size=100, embedding_dim=100, pad_token=train_iterator.sourceField.word2idx['PAD'], dropout=0.0)
        decoder = toy_seq2seq_same_encoder_decoder.Decoder(hidden_size=100, output_size=train_iterator.sourceField.nWords, num_layers=2, dropout=0.0)

        model = toy_seq2seq_same_encoder_decoder.Model(encoder=encoder, decoder=decoder, vocab_size=train_iterator.sourceField.nWords, embedding_dim=100, pad_token=encoder.pad_token).cuda()

        train_loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                             reduction='sum', perform_dimension_checks=True).cuda()
        valid_loss = Loss.NMTLoss(target_vocabulary_len=train_iterator.targetField.nWords, target_padding_idx=1,
                                  reduction='sum', perform_dimension_checks=True).cuda()
        _optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        args = Args(model, train_iterator, gradient_checks=False,
                    norm_method='sents', train_loss=train_loss,
                    valid_loss=valid_loss, optimizer=_optimizer)
        trainer = Trainer.Trainer(args)
        b_size = 128
        for steps in range(100):
            batch = train_iterator.next_batch(b_size)
            batch.transpose()
            _ = trainer.propagate(batch, steps)

        valid_stats1 = trainer.validate(valid_iterator, b_size=1)
        valid_iterator.reset()
        valid_stats2 = trainer.validate(valid_iterator, b_size=2)
        valid_iterator.reset()
        valid_stats3 = trainer.validate(valid_iterator, b_size=3)
        valid_iterator.reset()
        valid_stats4 = trainer.validate(valid_iterator, b_size=5)

        self.assertEqual(valid_stats1.accuracy(), valid_stats2.accuracy())
        self.assertEqual(valid_stats1.accuracy(), valid_stats3.accuracy())
        self.assertEqual(valid_stats1.accuracy(), valid_stats4.accuracy())

        self.assertAlmostEqual(valid_stats1._loss(), valid_stats2._loss(), delta=1e-4)
        self.assertAlmostEqual(valid_stats1._loss(), valid_stats3._loss(), delta=1e-4)
        self.assertAlmostEqual(valid_stats1._loss(), valid_stats4._loss(), delta=1e-4)

if __name__ == '__main__':
    unittest.main()
