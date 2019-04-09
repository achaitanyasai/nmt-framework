'''
tests for seq2seq_attn.py
'''
import nmt

import unittest
from models import seq2seq_attn
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import data_iterator_optimized
from models import seq2seq_attn, utils
from modules import Loss, Optimizer, Trainer, utils
from tests.test_modules import argsWrapper

class testModel(unittest.TestCase):

    def setUp(self):
        self.src_lang = data_iterator_optimized.Lang('./tests/test_data.src', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='source', chars=False, verbose=False, ignore_too_many_unknowns=True)
        self.tgt_lang = data_iterator_optimized.Lang('./tests/test_data.tgt', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='target', chars=False, verbose=False, ignore_too_many_unknowns=True)
        
    def test_embeddings(self):
        # Not much to test here.
        # Will be taken care by asserts in encoder and decoder.
        # See self.test_encoder() and self.test_decoder()
        return
        # iterator = data_iterator_optimized.dataIterator(self.src_lang, self.tgt_lang, shuffle = False)
        # batch = iterator.next_batch(2, None, self.src_lang, self.tgt_lang)
        # encoder = seq2seq_attn.encoder('LSTM', True, 2, 10, self.src_lang.n_words, 10, self.src_lang.word2idx['PAD'], self.src_lang.n_hashes, 0.0).cuda()
        # for p in encoder.parameters():
        #     p.data.uniform_(-0.1, 0.1)
        # ret = encoder.embeddings(batch.src)
        # self.assertEqual(ret.shape, (batch.src.shape[0], batch.src.shape[1], 10))
        # ret = encoder.apply_cnn_charngrams(batch.src_hashes, batch.src.shape[0], batch.src.shape[1])
        
        # tmp = batch.src_hashes.transpose(0, 1).contiguous().view(-1, 6)
        # tmp_emb = encoder.ngrams_embeddings(tmp).unsqueeze(1)
        # tmp_emb_cnn = [cnn(tmp_emb).squeeze(3).max(2)[0] for cnn in encoder.cnns]
        # tmp_emb_cnn = torch.cat(tmp_emb_cnn, 1)
        # actual = encoder.char_highway_network(tmp_emb_cnn).view(batch.src.shape[0], batch.src.shape[1], 128)
    
    def test_encoder(self):
        iterator = data_iterator_optimized.dataIterator(self.src_lang, self.tgt_lang, shuffle = False)
        batch = iterator.next_batch(2, None, self.src_lang, self.tgt_lang)
        encoder = seq2seq_attn.CustomEncoder('LSTM', True, 2, 10, self.src_lang.n_words, 10, self.src_lang.word2idx['PAD'], self.src_lang.n_hashes, 0.0).cuda()        
        
        hidden_t, outputs, _ = encoder.forward(batch.src, batch.src_hashes, batch.src_len, None)
        
        self.assertEqual(type(hidden_t), tuple)
        self.assertEqual(hidden_t[0].shape, (4, 2, 5))
        self.assertEqual(hidden_t[1].shape, (4, 2, 5))
        self.assertEqual(outputs.shape, (4, 2, 10))
    
    def test_decoder(self):
        iterator = data_iterator_optimized.dataIterator(self.src_lang, self.tgt_lang, shuffle = False)
        batch = iterator.next_batch(2, None, self.src_lang, self.tgt_lang)
        encoder = seq2seq_attn.CustomEncoder('LSTM', True, 2, 10, self.src_lang.n_words, 10, self.src_lang.word2idx['PAD'], self.src_lang.n_hashes, 0.0).cuda()        
        decoder = seq2seq_attn.decoder('LSTM', True, 2, 10, self.tgt_lang.n_words, self.tgt_lang.n_hashes, 10, self.tgt_lang.word2idx['PAD'], 'general', dropout=0.0).cuda()
        hidden_t, outputs, _ = encoder(batch.src, batch.src_hashes, batch.src_len, None)
        enc_state = decoder.init_decoder_state(batch.src, outputs, hidden_t)

        out, dec_state, attns, predictions = decoder(batch.tgt, batch.tgt_hashes, outputs, enc_state, context_lengths = batch.src_len)
        self.assertEqual(out.shape, (4, 2, 10))
        self.assertEqual(attns['std'].shape, (4, 2, 4))
        self.assertEqual(predictions.shape, (4, 2, 7)) #7: vocab size
    
    def test_model(self):
        iterator = data_iterator_optimized.dataIterator(self.src_lang, self.tgt_lang, shuffle = False)
        batch = iterator.next_batch(2, None, self.src_lang, self.tgt_lang)
        encoder = seq2seq_attn.CustomEncoder('LSTM', True, 2, 10, self.src_lang.n_words, 10, self.src_lang.word2idx['PAD'], self.src_lang.n_hashes, 0.0).cuda()        
        decoder = seq2seq_attn.decoder('LSTM', True, 2, 10, self.tgt_lang.n_words, self.tgt_lang.n_hashes, 10, self.tgt_lang.word2idx['PAD'], 'general', dropout=0.0).cuda()
        model = seq2seq_attn.Seq2SeqAttention(encoder, decoder)

        out, attns, dec_state, predictions, _ = model(batch.src, batch.tgt, batch.src_len, batch.src_hashes, batch.tgt_hashes)

        self.assertEqual(out.shape, (3, 2, 10)) # 3 because we need to remove last target token.
        self.assertEqual(attns['std'].shape, (3, 2, 4))
        self.assertEqual(predictions.shape, (3, 2, 7))
    
class testTrain_hn_te(unittest.TestCase):
    def setUp(self):
        self.args = argsWrapper()
        self.args.dispFreq = 100000
        self.args.lrate = 1.0

        self.te_lang = data_iterator_optimized.Lang('./tests/train.top1000.te', 50, max_word_len_allowed=60, max_vocab_size=5000, langtype='target', chars=False, verbose=False, ignore_too_many_unknowns=False, data_type='train', max_number_of_sentences_allowed=1000)
        self.hn_lang = data_iterator_optimized.Lang('./tests/train.top1000.hn', 50, max_word_len_allowed=60, max_vocab_size=4000, langtype='source', chars=False, verbose=False, ignore_too_many_unknowns=False, data_type='train', max_number_of_sentences_allowed=1000)

        self.te_lang_valid = data_iterator_optimized.Lang('./tests/valid.top100.te', 50, max_word_len_allowed=60, max_vocab_size=10000, langtype='target', chars=False, verbose=False, ignore_too_many_unknowns=True, data_type='valid')
        self.hn_lang_valid = data_iterator_optimized.Lang('./tests/valid.top100.hn', 50, max_word_len_allowed=60, max_vocab_size=10000, langtype='source', chars=False, verbose=False, ignore_too_many_unknowns=True, data_type='valid')

        self.hn_te_iterator = data_iterator_optimized.dataIterator(self.hn_lang, self.te_lang, shuffle=True)
        self.hn_te_val_iterator = data_iterator_optimized.dataIterator(self.hn_lang_valid, self.te_lang_valid, shuffle=False)

        self.encoder_hn_te = seq2seq_attn.CustomEncoder('LSTM', True, 2, 500, self.hn_lang.n_words, 500, self.hn_lang.word2idx['PAD'], self.hn_lang.n_hashes, dropout=0.0).cuda()        
        self.decoder_hn_te = seq2seq_attn.decoder('LSTM', True, 2, 500, self.te_lang.n_words, self.te_lang.n_hashes, 500, self.te_lang.word2idx['PAD'], 'general', dropout=0.0).cuda()
        self.model_hn_te = seq2seq_attn.Seq2SeqAttention(self.encoder_hn_te, self.decoder_hn_te)

        self.optimizer_hn_te = nmt.get_optimizer(self.args, self.model_hn_te, verbose=False)
        self.train_loss_hn_te = nmt.get_criterion(self.args, self.te_lang)
        self.valid_loss_hn_te = nmt.get_criterion(self.args, self.te_lang)
        self.Trainer_hn_te = Trainer.Trainer(
            self.model_hn_te, self.train_loss_hn_te, self.valid_loss_hn_te, self.optimizer_hn_te, norm_method='tokens',
            gradient_checks=False
        )

    def test_train_hn_te(self):
        return
        print 'Training model on hn-te-40K sentences for gradient checks'
        for p in self.model_hn_te.parameters():
            p.data.uniform_(-0.1, 0.1)
        
        for i in range(100):
            self.hn_te_iterator.reset()
            self.args.batch_size = 50
            try:
                train_stats = self.Trainer_hn_te.train(self.args, self.hn_te_iterator, self.hn_lang, self.te_lang, i)
            except Exception, error:
                if str(error) == 'Gradient too small':
                    self.fail(error)
                else:
                    # Re-running so as to track the exact error.
                    train_stats = self.Trainer_hn_te.train(self.args, self.hn_te_iterator, self.hn_lang, self.te_lang, i)
            train_stats.output(i + 1, 0)
            self.model_hn_te.train()
        self.assertGreaterEqual(train_stats.accuracy(), 99.0)
        self.assertLessEqual(train_stats._loss(), 0.1)
