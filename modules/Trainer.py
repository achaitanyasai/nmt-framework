#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Trains and validates the model.
'''
import logging
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import data_iterator

import models
from . import utils
from structs import *

class Trainer(object):
    '''
    Core training module
    Trains and validates the model
    '''
    def __init__(self, args):

        self.args = args
        self.args.model.train()
        self.padding_idx = self.args.iterators.train_iterator.sourceField.word2idx['PAD']
        assert self.args.iterators.train_iterator.sourceField.word2idx['PAD'] == self.args.iterators.train_iterator.targetField.word2idx['PAD']
        self.gradient_checks = args.gradient_checks

    def propagate(self, batch, cur_step):

        # TODO: Truncated BPTT.
        self.args.model.train()
        # Normalize with respect to sentences
        normalization = batch.batch_size
        if self.args.norm_method == 'tokens':
            # Normalize with respect to tokens
            normalization = batch.tgt.data.contiguous().view(-1).ne(self.padding_idx).sum()

        # logger.info("Norm: %d" % int(normalization))
        self.args.model.zero_grad()
        
        outputsFromModel = self.args.model(batch)

        loss, stats = self.args.train_loss.compute_loss(batch, outputsFromModel, cur_step)
        loss.div(normalization).backward()

        if self.gradient_checks:
            utils.check_gradients(self.args.model, self.args.optimizer.lr)

        self.args.optimizer.step()
        if outputsFromModel.dec_state is not None:
            outputsFromModel.dec_state.detach()

        return stats
    
    
    def validate(self, valid_iterator, b_size=20):
        '''
        Not a correct way of validation. Kept as a backup.
        See self.validate_fixed() for correct implementation.
        '''
        # return self.validate_fixed(args, valid_data_iterator, source_data, target_data, batch_size = batch_size)

        #setting validation mode
        self.args.model.eval()

        stats = utils.Statistics()
        cur_batch = 0
        normalization = 0
        assert valid_iterator.nSentences == valid_iterator.nSentencesPruned
        while cur_batch < valid_iterator.nSentences:
            batch = valid_iterator.next_batch(b_size)
            batch.transpose()
            if not batch:
                break
            assert batch.batch_size == batch.tgt.size(1)

            outputsFromModel = self.args.model(batch)
            loss, batch_stats = self.args.valid_loss.compute_loss(batch, outputsFromModel, cur_step=cur_batch)
            # data_iterator.logger.info('[validation] batch: %d, loss: %.6f, accuracy" %.6f' % (cur_batch, batch_stats._loss(), batch_stats.accuracy()))
            stats.update(batch_stats)
            cur_batch += batch.batch_size
            # if cur_batch >= 24:
            #     break
        
        #setting back to training mode
        self.args.model.train()
        return stats
    
    def validate_fixed(self, valid_iterator, b_size=20):
        #setting validation mode
        self.args.model.eval()

        stats = utils.Statistics()
        cur_batch = 0
        normalization = 0
        while cur_batch < valid_iterator.nSentences:
            batch = valid_iterator.next_batch(b_size)
            batch.transpose()
            if not batch:
                break
            assert batch.batch_size == batch.tgt.size(1)

            src = batch.src
            src_lens = batch.src_lens

            tgt = batch.tgt
            tgt = tgt[:-1]

            enc_hidden, context, penalty, encoder_embeddings = self.args.model.encoder(src, None, src_lens)
            dec_state = self.args.model.decoder.init_decoder_state(src, context, enc_hidden)

            L = tgt.shape[0]
            inp = tgt[0, :].unsqueeze(0)
            predictions = None
            all_predictions = []
            all_dec_out = []
            for i in range(L):

                if i > 0:
                    inp = predictions.max(2)[1]

                dec_out, dec_state, _, predictions = self.args.model.decoder(inp, None, context,
                                                                  encoder_embeddings, dec_state,
                                                                  context_lengths=src_lens)
                dec_out = dec_out.squeeze(0)
                all_dec_out.append(dec_out)
                all_predictions.append(predictions)

            all_dec_out = torch.stack(all_dec_out)
            all_predictions = torch.stack(all_predictions).squeeze(1)

            ret = DecoderOutputs(all_predictions, penalty, all_dec_out, dec_state)

            loss, batch_stats = self.args.valid_loss.compute_loss(batch, ret, cur_step=cur_batch)
            # logger.info('[validation_fixed] Batch: %d, Loss: %.6f, Accuracy: %.6f' % (cur_batch + 1, batch_stats._loss(), batch_stats.accuracy()))
            # print('[validation_fixed] Batch: %d, Loss: %.6f, Accuracy: %.6f' % (
            # cur_batch + 1, batch_stats._loss(), batch_stats.accuracy()))
            stats.update(batch_stats)
            cur_batch += batch.batch_size
        
        #setting back to training mode
        self.args.model.train()
        return stats

    def lr_step(self, loss, epoch):
        return self.args.optimizer.update_learning_rate(loss, epoch)
