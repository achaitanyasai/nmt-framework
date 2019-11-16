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

            stats.update(batch_stats)
            cur_batch += batch.batch_size
            # if cur_batch >= 24:
            #     break
        
        #setting back to training mode
        self.args.model.train()
        return stats
    
    def validate_fixed(self, args, valid_data_iterator, source_data, target_data, batch_size = 20):
        #setting validation mode
        self.model.eval()

        stats = utils.Statistics()
        cur_batch = 0
        normalization = 0
        padding_idx = target_data.word2idx['PAD']
        self.padding_idx = padding_idx
        while cur_batch < valid_data_iterator.n_samples:
            batch = valid_data_iterator.next_batch(batch_size, None, source_language_model = source_data, target_language_model = target_data)
            if not batch:
                break
            
            lines_src, lens_src = batch.src, batch.src_len
            lines_trg, lens_trg = batch.tgt, batch.tgt_len
            lines_src_hashes = batch.src_hashes
            #lines_trg_hashes = batch.tgt_hashes.transpose(0, 1)

            if self.norm_method == 'tokens':
                normalization += lines_trg.data.contiguous().view(-1).ne(self.padding_idx).sum()
            else:
                normalization += lines_trg.size(1) #shape is: maxlen x batches
            enc_states, context, penalty = self.model.encoder(lines_src, lines_src_hashes, lens_src)
            
            L = lines_trg.shape[0] - 1 #-1 because, we don't have to use last word to predict.
            
            dec_state = self.model.decoder.init_decoder_state(lines_src, context, enc_states)
            attns = []
            dec_out = []
            predictions = []
            previous_predictions = None
            cur_lines_trg_hashes = None
            for seq_idx in range(L):
                if seq_idx == 0:
                    cur_lines_trg = lines_trg[seq_idx].unsqueeze(0)
                    #cur_lines_trg_hashes = lines_trg_hashes[seq_idx].unsqueeze(0)
                    #cur_lines_trg_hashes = cur_lines_trg_hashes.transpose(0, 1)
                else:
                    cur_lines_trg = previous_predictions.max(2)[1]
                    #cur_lines_trg_hashes = utils.get_hashes(cur_lines_trg, target_data)

                cur_dec_out, dec_state, cur_attn, cur_predictions = self.model.decoder(cur_lines_trg, cur_lines_trg_hashes, context, dec_state, context_lengths = lens_src)
                previous_predictions = cur_predictions
                predictions.append(cur_predictions)
                attns.append(cur_attn['std'])
                dec_out.append(cur_dec_out)
            
            _attns = torch.cat(attns, 0)
            attns = {
                'std': _attns
            }
            outputs = torch.cat(dec_out, 0)
            predictions = torch.cat(predictions, 0)

            loss, batch_stats = self.valid_loss.compute_loss(batch, outputs, predictions, attns, 0, normalization, penalty)

            stats.update(batch_stats)

            cur_batch += batch.batch_size
        
        #setting back to training mode
        self.model.train()
        return stats

    def lr_step(self, loss, epoch):
        return self.args.optimizer.update_learning_rate(loss, epoch)
