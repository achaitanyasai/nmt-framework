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
import utils


class Trainer(object):
    '''
    Core training module
    Trains and validates the model
    '''
    def __init__(self, model, train_loss, valid_loss,
                 optimizer, norm_method='sents', gradient_checks=False):
        
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optimizer = optimizer
        self.norm_method = norm_method
        self.padding_idx = 1 #By convention
        self.model.train()
        self.gradient_checks = gradient_checks

    def train(self, args, train_data_iterator, source_data,
              target_data, cur_epoch):

        stats = utils.Statistics()
        
        #setting training mode
        self.model.train()
        
        normalization = 0
        padding_idx = target_data.word2idx['PAD']
        self.padding_idx = padding_idx
        cur_batch = 0
        while cur_batch < train_data_iterator.n_samples:
            batch = train_data_iterator.next_batch(args.batch_size, None)
            if self.norm_method == 'tokens':
                #Loss normalization based on words
                normalization += batch.src.data.contiguous().view(-1).ne(self.padding_idx).sum()
            else:
                #Loss normalization based on sentences
                #lines_trg shape is: maxlen x batch size
                #.size(1) is batch size
                #see https://discuss.pytorch.org/t/the-default-value-of-size-average-true-in-loss-function-is-a-trap/4251/4
                normalization += batch.src.size(1)
            
            #Forward pass and back propagate the loss on the current batch
            self.propagate(batch, stats, normalization, cur_epoch)
            
            normalization = 0
            cur_batch += batch.batch_size
            
            if cur_batch % args.dispFreq == 0:
                stats.output(cur_epoch, cur_batch)
        return stats
    
    
    def validate(self, args, valid_data_iterator, source_data, target_data, batch_size = 20):
        '''
        Not a correct way of validation. Kept as a backup.
        See self.validate_fixed() for correct implementation.
        '''
        # return self.validate_fixed(args, valid_data_iterator, source_data, target_data, batch_size = batch_size)
        #setting validation mode
        self.model.eval()

        stats = utils.Statistics()
        cur_batch = 0
        normalization = 0
        padding_idx = target_data.word2idx['PAD']
        self.padding_idx = padding_idx
        while cur_batch < valid_data_iterator.n_samples:
            batch = valid_data_iterator.next_batch(batch_size, None, source_language_model = source_data, target_language_model = target_data)
            
            lines_src, lens_src = batch.src, batch.src_len
            lines_trg, lens_trg = batch.tgt, batch.tgt_len
            lines_src_hashes = batch.src_hashes
            lines_trg_hashes = batch.tgt_hashes

            if self.norm_method == 'tokens':
                normalization += lines_trg.data.contiguous().view(-1).ne(self.padding_idx).sum()
            else:
                normalization += lines_trg.size(1) #shape is: maxlen x batches
            
            #Predict the outputs from current lines_src
            # outputs, attns, dec_state, predictions = self.model(lines_src, lines_trg, lens_src)
            outputs, attns, dec_state, predictions, penalty = self.model(lines_src, lines_trg, lens_src, lines_src_hashes, lines_trg_hashes)
            #Compute loss
            loss, batch_stats = self.valid_loss.compute_loss(batch, outputs, predictions, attns, 0, normalization, penalty)

            stats.update(batch_stats)
            cur_batch += batch.batch_size
        
        #setting back to training mode
        self.model.train()
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
    
    def propagate(self, batch, total_stats, normalization, cur_epoch):
        
        #Forward pass and backpropagate
        #Originally it's truncated BPTT specified by trunc_size
        #As of now, it's only BPTT.
        #TODO: Truncated BPTT.
        lines_src, lens_src = batch.src, batch.src_len
        lines_trg, lens_trg = batch.tgt, batch.tgt_len
        lines_src_hashes = batch.src_hashes
        lines_trg_hashes = batch.tgt_hashes

        #getting seq lengths
        target_size = lines_trg.size(0) #seq len x batch size
        
        trunc_size = target_size #TODO: add truncated size
        dec_state = None
        
        for j in range(0, target_size, trunc_size):
            tgt = lines_trg[j : j + trunc_size]

            #resetting the gradient
            self.model.zero_grad()

            #Forward pass: predict the outputs from lines_src
            outputs, attns, dec_state, predictions, penalty = self.model(lines_src, tgt, lens_src, lines_src_hashes, lines_trg_hashes)
            
            #Backpropagate: compute the loss and update weights
            loss, batch_stats = self.train_loss.compute_loss(batch, outputs, predictions, attns, j, normalization, penalty)
            
            #backpropogate
            loss.div(normalization).backward()

            if self.gradient_checks and cur_epoch >= 5:
                utils.check_gradients(self.model, self.optimizer.lr)
            
            self.optimizer.step()

            total_stats.update(batch_stats)

            #This snippet is not required
            #Remove once finalized
            if dec_state is not None:
                dec_state.detach()

    def epoch_step(self, ppl, epoch):
        return self.optimizer.update_learning_rate(ppl, epoch)
