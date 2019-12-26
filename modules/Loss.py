'''
Loss function(s)
'''

import logging
import torch
import torch.nn
from torch.autograd import Variable

from . import utils
from structs import *

class LossBase(torch.nn.Module):
    '''
    Loss module base class
    '''
    def __init__(self, target_vocabulary_len, target_padding_idx):
        super(LossBase, self).__init__()
        self.target_vocabulary_len = target_vocabulary_len
        self.target_padding_idx = target_padding_idx
    
    # def _bottle(self, v):
    #     return v.view(-1, v.size(2))
    #
    # def _unbottle(self, v, batch_size):
    #     return v.view(-1, batch_size, v.size(-1))

class NMTLoss(LossBase):
    def __init__(self, target_vocabulary_len, target_padding_idx,
                 reduction='sum', perform_dimension_checks=False):
        super(NMTLoss, self).__init__(target_vocabulary_len, target_padding_idx)
        self.targetVocabularyLen = target_vocabulary_len
        self.target_padding_idx = target_padding_idx
        self.perform_dimension_checks = perform_dimension_checks

        weight = torch.ones(self.targetVocabularyLen, dtype=torch.float32)
        weight[target_padding_idx] = 0.0
        # weight[tgt_vocab['UNK']] = 0.4
        
        #Declaring size_average as False
        #See: https://discuss.pytorch.org/t/the-default-value-of-size-average-true-in-loss-function-is-a-trap/4251
        self.criterion = torch.nn.NLLLoss(weight=weight, reduction=reduction)
    
    def compute_loss(self, batch, modelOutputs, cur_step):
        """
        lines_src shape: seq len x batch size
        lines_trg shape: seq len x batch size
        predictions shape: seq len x batch size x vocab size
        """

        lines_src = batch.src
        lines_trg = batch.tgt
        predictions = modelOutputs.predictions
        penalty = modelOutputs.penalty

        if self.perform_dimension_checks:
            # logger.info('Performing dimension checks')
            assert lines_src.shape[1] == lines_trg.shape[1]
            assert lines_src.shape[1] == predictions.shape[1]
            assert lines_trg.shape[0] - 1 == predictions.shape[0]
            assert predictions.shape[2] == self.targetVocabularyLen
            # logger.info('Dimension checks OK')
        else:
            # logger.warning('Run with perform_dimension_checks flag to be 100% sure')
            pass

        # popping first token because of SOS
        lines_trg = lines_trg[1:]

        gtruth = lines_trg.contiguous().view(-1)
        predictions = predictions.contiguous().view(-1, self.targetVocabularyLen)

        if penalty:
            loss = self.criterion(predictions, gtruth) + 0.7 * penalty
        else:
            loss = self.criterion(predictions, gtruth)

        cur_n_sentences = lines_src.size(1)
        cur_n_correct, cur_n_words = utils.calculate_correct_predictions(predictions.data, gtruth.data, self.target_padding_idx)
        
        stats = utils.Statistics(loss, cur_n_words, cur_n_correct, cur_n_sentences)
        # if cur_step % 1 == 0:
        #     logger.info("Step: %4d, Loss: %.20f, Accuracy: %.6f" % (cur_step, stats._loss(), stats.accuracy()))
        return loss, stats
