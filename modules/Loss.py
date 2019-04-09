'''
Loss function(s)
'''

import torch
import torch.nn
from torch.autograd import Variable

import utils


class LossBase(torch.nn.Module):
    '''
    Loss module base class
    '''
    def __init__(self, tgt_vocab, target_padding_idx):
        super(LossBase, self).__init__()
        self.tgt_vocab = tgt_vocab
        self.target_padding_idx = target_padding_idx
    
    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(-1))

class NMTLoss(LossBase):
    def __init__(self, tgt_vocab, target_padding_idx, normalization_type = 'sents'):
        super(NMTLoss, self).__init__(tgt_vocab, target_padding_idx)
        self.tgt_vocab = tgt_vocab
        self.normalization_type = normalization_type
        self.target_padding_idx = target_padding_idx

        weight = torch.ones(len(tgt_vocab))
        weight[target_padding_idx] = 0
        # weight[tgt_vocab['UNK']] = 0.4
        
        #Declaring size_average as False
        #See: https://discuss.pytorch.org/t/the-default-value-of-size-average-true-in-loss-function-is-a-trap/4251
        self.criterion = torch.nn.NLLLoss(weight, reduction = 'sum')
    
    def compute_loss(self, batch, outputs, predictions, attns, cur_trunc, normalization, penalty):
        """
        lines_src shape: seq len x batch size
        lines_trg shape: seq len x batch size
        predictions shape: seq len x batch size x vocab size
        """

        lines_src, lens_src = batch.src, batch.src_len
        lines_trg, lens_trg = batch.tgt, batch.tgt_len

        #popping first token because of SOS
        lines_trg = lines_trg[1:]

        gtruth = lines_trg.contiguous().view(-1)
        predictions = predictions.contiguous().view(-1, len(self.tgt_vocab))
        
        loss = self.criterion(predictions, gtruth)
        if penalty is not None:
            loss = loss + (penalty)
        cur_n_sentences = lines_src.size(1)
        cur_n_correct, cur_n_words = utils.calculate_correct_predictions(predictions.data, gtruth.data, self.target_padding_idx)
        
        batch_stats = self.stats(loss, cur_n_words, cur_n_correct, cur_n_sentences)
        return loss, batch_stats

    def stats(self, loss, cur_n_words, cur_n_correct, cur_n_sentences):
        return utils.Statistics(loss.item(), cur_n_words, cur_n_correct, cur_n_sentences)
