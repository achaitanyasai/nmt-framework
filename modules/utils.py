
import logging
import math
import time

import torch
import torch.nn
from torch.autograd import Variable
import numpy as np
logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

class Statistics(object):
    def __init__(self, loss = 0, n_words = 0, n_correct = 0, n_sentences = 0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_sentences = n_sentences
        self.start_time = time.time()
    
    def update(self, stat):
        self.loss = torch.sum(torch.tensor([self.loss, stat.loss]))
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        self.n_sentences += stat.n_sentences
    
    def _loss(self):
        if self.n_words == 0:
            return 0
        return self.loss.item() / float(self.n_words)

    def accuracy(self):
        if self.n_words == 0:
            return 0
        return 100 * (self.n_correct / float(self.n_words))

    def perplexity(self):
        #FIXME: isn't ppl: 2^(Cross Entropy Loss) instead of e^(Cross Entropy Loss) (??)
        return math.exp(min(self._loss(), 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def speed(self, t):
        return int(self.n_sentences / t)

    def output(self, epoch, batch):
        t = self.elapsed_time()
        log = 'Epoch: %d, Batch: %5d, Loss: %5.3f, Ppl: %5.3f, Acc: %5.3f, %3d sent/sec, Elapsed time: %3d sec'
        log = log % (epoch, batch, self._loss(), self.perplexity(), self.accuracy(), self.speed(t), t)
        if batch == -1:
            log += '\n'
        # logger.info(log)

    def _print(self, v):
        print('%d, nSents: %d, nWords: %d, nCorrect: %d, accuracy: %.12f, loss: %.12f, normalized loss: %.12f' % (v, self.n_sentences, self.n_words, self.n_correct, self.accuracy(), self.loss, self._loss()))

def calculate_correct_predictions(predictions, gtruth, target_padding_idx):
    pred = predictions.max(1)[1]
    non_padding = gtruth.ne(target_padding_idx)
    num_correct = pred.eq(gtruth).masked_select(non_padding).sum().item()

    return num_correct, non_padding.sum().item()

def var(a):
    return Variable(a)

def rvar(a, beam_size):
    return var(a.repeat(1, beam_size, 1))

def bottle(m, beam_size, batch_size):
    return m.view(batch_size * beam_size, -1)

def unbottle(m, beam_size, batch_size):
    return m.view(beam_size, batch_size, -1)

def get_hashes(inp, lang):
    '''
    Given word indices, this method returns 
    the hashes of the corresponding words.
    '''
    res = []
    inp = inp.transpose(0, 1)
    for i in inp:
        idx = i.data.cpu().numpy()[0]
        word = lang.idx2word[idx]
        if(word == ''):
            assert False
            word = 'UNK'
        ret = lang.word2HashSequence(word, None, 2)
        res.append(ret)
    res = Variable(torch.LongTensor(res), requires_grad=False).cuda()
    res = res.unsqueeze(1)
    return res

def check_gradients(model, lr):
    #TODO: investigate exploding gradient problem.

    for (name, p) in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            #TODO: attn: I need to go through what's happening internally.
            if 'embeddings' in name or 'attn' in name:
                continue
            if p.grad is None:
                continue
            param = np.linalg.norm(p.contiguous().view(-1).data.cpu().numpy())
            update = -lr * p.grad
            update = np.linalg.norm(update.contiguous().view(-1).data.cpu().numpy())
            if 1e-4 <= (update / param):
                pass
            else:
                # Vanishing gradient problem
                print("===========")
                print(lr)
                print(name)
                print(update, param)
                print(update / param)
                print("===========")
            if (update / param) < 1e-4:
                raise Exception('Gradient too small')