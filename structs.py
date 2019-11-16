import os
import logging
import torch

# FIXME: Workaround for the time being. Fix in later versions.
os.system('mkdir -p ./data')

logger = logging.getLogger('NMT')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('./data/training_logs.txt')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)8s --- %(message)s (%(filename)s:%(lineno)s)','%Y-%m-%d | %H:%M:%S')
fh.setFormatter(formatter)
logger.addHandler(fh)

class Criterions(object):
    def __init__(self, train_criterion,
                 valid_criterion):
        self.train_criterion = train_criterion
        self.valid_criterion = valid_criterion


class TrainValidTestIterator(object):
    def __init__(self, train_iterator, valid_iterator, test_iterator):

        self.train_iterator = train_iterator
        self.valid_iterator = valid_iterator
        self.test_iterator = test_iterator

    def __del__(self):
        del self.train_iterator
        del self.valid_iterator
        del self.test_iterator

class EncoderOutputs(object):
    def __init__(self, hidden_t, outputs, encoder_embeddings=None):
        self.hidden_t = hidden_t
        self.outputs = outputs
        self.encoder_embeddings = encoder_embeddings

    def combine_forward_backward(self):
        h = self._concat(self.hidden_t[0])
        c = self._concat(self.hidden_t[1])
        self.hidden_t = (h, c)

    def _concat(self, x):
        return torch.cat([x[0:x.size(0):2], x[1:x.size(0):2]], 2)


class DecoderOutputs(object):
    def __init__(self, predictions, penalty=None, hidden=None, dec_state=None):
        self.predictions = predictions
        self.penalty = penalty
        self.hidden = hidden
        self.dec_state = dec_state

class ModelOutputs(object):
    def __init__(self, predictions, penalty=None, dec_state=None):
        self.predictions = predictions
        self.penalty = penalty
        self.dec_state = dec_state
