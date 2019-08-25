# -*- coding: utf-8 -*-


import random

import numpy as np
import torch

import data_iterator
import sys
import logging.config

use_cuda = torch.cuda.is_available()

#The below line is for reproducibility of results, refer:
# https://github.com/pytorch/pytorch/issues/114 

torch.backends.cudnn.deterministic = True

random.seed(3435)
np.random.seed(3435)
torch.manual_seed(3435)
torch.cuda.manual_seed_all(3435)
torch.cuda.manual_seed(3435)

train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_very_small/data.csv', shuffle=True,
                                              data_type='train',
                                              src_max_len=100, tgt_max_len=100, src_max_vocab_size=100,
                                              tgt_max_vocab_size=100, ignore_too_many_unknowns=True)
for i in range(100):
    a = train_iterator.next_batch(1)
    print(a.src)
    sys.stdout.flush()