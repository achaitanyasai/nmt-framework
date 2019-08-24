#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import logging
import math
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

import data_iterator_optimized as data_iterator
import util
import subprocess
import logging.config
import warnings
import datetime

use_cuda = torch.cuda.is_available()

#The below line is for reproducibility of results, refer:
# https://github.com/pytorch/pytorch/issues/114 

torch.backends.cudnn.deterministic = True

random.seed(3435)
np.random.seed(3435)
torch.manual_seed(3435)
torch.cuda.manual_seed_all(3435)
torch.cuda.manual_seed(3435)

logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

src_lang = data_iterator.Lang('/tmp/in.src', 50, max_word_len_allowed=20, max_vocab_size=70000, langtype='source', chars = False, verbose=True, ignore_too_many_unknowns=True, max_number_of_sentences_allowed=10000000000)
tgt_lang = data_iterator.Lang('/tmp/in.tgt', 50, max_word_len_allowed=20, max_vocab_size=70000, langtype='target', chars = False, verbose=True, ignore_too_many_unknowns=True, max_number_of_sentences_allowed=10000000000)
        
iterator = data_iterator.dataIterator(src_lang, tgt_lang, shuffle = True)
cur_batch = 0

a = {}

while cur_batch < iterator.n_samples:

    batch = iterator.next_batch(80, None)
    if not batch:
        break
    print('%d/%d, %d' % (cur_batch, iterator.n_samples, batch.batch_size))
    for j in batch.src_raw:
        try:
            a[tuple(j)] += 1
        except KeyError:
            a[tuple(j)] = 1
    cur_batch += batch.batch_size

# for j in a:
#     if a[j] != 1:
#         print(j, a[j])
print("OK")
exit(0)
N = 10
i = 0
while i < N:
    # print iterator.next_batch(batch_size=1)
    try:
        print iterator.next()
        i += 1
    except Exception:
        pass