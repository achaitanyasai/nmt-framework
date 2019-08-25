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

import data_iterator
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
