"""
Sequence to Sequence attention
Decoder is Input feed decoder
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from . import GlobalAttention
# import GlobalAttention
from . import utils
# import utils
import warnings

class encoder(nn.Module):
    def __init__(
            self, 
            rnn_type, 
            bidirectional, 
            num_layers, 
            hidden_size,
            vocab_size,
            embedding_dim,
            pad_token,
            dropout = 0.4):
        
        super(encoder, self).__init__()
    
    def forward(self, input, input_hashes, lengths = None, hidden = None):
        return None

class decoderBase(nn.Module):
    '''
    decoder base module
    '''
    def __init__(self, 
        rnn_type, 
        bidirectional_encoder,
        num_layers,
        hidden_size,
        vocab_size,
        embedding_dim,
        pad_token,
        attn_type = 'general',
        coverage_attn = False,
        copy_attn = False,
        context_gate = None,
        dropout = 0.0):
        super(decoderBase, self).__init__()

    def forward(
        self,
        input,
        input_charngrams,
        embedding_context,
        context,
        state,
        context_lengths):
        return None


class decoder(decoderBase):
    '''
    Input feed decoder
    '''
    def _run_forward_pass(self, input, input_hashes, embedding_context, context, state, context_lengths = None):
        pass

class Seq2SeqAttention(nn.Module):
    """
    encoder + decoder model
    """
    def __init__(self, encoder, decoder):
        super(Seq2SeqAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, tgt, lengths, src_hashes, tgt_hashes):
        return None
