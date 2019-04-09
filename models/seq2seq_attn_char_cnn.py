"""
Sequence to Sequence attention
Encoder is (char_cnn on characters of a word to get word repr) + (LSTM on words)
Decoder is Input feed decoder
"""

"""
Is it good to put common modules(decoder for instance) in a seperate module (??)
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import GlobalAttention
import utils


class DecoderState(object):
    def detach(self):
        for h in self._all:
            if h is not None:
                h.detach_()
    
    def beam_update(self, idx, positions, beam_size):
        for e in self._all:
            a, br, d = e.size()
            sent_states = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sent_states.data.copy_(sent_states.data.index_select(1, positions))

class RNNDecoderState(DecoderState):
    def __init__(self, context, hidden_size, rnnstate):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None

        # Init the input feed.
        batch_size = context.size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = Variable(context.data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]

class encoder(nn.Module):
    def __init__(
            self, 
            rnn_type, 
            bidirectional, 
            num_layers, 
            hidden_size,
            vocab_size,
            embedding_dim,
            char_embedding_dim,
            pad_token, 
            kernel_sizes = [3, 4, 5, 6],
            filters = [16, 16, 16, 16],
            dropout = 0.4):
        super(encoder, self).__init__()
        self.no_pack_padded_seq = False
        self.vocab_size = vocab_size

        self.num_layers = num_layers
        self.pad_token = pad_token
        self.embedding_dim = embedding_dim
        self.num_directions = 1 + bidirectional
        self.bidirectional = bidirectional

        self.char_embedding_dim = char_embedding_dim
        self.kernel_sizes = kernel_sizes
        self.filters = filters
        self.highway_network_input_dim = sum(filters) #Should get updated according to n_filters
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        assert(hidden_size % self.num_directions == 0)
        self.hidden_size = hidden_size // self.num_directions

        self.rnn = getattr(nn, rnn_type)( #LSTM or GRU
            input_size = embedding_dim, 
            hidden_size = self.hidden_size, 
            num_layers = num_layers, 
            dropout = dropout, 
            bidirectional = self.bidirectional
        )

        self.embeddings = nn.Embedding(
            vocab_size,
            char_embedding_dim,
            pad_token
        )

        self.highway_network = utils.Highway(self.highway_network_input_dim, self.embedding_dim)

        self.cnns = nn.ModuleList()
        self.build_cnns()

    
    def build_cnns(self):
        l = len(self.kernel_sizes)
        for i in xrange(l):
            self.cnns.append(nn.Conv2d(1, self.filters[i], (self.kernel_sizes[i], self.char_embedding_dim)))
    
    def apply_cnn(self, input):
        s_len, n_batch, w_len = input.size()[0], input.size()[1], input.size()[2]
        
        input_shape = input.size()
        res = []
        input = input.contiguous().view(-1, input.size(2))
        emb = self.embeddings(input)
        emb = emb.unsqueeze(1)
        conv = [F.tanh(cnn(emb)).max(2)[0].squeeze(2) for cnn in self.cnns]
        conv = torch.cat(conv, 1)
        res = self.highway_network(conv)
        res = res.view((s_len, n_batch, -1))
        res = self.dropout(res)
        return res
        
    def forward(self, input, lengths = None, hidden = None):
        self.rnn.flatten_parameters()
        s_len, n_batch = input.size()[0], input.size()[1]
        if lengths is not None:
            n_batch_, = lengths.size()
            utils.aeq(n_batch, n_batch_)
        
        #emb = self.embeddings(input)
        emb = self.apply_cnn(input)

        s_len, batch, embedding_dim = emb.size()
        assert(embedding_dim == self.embedding_dim)

        #Packed seq. see:
        # https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Models.py#L131

        packed_emb = emb
        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)
        
        outputs, hidden_t = self.rnn(packed_emb, hidden)
        if lengths is not None:
            outputs = unpack(outputs)[0]

        return hidden_t, outputs


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
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.attn_type = attn_type
        
        self.context_gate = context_gate
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.embeddings = nn.Embedding(
            vocab_size,
            embedding_dim,
            pad_token
        )

        self.rnn = self._build_rnn(
            rnn_type, 
            self.embedding_dim + self.hidden_size, #FIXME: If not input feed, it's only self.embedding_dim
            self.hidden_size,
            self.num_layers,
            dropout
        )

        self.decoder2vocab = nn.Sequential(
            nn.Linear(self.hidden_size, self.vocab_size),
            nn.LogSoftmax(dim=-1)
        )

        self._coverage = coverage_attn

        self.attn = GlobalAttention.GlobalAttention(
            self.hidden_size,
            coverage=coverage_attn,
            attn_type=self.attn_type
        )

        self._copy = False
        if copy_attn:
            self.copy_attn = GlobalAttention.GlobalAttention(
                self.hidden_size,
                attn_type=self.attn_type
            )
            self._copy = True
    
    def forward(
        self,
        input,
        context,
        state,
        context_lengths):

        assert isinstance(state, RNNDecoderState)
        input_len, input_batch = input.size()
        context_len, context_batch, _ = context.size()

        utils.aeq(input_batch, context_batch)

        hidden, outputs, attns, coverage = self._run_forward_pass(input, context, state, context_lengths = context_lengths)

        final_output = outputs[-1]
        state.update_state(hidden, final_output.unsqueeze(0), coverage.unsqueeze(0) if coverage is not None else None)

        outputs = torch.stack(outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])
        
        predictions = self.decode(outputs)

        return outputs, state, attns, predictions

    def decode(self, outputs):
        batch_size = outputs.size(1)
        outputs = outputs.view(-1, outputs.size(2))
        predictions = self.decoder2vocab(outputs)
        predictions = predictions.view(-1, batch_size, predictions.size(1))
        return predictions

    def _fix_enc_hidden(self, h):
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, src, context, enc_hidden):
        if isinstance(enc_hidden, tuple): #LSTM
            return RNNDecoderState(context, self.hidden_size, tuple([self._fix_enc_hidden(enc_hidden[i]) for i in xrange(len(enc_hidden))]))
        else: #GRU
            return RNNDecoderState(context, self.hidden_size, self._fix_enc_hidden(enc_hidden))
        
class decoder(decoderBase):
    '''
    Input feed decoder
    '''

    def _run_forward_pass(self, input, context, state, context_lengths = None):
        output = state.input_feed.squeeze(0)
        output_batch, _ = output.size()
        input_len, input_batch = input.size()
        utils.aeq(input_batch, output_batch)
        
        assert not self._copy
        assert not self._coverage

        outputs = []
        attns = {'std' : []}
        if self._copy:
            attns['copy'] = []
        if self._coverage:
            attns['coverage'] = []

        emb = self.embeddings(input)
        assert emb.dim() == 3
        
        hidden = state.hidden
        coverage = state.coverage.squeeze(0) if state.coverage is not None else None
        
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            emb_t = torch.cat([emb_t, output], 1) #Input feed

            rnn_output, hidden = self.rnn(emb_t, hidden)
            attn_output, attn = self.attn(
                rnn_output,
                context.transpose(0, 1),
                context_lengths = context_lengths)
            #TODO: context gate, see:
            #https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Models.py#L465
            output = self.dropout(attn_output)
            outputs += [output]
            attns['std'] += [attn]

            if self._coverage:
                coverage = coverage + attn \
                    if coverage is not None else attn
                attns['coverage'] += [coverage]
            
            if self._copy:
                _, copy_attn = self.copy_attn(output, context.transpose(0, 1))
                attns['copy'] += [copy_attn]
            
        return hidden, outputs, attns, coverage

    def _build_rnn(
        self,
        rnn_type,
        input_size,
        hidden_size,
        num_layers,
        dropout):
        cell = utils.StackedLSTM #Multi Layered LSTM for input feed
        if rnn_type == 'GRU':
            cell = utils.StackedGRU #Multi Layered GRU for input feed
        return cell(num_layers, input_size, hidden_size, dropout)

class Seq2SeqAttention(nn.Module):
    """
    encoder + decoder model
    """
    def __init__(self, encoder, decoder):
        super(Seq2SeqAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.counter = 0
    
    def forward(self, src, tgt, lengths):
        tgt = tgt[:-1] #Excluding last target from inputs
        enc_hidden, context = self.encoder(src, lengths)
        enc_state = self.decoder.init_decoder_state(src, context, enc_hidden)
        #be careful that dec_state is required in case of truncated bptt.
        out, dec_state, attns, predictions = self.decoder(tgt, context, enc_state, context_lengths = lengths)
        return out, attns, dec_state, predictions
