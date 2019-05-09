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

import GlobalAttention
import utils
import warnings

class DecoderState(object):
    def detach(self):
        for h in self._all:
            if h is not None:
                h.detach()
    
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
        vars = [Variable(e.data.repeat(1, beam_size, 1))
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
            pad_token,
            ngrams_vocab_size,
            dropout = 0.4,
            ngram_pad_token=1): 
        
        super(encoder, self).__init__()
        self.no_pack_padded_seq = False
        self.vocab_size = vocab_size

        self.num_layers = num_layers
        self.pad_token = pad_token
        self.embedding_dim = embedding_dim
        self.num_directions = 1 + bidirectional
        self.bidirectional = bidirectional

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
            embedding_dim,
            pad_token
        )

    def forward(self, input, input_hashes, lengths = None, hidden = None):
        self.rnn.flatten_parameters()
        s_len, n_batch = input.size()[0], input.size()[1]
        if lengths is not None:
            n_batch_, = lengths.size()
            utils.aeq(n_batch, n_batch_)
        
        emb = self.embeddings(input)

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

        return hidden_t, outputs, None

class CustomEncoder(nn.Module):
    def __init__(
            self,
            rnn_type,
            bidirectional,
            num_layers,
            hidden_size,
            vocab_size,
            embedding_dim,
            pad_token,
            ngrams_vocab_size,
            dropout,
            ngram_pad_token=1):
        super(CustomEncoder, self).__init__()
        #f = open('/home/chaitanya/tmp.txt', 'w')
        #f.close()

        self.no_pack_padded_seq = False
        self.vocab_size = vocab_size

        self.num_layers = num_layers
        self.pad_token = pad_token

        self.embedding_dim = 500#embedding_dim# + 128
        self.num_directions = 1 + bidirectional
        self.bidirectional = bidirectional
        assert hidden_size % self.num_directions == 0
        self.hidden_size = hidden_size // self.num_directions

        self.vector_space = 2 
        self.layers = nn.ModuleList()
        self.dropout_prob = dropout
        if self.dropout_prob == 0.0:
            warnings.warn('Dropout is zero')
        self.dropout = nn.Dropout(self.dropout_prob)
        input_size = embedding_dim
        
        for i in range(num_layers):
            layer = nn.ModuleList()
            layer.append(nn.LSTMCell(input_size, self.hidden_size))
            if self.bidirectional:
                layer.append(nn.LSTMCell(input_size, self.hidden_size))
            self.layers.append(layer)
            input_size = self.hidden_size * (1 + self.bidirectional)
    
        self.embeddings1 = nn.Embedding(
            vocab_size,
            embedding_dim,
            pad_token
        )

        self.embeddings2 = nn.Embedding(
            vocab_size,
            embedding_dim,
            pad_token
        )

        self.embeddings3 = nn.Embedding(
            vocab_size,
            embedding_dim,
            pad_token
        )

        self.embeddings4 = nn.Embedding(
            vocab_size,
            embedding_dim,
            pad_token
        )

        self.embeddings5 = nn.Embedding(
            vocab_size,
            embedding_dim,
            pad_token
        )
        

        # self.reform_h = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # )

        # self.reform_emb = nn.Sequential(
        #     nn.Linear(embedding_dim, self.hidden_size, bias=False)
        # )

        # self.get_weights = nn.Sequential(
        #     nn.Linear(self.hidden_size + embedding_dim, self.hidden_size + embedding_dim),
        #     nn.Tanh(),
        #     nn.Linear(self.hidden_size + embedding_dim, 1)
        # )

        self.embedding_l1 = nn.Linear(embedding_dim, self.hidden_size)
        self.context_h1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.to_scalar = nn.Linear(self.hidden_size, 1)

        self.embedding_dim1 = embedding_dim
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.file_log_counter = 0
        f = open('/home/chaitanya/tmp.txt', 'w')
        f.close()
    
    def reverse(self, inputs, lengths, batch_first=False):
        if batch_first:
            inputs = inputs.transpose(0, 1)
        if inputs.size(1) != len(lengths):
            raise ValueError('inputs incompatible with lengths.')
        reversed_inputs = Variable(inputs.data.clone())
        for i, length in enumerate(lengths):
            time_ind = torch.LongTensor(list(reversed(range(length))))
            reversed_inputs[:length, i] = inputs[:, i][time_ind]
        if batch_first:
            reversed_inputs = reversed_inputs.transpose(0, 1)
        return reversed_inputs
    
    def forward(self, input, input_hashes, lengths = None, hidden = None):
        s_len, n_batch = input.size()
        a = input.data.cpu().tolist()
        if lengths is not None:
            n_batch_, = lengths.size()
            utils.aeq(n_batch, n_batch_)
        final_hidden_h = []
        final_hidden_c = []
        penalty = None
        word_embeddings = None
        for layer_num in range(self.num_layers):
            if layer_num == 0:
                # input = self.embeddings(input)
                pass

            apply_dropout = True
            if layer_num + 1 == self.num_layers:
                apply_dropout = False
            if self.bidirectional:
                hidden_forward, outputs_forward, _penalty = self._forward(input, None, lengths, hidden, layer_num, True, apply_dropout)
                if layer_num == 0:
                    if penalty is None:
                        penalty = _penalty
                    else:
                        penalty = penalty + _penalty
                input_reverse = self.reverse(input, lengths)        
                # input_hashes_reverse = self.reverse(input_hashes.transpose(0, 1), lengths).transpose(0, 1)

                hidden_backward, outputs_backward, _penalty = self._forward(input_reverse, None, lengths, hidden, layer_num, False, apply_dropout)
                if layer_num == 0:
                    if penalty is None:
                        penalty = _penalty
                    else:
                        penalty = penalty + _penalty

                outputs_backward = self.reverse(outputs_backward, lengths)
                # if layer_num == 0:
                    # embedding_context_backward = self.reverse(embedding_context_backward, lengths)
                    # word_embeddings = embedding_context_forward + embedding_context_backward
                outputs = torch.cat([outputs_forward, outputs_backward], 2)
                input = outputs
                final_hidden_h.append(hidden_forward[0])
                final_hidden_h.append(hidden_backward[0])

                final_hidden_c.append(hidden_forward[1])
                final_hidden_c.append(hidden_backward[1])
            else:
                raise Exception('Please use bidirectional encoder for better performance.')
                hidden_forward, outputs_forward, _penalty = self._forward(input, input_hashes, lengths, hidden, layer_num, True, apply_dropout)                
                if layer_num == 0:
                    if penalty is None:
                        penalty = _penalty
                    else:
                        penalty = penalty + _penalty        
                input = outputs_forward
                final_hidden_h.append(hidden_forward[0])
                final_hidden_c.append(hidden_forward[1])

        final_hidden_h = torch.cat(final_hidden_h, 0)
        final_hidden_c = torch.cat(final_hidden_c, 0)
        final_hidden = (final_hidden_h, final_hidden_c)
        # assert penalty is not None
        return final_hidden, input, penalty

    def embedding_attention(self, cur_input, context):
        penalty = None
        b_size = cur_input.shape[0]
        emb1 = self.embeddings1(cur_input)
        emb2 = self.embeddings2(cur_input)
        emb3 = self.embeddings3(cur_input)
        emb4 = self.embeddings4(cur_input)
        emb5 = self.embeddings5(cur_input)

        context = context.squeeze(0)

        emb_a = self.to_scalar(torch.tanh(self.embedding_l1(emb1) + self.context_h1(context)))
        emb_b = self.to_scalar(torch.tanh(self.embedding_l1(emb2) + self.context_h1(context)))
        emb_c = self.to_scalar(torch.tanh(self.embedding_l1(emb3) + self.context_h1(context)))
        emb_d = self.to_scalar(torch.tanh(self.embedding_l1(emb4) + self.context_h1(context)))
        emb_e = self.to_scalar(torch.tanh(self.embedding_l1(emb5) + self.context_h1(context)))

        sc = self.softmax(torch.cat([emb_a, emb_b, emb_c, emb_d, emb_e], dim = 1)).unsqueeze(1)
        emb = torch.cat([emb1.unsqueeze(2), emb2.unsqueeze(2), emb3.unsqueeze(2), emb4.unsqueeze(2), emb5.unsqueeze(2)], dim=2).transpose(1, 2)
        res = torch.bmm(sc, emb).squeeze(1)
        return res, penalty

    def _forward(self, input, input_hashes, lengths = None, hidden = None, layer_num=0, forward_rnn=True, apply_dropout=False):
        #TODO: In case of dropout, this is not equal
        # to inbuilt LSTM. Inbuilt LSTM is using masking 
        # at each time step. Look into it.
        #TODO: Precision errors between LSTMCell() and LSTM()

        if layer_num == 0:
            # s_len, n_batch = input.size()
            # charngrams_emb = self.apply_cnn_charngrams(input_hashes, s_len, n_batch)
            # input = self.embeddings(input)

            # assert input.shape[0] == charngrams_emb.shape[0]
            # assert input.shape[1] == charngrams_emb.shape[1]
            # input = self.fusion(input, charngrams_emb)
            s_len, n_batch = input.size()
            # assert(embedding_dim == self.embedding_dim)
            pass
        
        else:
            s_len, n_batch, embedding_dim = input.size()
            assert(embedding_dim == self.hidden_size * (1 + self.bidirectional))
        
        j = n_batch - 1
        if hidden == None:
            h_0 = []
            c_0 = []
            h_0 = Variable(torch.zeros(1, n_batch, self.hidden_size), requires_grad=False).cuda()
            c_0 = Variable(torch.zeros(1, n_batch, self.hidden_size), requires_grad=False).cuda()
        
        outputs = []
        final_h = Variable(torch.zeros(1, n_batch, self.hidden_size)).cuda()
        final_c = Variable(torch.zeros(1, n_batch, self.hidden_size)).cuda()
        penalty = None

        for i in range(s_len):
            while j >= 0 and lengths[j] < i + 1:
                j -= 1
            cur_input = input[i][:j + 1]
            h_0 = h_0[:,:j + 1,:]
            c_0 = c_0[:,:j + 1,:]
            if layer_num == 0:
                cur_input, cur_penalty = self.embedding_attention(cur_input, c_0)
                if penalty is None:
                    penalty = cur_penalty
                else:
                    penalty = penalty + cur_penalty
            h_1 = []
            c_1 = []
            _output = []
            layer_id = 0
            if forward_rnn == False:
                layer_id = 1
            h_j, c_j = self.layers[layer_num][layer_id](cur_input, (h_0[0], c_0[0]))
            cur_output = h_j
            if apply_dropout and self.dropout_prob > 0:
                cur_output = self.dropout(cur_output)
            _output += [cur_output]
            h_1 += [h_j]
            c_1 += [c_j]
            _output = torch.stack(_output)
            h_0 = torch.stack(h_1)
            c_0 = torch.stack(c_1)

            final_h[:,:j + 1,:] = h_0[:,:j + 1,:]
            final_c[:,:j + 1,:] = c_0[:,:j + 1,:]
            outputs = [_output] + outputs
        final_outputs = []
        for output in outputs:
            target = Variable(torch.zeros(1, n_batch, self.hidden_size)).cuda()
            target[:,:output.shape[1],:] = output[-1:,:]
            final_outputs = [target] + final_outputs
        
        final_outputs = torch.cat(final_outputs, 0)
        return (final_h, final_c), final_outputs, penalty

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
        ngrams_vocab_size,
        embedding_dim,
        pad_token,
        attn_type = 'general',
        coverage_attn = False,
        copy_attn = False,
        context_gate = None,
        dropout = 0.0,
        ngram_pad_token = 1):
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
            self.embedding_dim + self.hidden_size,
            self.hidden_size,
            self.num_layers,
            dropout
        )

        #FIXME: think about using Tanh() here.
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
        input_charngrams,
        context,
        state,
        context_lengths):

        assert isinstance(state, RNNDecoderState)
        input_len, input_batch = input.size()
        context_len, context_batch, _ = context.size()

        utils.aeq(input_batch, context_batch)

        hidden, outputs, attns, coverage = self._run_forward_pass(input, input_charngrams, context, state, context_lengths = context_lengths)

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
        # return h
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, src, context, enc_hidden):
        if isinstance(enc_hidden, tuple): #LSTM
            return RNNDecoderState(context, self.hidden_size, tuple([self._fix_enc_hidden(enc_hidden[i]) for i in range(len(enc_hidden))]))
        else: #GRU
            return RNNDecoderState(context, self.hidden_size, self._fix_enc_hidden(enc_hidden))
        
class decoder(decoderBase):
    '''
    Input feed decoder
    '''
    def _run_forward_pass(self, input, input_hashes, context, state, context_lengths = None):
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
        
        # charngrams_emb = self.apply_cnn_charngrams(input_hashes, input_len, input_batch)        
        emb = self.embeddings(input)
        # assert(emb.shape[0] == charngrams_emb.shape[0])
        # assert(emb.shape[1] == charngrams_emb.shape[1])

        # emb = torch.cat([emb, charngrams_emb], 2)

        s_len, batch_size, embedding_dim = emb.size()
        assert(embedding_dim == self.embedding_dim)
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
    
    def forward(self, src, tgt, lengths, src_hashes, tgt_hashes):

        tgt = tgt[:-1] #Excluding last target from inputs
        #tgt_hashes = tgt_hashes.transpose(0, 1)
        #tgt_hashes = tgt_hashes[:-1]
        #tgt_hashes = tgt_hashes.transpose(0, 1)
        enc_hidden, context, penalty = self.encoder(src, src_hashes, lengths)
        enc_state = self.decoder.init_decoder_state(src, context, enc_hidden)
        
        out, dec_state, attns, predictions = self.decoder(tgt, tgt_hashes, context, enc_state, context_lengths = lengths)
        return out, attns, dec_state, predictions, penalty
