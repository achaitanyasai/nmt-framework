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

from . import GlobalAttention as GlobalAttention
from . import utils
import warnings
from structs import *

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
        self.input_feed = context.data.new(*h_size).zero_().unsqueeze(0)

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
        vars = [e.data.repeat(1, beam_size, 1)
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
            dropout = 0.4):
        
        super(encoder, self).__init__()
        self.dropout_prob = dropout
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

        self.layers = nn.ModuleList()
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
            pad_token,

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

        self.embedding_transform = nn.Linear(embedding_dim, self.hidden_size)
        self.context_transform = nn.Linear(self.hidden_size, self.hidden_size)
        self.to_scalar = nn.Linear(self.hidden_size, 1)
        self.general_attention_in = nn.Linear(self.embedding_dim, self.hidden_size, bias=False)

        self.embedding_dim1 = embedding_dim
        self.softmax = nn.Softmax(dim=-1)

        # LSTM initial parameters
        self.forward_initial_h = nn.Parameter(torch.rand(self.hidden_size), requires_grad=True).cuda()
        self.forward_initial_c = nn.Parameter(torch.rand(self.hidden_size), requires_grad=True).cuda()
        self.backward_initial_h = nn.Parameter(torch.rand(self.hidden_size), requires_grad=True).cuda()
        self.backward_initial_c = nn.Parameter(torch.rand(self.hidden_size), requires_grad=True).cuda()

        # Embedding gate to combine forward and backward embeddings
        self.embedding_gate = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.Sigmoid()
        )
        # self.embedding_dropout = nn.Dropout(0.3)
        logger.info('Not using embedding attention')

    def reverse(self, inputs, lengths, batch_first=False, requires_grad=False):
        # FIXME: fix requires_grad in reverse
        if batch_first:
            inputs = inputs.transpose(0, 1)
        if inputs.size(1) != len(lengths):
            raise ValueError('inputs incompatible with lengths.')
        reversed_inputs = inputs.clone().detach() #.requires_grad_(False) # inputs.new_tensor()
        for i, length in enumerate(lengths):
            time_ind = torch.LongTensor(list(reversed(range(length))))
            reversed_inputs[:length, i] = inputs[:, i][time_ind]
        if batch_first:
            reversed_inputs = reversed_inputs.transpose(0, 1)
        return reversed_inputs

    def combine_embeddings(self, pred_emb_forward, pred_emb_backward):
        s_len, n_batch, _ = pred_emb_forward.shape
        a = torch.cat([pred_emb_forward, pred_emb_backward], dim=2)
        a = a.contiguous().view(s_len * n_batch, -1)
        gate = self.embedding_gate(a).view(s_len, n_batch, -1)
        pred_emb_forward = pred_emb_forward.mul(gate)
        pred_emb_backward = pred_emb_backward.mul(-gate.sub(1))
        pred_emb_forward = pred_emb_forward.add(pred_emb_backward)
        # pred_emb_forward = self.embedding_dropout(pred_emb_forward)
        return pred_emb_forward

    def forward(self, seq_input, input_hashes, lengths = None, hidden = None):
        self.rnn.flatten_parameters()
        s_len, n_batch = seq_input.shape[0], seq_input.shape[1]
        final_hidden_h = []
        final_hidden_c = []
        penalty = None
        # word_embeddings = None
        pred_emb = None
        for layer_num in range(self.num_layers):
            if layer_num == 0:
                pass

            apply_dropout = True
            if layer_num + 1 == self.num_layers:
                apply_dropout = False
            if self.bidirectional:
                hidden_forward, outputs_forward, _penalty, pred_emb_forward = self._forward(seq_input, None, lengths, (self.forward_initial_h, self.forward_initial_c), layer_num, True, apply_dropout)
                requires_grad = True
                if layer_num == 0:
                    requires_grad = False
                    if penalty is None:
                        penalty = _penalty
                    else:
                        penalty = penalty + _penalty
                input_reverse = self.reverse(seq_input, lengths, requires_grad=requires_grad)
                hidden_backward, outputs_backward, _penalty, pred_emb_backward = self._forward(input_reverse, None, lengths, (self.backward_initial_h, self.backward_initial_c), layer_num, False, apply_dropout)
                if layer_num == 0:
                    if penalty is None:
                        penalty = _penalty
                    else:
                        penalty = penalty + _penalty
                    assert pred_emb_forward is not None
                    assert pred_emb_backward is not None

                    pred_emb_backward = self.reverse(pred_emb_backward, lengths, requires_grad=True)
                    # TODO: What if outputs_forward and outputs_backward are same? Think of better approach for combining forward and backward representations
                    # TODO: combine_embeddings() is one way to combine using gates. Think of other ways.
                    # pred_emb = torch.cat([pred_emb_forward, pred_emb_backward], dim=-1)

                    pred_emb = self.combine_embeddings(pred_emb_forward, pred_emb_backward)
                    assert pred_emb.shape == (s_len, n_batch, self.hidden_size * (1 + self.bidirectional))

                outputs_backward = self.reverse(outputs_backward, lengths, requires_grad=True)
                outputs = torch.cat([outputs_forward, outputs_backward], 2)
                seq_input = outputs
                final_hidden_h.append(hidden_forward[0])
                final_hidden_h.append(hidden_backward[0])

                final_hidden_c.append(hidden_forward[1])
                final_hidden_c.append(hidden_backward[1])
            else:
                raise Exception('Please use bidirectional encoder')

        final_hidden_h = torch.cat(final_hidden_h, 0)
        final_hidden_c = torch.cat(final_hidden_c, 0)
        final_hidden = (final_hidden_h, final_hidden_c)

        assert seq_input.shape == pred_emb.shape

        return final_hidden, seq_input, penalty, pred_emb

    def _forward(self, input, input_hashes, lengths, hidden, layer_num=0, forward_rnn=True, apply_dropout=False):
        predicted_embeddings = None
        if layer_num == 0:
            s_len, n_batch = input.size()
            predicted_embeddings = torch.zeros((s_len, n_batch, self.embedding_dim), requires_grad=True).cuda()
        else:
            s_len, n_batch, emb_dim = input.size()
            assert emb_dim == self.hidden_size * (1 + self.bidirectional)

        j = n_batch - 1
        if hidden == None:
            raise Exception("Please use randomly initialized hidden states instead of zero's")
            # h_0 = torch.zeros(1, n_batch, self.hidden_size, requires_grad=True).cuda()
            # c_0 = torch.zeros(1, n_batch, self.hidden_size, requires_grad=True).cuda()
        else:
            h_0 = hidden[0].repeat(n_batch, 1).unsqueeze(0)
            c_0 = hidden[1].repeat(n_batch, 1).unsqueeze(0)

        outputs = []
        final_h = torch.zeros(1, n_batch, self.hidden_size, requires_grad=True).cuda()
        final_c = torch.zeros(1, n_batch, self.hidden_size, requires_grad=True).cuda()
        penalty = None

        for i in range(s_len):
            while j >= 0 and lengths[j] < i + 1:
                j -= 1
            cur_input = input[i][:j + 1]
            h_0 = h_0[:, :j + 1, :]
            c_0 = c_0[:, :j + 1, :]
            if layer_num == 0:
                cur_input, cur_penalty = self.embedding_attention(cur_input, h_0)
                predicted_embeddings[i][:j + 1] = cur_input
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
            _output.append(cur_output)
            _output = torch.stack(_output)
            h_1.append(h_j)
            c_1.append(c_j)

            h_0 = torch.stack(h_1)
            c_0 = torch.stack(c_1)

            final_h[:, :j + 1, :] = h_0[:, :j + 1, :]
            final_c[:, :j + 1, :] = c_0[:, :j + 1, :]
            outputs = [_output] + outputs

        final_outputs = []
        for output in outputs:
            target = torch.zeros(1, n_batch, self.hidden_size, requires_grad=True).cuda()
            target[:, :output.shape[1], :] = output[-1:, :]
            final_outputs = [target] + final_outputs

        final_outputs = torch.cat(final_outputs, 0)
        return (final_h, final_c), final_outputs, penalty, predicted_embeddings

    def embedding_attention(self, cur_input, context):
        penalty = None

        emb1 = self.embeddings1(cur_input)
        emb2 = self.embeddings2(cur_input)
        emb3 = self.embeddings3(cur_input)

        penalty = torch.sum(F.relu(F.cosine_similarity(emb1, emb2) - 0.3))
        penalty = penalty + torch.sum(F.relu(F.cosine_similarity(emb1, emb3) - 0.3))
        penalty = penalty + torch.sum(F.relu(F.cosine_similarity(emb2, emb3) - 0.3))

        context = context.squeeze(0)

        # Bahdanau attention:
        a = self.to_scalar(torch.tanh(self.embedding_transform(emb1) + self.context_transform(context)))
        b = self.to_scalar(torch.tanh(self.embedding_transform(emb2) + self.context_transform(context)))
        c = self.to_scalar(torch.tanh(self.embedding_transform(emb3) + self.context_transform(context)))

        scores = self.softmax(torch.cat([a, b, c], dim=1)).unsqueeze(1)
        emb = torch.cat([emb1.unsqueeze(2), emb2.unsqueeze(2), emb3.unsqueeze(2)], dim=2).transpose(1, 2)
        emb = torch.bmm(scores, emb).squeeze(1)
        # TODO: penalty: i.e., Make the vectors diverse?
        return emb, penalty

        # Luong General Attention:
        # a = torch.bmm(self.general_attention_in(emb1).unsqueeze(1), context.unsqueeze(2)).squeeze(2)
        # b = torch.bmm(self.general_attention_in(emb2).unsqueeze(1), context.unsqueeze(2)).squeeze(2)
        # c = torch.bmm(self.general_attention_in(emb3).unsqueeze(1), context.unsqueeze(2)).squeeze(2)
        # scores = self.softmax(torch.cat([a, b, c], dim=1)).unsqueeze(1)
        # emb = torch.cat([emb1.unsqueeze(2), emb2.unsqueeze(2), emb3.unsqueeze(2)], dim=2).transpose(1, 2)
        # emb = torch.bmm(scores, emb).squeeze(1)
        # return emb, penalty



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

        # self.encoder_context_transform = nn.Sequential(
        #     nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        # )

        self.context_transform = nn.Linear(self.hidden_size, self.hidden_size)
        self.src_embedding_transform = nn.Linear(self.hidden_size, self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.context_dropout = nn.Dropout(0.2)
    
    def forward(
        self,
        input,
        input_charngrams,
        context,
        encoder_embeddings,
        state,
        context_lengths):

        assert isinstance(state, RNNDecoderState)
        input_len, input_batch = input.size()
        context_len, context_batch, _ = context.size()

        utils.aeq(input_batch, context_batch)

        hidden, outputs, attns, coverage = self._run_forward_pass(input, input_charngrams, context, encoder_embeddings, state, context_lengths = context_lengths)

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

    def combine_embeddings(self, pred_emb_forward, pred_emb_backward):
        s_len, n_batch, _ = pred_emb_forward.shape
        pred_emb_forward = pred_emb_forward.contiguous().view(s_len * n_batch, -1)
        pred_emb_backward = pred_emb_backward.contiguous().view(s_len * n_batch, -1)
        gate = self.sigmoid(self.context_transform(pred_emb_forward) + self.src_embedding_transform(pred_emb_backward)).view(s_len, n_batch, -1)
        pred_emb_forward = pred_emb_forward.contiguous().view(s_len, n_batch, -1)
        pred_emb_backward = pred_emb_backward.contiguous().view(s_len, n_batch, -1)
        pred_emb_forward = pred_emb_forward.mul(gate)
        pred_emb_backward = pred_emb_backward.mul(-gate.sub(1))
        pred_emb_forward = pred_emb_forward.add(pred_emb_backward)
        # pred_emb_forward = self.context_dropout(pred_emb_forward)
        return pred_emb_forward

        
class decoder(decoderBase):
    '''
    Input feed decoder
    '''
    def _run_forward_pass(self, input, input_hashes, context, encoder_embeddings, state, context_lengths = None):
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

        # context = torch.cat([context, encoder_embeddings], dim=2)
        context = self.combine_embeddings(context, encoder_embeddings)
        # b_size, s_len, dim = context.shape
        # context = context.view(b_size * s_len, -1)
        # context = self.encoder_context_transform(context)
        # context = context.view(b_size, s_len, -1)

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
                raise NotImplementedError("Learn about coverage attention first")
                coverage = coverage + attn \
                    if coverage is not None else attn
                attns['coverage'] += [coverage]
            
            if self._copy:
                raise NotImplementedError("Learn about copy attention first")
                # _, copy_attn = self.copy_attn(output, context.transpose(0, 1))
                # attns['copy'] += [copy_attn]
        
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
    
    def forward(self, batch):
        src = batch.src
        src_lens = batch.src_lens

        tgt = batch.tgt
        tgt = tgt[:-1]

        enc_hidden, context, penalty, encoder_embeddings = self.encoder(src, None, src_lens)
        encoderOutputs = EncoderOutputs(enc_hidden, context, encoder_embeddings)

        enc_state = self.decoder.init_decoder_state(src, context, enc_hidden)

        out, dec_state, attns, predictions = self.decoder(tgt, None, encoderOutputs.outputs, encoderOutputs.encoder_embeddings, enc_state, context_lengths=src_lens)
        decoderOutputs = DecoderOutputs(predictions, penalty, out, dec_state)

        return decoderOutputs
