import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from structs import *

class Encoder(nn.Module):

    def __init__(self,
                 bidirectional=True,
                 num_layers=1,
                 hidden_size=100,
                 embedding_dim=100,
                 pad_token=1,
                 dropout=0.0):

        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.pad_token = pad_token
        self.dropout = dropout
        self.num_directions = 1 + self.bidirectional

        self.dropOutLayer = nn.Dropout(dropout)
        assert self.hidden_size % self.num_directions == 0
        self.hidden_size = hidden_size // self.num_directions

        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

    def forward(self, input, lengths):
        self.rnn.flatten_parameters()
        s_len, n_batch, emb_dim = input.shape[0], input.shape[1], input.shape[2]
        if lengths is not None:
            n_batch_, = lengths.size()
            assert n_batch == n_batch_

        emb = input

        packed_emb = emb
        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, None)
        if lengths is not None:
            outputs = unpack(outputs)[0]

        ret = EncoderOutputs(hidden_t=hidden_t, outputs=outputs)
        return ret

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.output_size = output_size
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, output, hidden):
        output, hidden = self.rnn(output, hidden)
        b_size = output.shape[1]
        output = output.view(-1, self.hidden_size)
        output = self.softmax(self.out(output))
        output = output.view(-1, b_size, self.output_size)
        ret = DecoderOutputs(hidden=hidden, predictions=output)
        return ret

class Model(nn.Module):
    def __init__(self, encoder, decoder, vocab_size,
                 embedding_dim, pad_token):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            pad_token
        )

    def forward(self, batch):
        src = batch.src
        src_lens = batch.src_lens

        tgt = batch.tgt
        tgt_lens = batch.tgt_lens

        tgt = tgt[:-1]

        src = self.embedding(src)
        encoderOutputs = self.encoder(src, src_lens)
        encoderOutputs.combine_forward_backward()

        tgt = self.embedding(tgt)
        decoderOutputs = self.decoder(tgt, encoderOutputs.hidden_t)
        return decoderOutputs