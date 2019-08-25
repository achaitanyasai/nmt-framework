import random

import torch
import numpy as np
from .. import base_config as config

import math

torch.backends.cudnn.deterministic = True

random.seed(3435)
np.random.seed(3435)
torch.manual_seed(3435)
torch.cuda.manual_seed_all(3435)
torch.cuda.manual_seed(3435)

word_to_ix = {"hello": 0, "world": 1}
embeds = torch.nn.Embedding(2, 5, max_norm=1.0)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)
print(torch.norm(hello_embed))

exit(0)

# def reverse_padded_sequence(inputs, lengths, batch_first=False):
#     if batch_first:
#         inputs = inputs.transpose(0, 1)
#     if inputs.size(1) != len(lengths):
#         raise ValueError('inputs incompatible with lengths.')
#     reversed_inputs = torch.autograd.Variable(inputs.data.clone())
#     for i, length in enumerate(lengths):
#         time_ind = torch.LongTensor(list(reversed(range(length))))
#         reversed_inputs[:length, i] = inputs[:, i][time_ind]
#     if batch_first:
#         reversed_inputs = reversed_inputs.transpose(0, 1)
#     return reversed_inputs

# p = torch.autograd.Variable(torch.rand(5, 6), requires_grad=True)
# print p

# b = p
# drop = torch.nn.Dropout(0.5)
# b = drop(b)
# print b
# print b / p

# lengths = torch.LongTensor([3, 1])

# p = torch.autograd.Variable(torch.rand(2, 6), requires_grad=True)

# print p

# a = torch.sin(p)

# a = reverse_padded_sequence(a, lengths, True)

# c = torch.sum(a)
# print c

# c.backward()
# print p.grad
