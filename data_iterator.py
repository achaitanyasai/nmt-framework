#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Data iterators
'''

import logging
import random

import numpy as np
import torch
from torch.autograd import Variable

import util
import sys
import operator
logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('./data/training_logs.txt')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

class Lang(object):
    '''
    This class contains the data related to a language.
    '''
    #TODO:
    #1) Refactor
    #2) Add character level features
    #3) Sort vocab based on freq
    def __init__(self, fname, maxlen = 50, max_word_len_allowed = 30, 
                max_vocab_size = 50000, langtype = 'source', chars = False, 
                verbose = True, ignore_too_many_unknowns = False, data_type='train', 
                max_number_of_sentences_allowed = 100000000000):
        if verbose:
            logger.info('Reading %s (%s language)' % (fname, langtype))
        self.ignore_too_many_unknowns = ignore_too_many_unknowns
        self.fname = fname
        self.langtype = langtype
        self.unkcount = 0
        self.raw_data = util.fread(self.fname).lower().split('\n')[:max_number_of_sentences_allowed]
        self.max_number_of_sentences_allowed = max_number_of_sentences_allowed
        self.maxlen = maxlen
        self.max_word_len_allowed = max_word_len_allowed
        self.max_vocab_size = max_vocab_size
        self.verbose = verbose
        self.data_type = data_type
        
        assert self.data_type in ['valid', 'train', 'test']
        assert self.langtype in ['target', 'source']
        
        # 'CUNK': 0, 'CPAD': 1, 'SOC': 2, 'EOC': 3
        # Other tokens(SOS, EOS, PAD, UNK) are from word level
        self.hash2idx = {'CUNK' : 0, 'CPAD' : 1, 'SOC': 2, 'EOC': 3, 'SOS': 4, 'EOS': 5, 'PAD': 6, 'UNK': 7}
        self.idx2hash = {0 : 'CUNK', 1 : 'CPAD', 2 : 'SOC', 3 : 'EOC', 4: 'SOS', 5: 'EOS', 6: 'PAD', 7 : 'UNK'}
        self.n_hashes = 8

        if(langtype == 'source'):
            #'UNK': 0, 'PAD': 1
            self.word2idx = {'UNK' : 0, 'PAD' : 1}
            self.word2count = {'UNK' : 1, 'PAD' : 1}
            self.idx2word = {0 : 'UNK', 1 : 'PAD'}
            self.n_words = 2
        else:
            #'UNK': 0, 'PAD': 1, 'SOS: 2, 'EOS': 3
            self.word2idx = {'SOS' : 2, 'EOS' : 3, 'UNK' : 0, 'PAD' : 1}
            self.word2count = {'SOS' : 1, 'EOS' : 1, 'UNK' : 1, 'PAD' : 1}
            self.idx2word = {0 : 'UNK', 1 : 'PAD', 2 : 'SOS', 3 : 'EOS'}
            self.n_words = 4

        self.normalized_data = []
        self.n_sentences = 0
        self.max_sent_len = maxlen
        self.langtype = langtype
        self.sentidx = 0
        self.max_word_len = 0

        self.build_vocab()
        L = len(self.raw_data)        

        for j, _ in enumerate(self.raw_data):
            sentence = _.strip().split()[:self.maxlen]
            self.n_sentences += 1
            for word in sentence:
                self.addWord(word)
            self.normalized_data.append(sentence)            
        del self._d
        del self.raw_data
        
    def __iter__(self):
        return self
    
    def reset(self):
        self.sentidx = 0
    
    def next(self):
        if(self.sentidx >= self.n_sentences):
            self.reset()
            return self.next()
        else:
            self.sentidx += 1
            return self.normalized_data[self.sentidx - 1]
    
    def addWord(self, word):
        if self.langtype == 'source':
            self.addHash(word, 2)
        try:
            tmp = self._d[word]
        except KeyError:
            self.word2count[word] = 0
            return False
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.word2count[word] = 1
            self.idx2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
        if self.langtype == 'target':
            self.addHash(word, 2)
        return True

    def addHash(self, word, ngram):
        word = util.combine_secondary_symbols(list(word.decode('utf-8')))[:self.max_word_len_allowed]
        self.max_word_len = max(len(word), self.max_word_len)
        hashes = ['SOC'] + word + ['EOC']
        i = 0
        while i < len(hashes):
            if(i + ngram - 1 >= len(hashes)):
                break
            cur = ''.join(hashes[i : i + ngram])
            if cur not in self.hash2idx:
                self.hash2idx[cur] = self.n_hashes
                self.idx2hash[self.n_hashes] = cur
                self.n_hashes += 1
            i += 1

    def build_vocab(self):
        _d = {}
        self._d = {}
        for _ in self.raw_data:
            sentence = _.strip().split()[:self.max_sent_len]
            for word in sentence:
                try:
                    _d[word] += 1
                except KeyError:
                    _d[word] = 1
        _sd = sorted(_d.items(), key=operator.itemgetter(1), reverse = True)
        num_unknown_tokens = max(0, len(_sd) - self.max_vocab_size)
        self.unkcount = num_unknown_tokens
        if self.verbose:
            logger.info('Unknown tokens: %d (%.2f%%)' % (num_unknown_tokens, (num_unknown_tokens * 100.0) / len(_sd)))
            logger.info('Vocab size: %d (%.2f%%)' % (len(_sd) - num_unknown_tokens, 100.0 - (num_unknown_tokens * 100.0) / len(_sd)))
            sys.stdout.flush()
        if self.data_type == 'train':
            if not self.ignore_too_many_unknowns and (num_unknown_tokens * 100.0) / len(_sd) >= 20.00:
                if self.max_vocab_size >= 100000:
                    logger.info('Ignoring too many unknowns as vocabulary size is more than 100K')
                else:
                    raise Exception('Too many unknowns. Please increase vocabulary size\n')
            if not self.ignore_too_many_unknowns and (num_unknown_tokens * 100.0) / len(_sd) <= 2.00:
                raise Exception('Very few unknowns. Please decrease vocabulary size\n')
        if self.langtype == 'source':
            v = 2
        else:
            v = 4
        for i in _sd[:self.max_vocab_size - v]:
            self._d[i[0]] = i[1]
        del _sd
        del _d
            
    def sentence2Sequence(self, sentence, add_sos = True, add_eos = True, maxlen = 0, pad = False, max_word_len = 0):
        seq_words = []
        seq_hashes = []
        maxlen = maxlen + (add_sos) + (add_eos)

        if(isinstance(sentence, str)):
            orgsentence = sentence.split()
            sentence = sentence.split()
        else:
            for i in sentence:
                assert i != ''
            orgsentence = ' '.join(sentence)
            assert(isinstance(sentence, list))
        
        assert(maxlen > 0)

        if add_sos:
            if sentence[0] != 'SOS' :
                sentence = ['SOS'] + sentence
        else:
            if self.langtype == 'target':
                raise Exception('SOS is required for target language')
            if sentence[0] == 'SOS':
                assert False
                sentence = sentence[1::]
                
        if add_eos:
            if sentence[-1] != 'EOS':
                sentence = sentence + ['EOS']
        else:
            if self.langtype == 'target':
                raise Exception('EOS is required for target language')
            if sentence[-1] == 'EOS':
                assert False
                sentence = sentence[:-1]
        c = 0
        newsent = []
        for word in sentence:
            if self.langtype == 'source':
                seq_hashes.append(self.word2HashSequence(word, None, 2))
            try:
                tmp = self.word2idx[word]
                newsent.append(word)
                seq_words.append(tmp)
                if self.langtype == 'target':
                    seq_hashes.append(self.word2HashSequence(word, None, 2))
            except KeyError:
                seq_words.append(self.word2idx['UNK'])
                newsent.append('UNK')
                if self.langtype == 'target':
                    seq_hashes.append(self.word2HashSequence('UNK', None, 2))
        if pad:
            while len(seq_words) < maxlen:
                seq_words = seq_words + [self.word2idx['PAD']]
                seq_hashes.append(self.word2HashSequence('PAD', None, 2))
        return seq_words, seq_hashes
    
    def word2HashSequence(self, word, max_word_len, ngram):
        seq = []
        if max_word_len == None:
            max_word_len = self.max_word_len + 2
        else:
            max_word_len = max_word_len + 2
            raise Exception('Uncomment this line, if you are sure that max_word_len is based on batched data. That is, using character RNN.')
        max_word_len = max_word_len - 2 + 1
        if word in ['UNK', 'SOS', 'EOS', 'PAD']:
            if word == 'UNK':
                assert self.langtype == 'target'
            seq.append(self.hash2idx[word])
        else:
            word = util.combine_secondary_symbols(list(word.decode('utf-8')))[:self.max_word_len_allowed]
            hashes = ['SOC'] + word + ['EOC']
            i = 0
            while i < len(hashes):
                if(i + ngram - 1 >= len(hashes)):
                    break
                cur = ''.join(hashes[i : i + ngram])
                try:
                    seq.append(self.hash2idx[cur])
                except KeyError:
                    seq.append(self.hash2idx['CUNK'])
                i += 1
        while len(seq) < max_word_len:
            seq.append(self.hash2idx['CPAD'])
        assert(len(seq) == max_word_len)
        return seq
    
    def sequence2Hashes(self, seq):
        res = []
        if isinstance(seq, list):
            for i in seq:
                if isinstance(i, int):
                    res.append(self.idx2hash[i])    
                else:
                    raise Exception('Unknown instance: %s' % (str(type(i))))
        else:
            raise Exception('Unknown instance: %s' % (str(type(seq))))
        return res

    def sequence2Sentence(self, seq):
        res = []
        if isinstance(seq, list):
            for i in seq:
                if isinstance(i, int):
                    res.append(self.idx2word[i])    
                else:
                    raise Exception('Unknown instance: %s' % (str(type(i))))
        else:
            raise Exception('Unknown instance: %s' % (str(type(seq))))
        return res
    

class BatchData(object):
    '''
    A container for samples in a batch
    '''
    def __init__(self, src, tgt, src_len, tgt_len,
                 indices, src_raw, tgt_raw, batch_size,
                 src_hashes, tgt_hashes):
        self.indices = indices
        self.src = src  # Source sequence of shape: seq_len x batch_size
        self.tgt = tgt  # Target sequence of shape: seq_len x batch_size or None
        self.src_len = src_len #lengths of the source sequence. Shape: batch_size
        self.tgt_len = tgt_len #lengths of the target sequence. Shape: batch_size or None
        self.src_raw = src_raw #Raw source sentences
        self.tgt_raw = tgt_raw #Raw target sentences or None
        self.batch_size = batch_size #Number of samples in current batch
        self.src_hashes = src_hashes #Source hashes of shape: batch_size x seq_len x max_hash_len
        self.tgt_hashes = tgt_hashes #Target hashes of shape: batch_size x seq_len x max_hash_len or None

class dataIterator(object):
    '''
    Iterator for data.
    each call to .next() method returns a pair of sequences (source sequence, target dequence) 
    which can be fed to encoder and decoder respectively.
    '''
    def __init__(self, source_lang, target_lang, shuffle = True):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.shuffle = shuffle
        self.order = [i for i in range(self.source_lang.n_sentences)]
        self.source_lens = [len(i) for i in self.source_lang.normalized_data]
        # self.target_lens = [len(i) for i in self.target_lang.normalized_data]
        self.n_samples = self.source_lang.n_sentences
        self.reset()

    def __iter__(self):
        return self
    
    def reset(self):
        self.sentidx = 0
        if self.shuffle:
            random.shuffle(self.order)
        
    def next(self):
        if self.sentidx >= self.n_samples:
            self.reset()
            raise Exception('Stop')
        else:
            source_text = self.source_lang.normalized_data[self.order[self.sentidx]]
            if self.target_lang is not None:
                target_text = self.target_lang.normalized_data[self.order[self.sentidx]]
            else:
                target_text = None
            self.sentidx += 1
            #Returns source raw text, target raw text, corresponding sentence index
            return source_text, target_text, self.sentidx - 1
    
    def next_batch(self, batch_size, max_length, source_language_model = None, target_language_model = None, return_texts = False):
        #TODO: Implement return_texts feature
        #TODO: Refactor it.
        source_max_len = 0
        target_max_len = 0

        source_lines_unsorted = []
        target_lines_unsorted = []
        sentences_indices_unsorted = []
        source_lens_unsorted = []
        target_lens_unsorted = []

        samples = 0
        for _ in range(batch_size):
            try:
                source_line, target_line, sentidx = self.next()
                samples += 1
            except Exception as e:
                break
            source_lines_unsorted.append(source_line)
            source_lens_unsorted.append(len(source_line))
            sentences_indices_unsorted.append(sentidx)

            if target_line is not None:
                target_lines_unsorted.append(target_line)
                target_lens_unsorted.append(len(target_line))
        
        source_max_len = max(source_lens_unsorted)
        if len(target_lens_unsorted) > 0:
            target_max_len = max(target_lens_unsorted)
        else:
            target_max_len = 0

        #Sorting samples in descending order of source lengths
        indices = np.argsort(np.array(source_lens_unsorted))
        indices = indices[::-1]

        sentences_indices = []
        source_lines = [source_lines_unsorted[i] for i in indices]
        source_raw = [source_lines_unsorted[i] for i in indices]
        source_lens = [source_lens_unsorted[i] for i in indices]
        sentences_indices = [sentences_indices_unsorted[i] for i in indices]
        if len(target_lines_unsorted) > 0:
            target_lines = [target_lines_unsorted[i] for i in indices]
            target_raw = [target_lines_unsorted[i] for i in indices]
            target_lens = [target_lens_unsorted[i] + 2 for i in indices] # + 2 because of SOS and EOS
        else:
            target_lines = []
            target_raw = []
            target_lens = []

        #Fixing base language model. (Here language model is different from actual LM in NLP)
        #In case of test set or validation set, we need to fix train language model as base language model.
        #because, we need to convert the texts from test/valid set into sequences according to training data.
        source_base_language = self.source_lang
        if source_language_model != None:
            source_base_language = source_language_model
        target_base_language = self.target_lang
        if target_language_model != None :
            target_base_language = target_language_model
        
        #Forming source and target sequences.
        _source_lines = []
        for i in source_lines:
            _source_lines.append(i)
        source_lines = [source_base_language.sentence2Sequence(line, add_sos = False, add_eos = False, maxlen = source_max_len, pad = True, max_word_len = source_base_language.max_word_len)[0] for line in _source_lines]
        source_lines_hashes = [source_base_language.sentence2Sequence(line, add_sos = False, add_eos = False, maxlen = source_max_len, pad = True, max_word_len = source_base_language.max_word_len)[1] for line in _source_lines]
        
        source_lines = Variable(torch.LongTensor(source_lines), requires_grad=False).cuda()
        source_lines_hashes = Variable(torch.LongTensor(source_lines_hashes), requires_grad=False).cuda()
        source_lens = torch.LongTensor(source_lens).cuda()
        sentences_indices = torch.LongTensor(sentences_indices).cuda()
        #transposing because, it's shape is currently batch size x seq len
        #But the required shape should be seq len x batch size
        source_lines = source_lines.transpose(0, 1)

        if len(target_lines) > 0:
            _target_lines = []
            for i in target_lines:
                _target_lines.append(i)
            target_lines = [target_base_language.sentence2Sequence(line, add_sos = True, add_eos = True, maxlen = target_max_len, pad = True, max_word_len=target_base_language.max_word_len)[0] for line in _target_lines]
            target_lines_hashes = [target_base_language.sentence2Sequence(line, add_sos = True, add_eos = True, maxlen = target_max_len, pad = True, max_word_len=target_base_language.max_word_len)[1] for line in _target_lines]
            target_lines = Variable(torch.LongTensor(target_lines), requires_grad=False).cuda()
            target_lines_hashes = Variable(torch.LongTensor(target_lines_hashes), requires_grad=False).cuda()
            target_lens = torch.LongTensor(target_lens).cuda()
            target_lines = target_lines.transpose(0, 1)
        else:
            target_lens = []
            target_raw = []
            target_lines_hashes = []
        
        batch = BatchData(source_lines, target_lines, source_lens, target_lens,
                          sentences_indices, source_raw, target_raw, samples, 
                          source_lines_hashes, target_lines_hashes)
        return batch
