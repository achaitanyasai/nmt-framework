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

class Lang(object):
    '''
    This class contains the data related to a language.
    '''
    #TODO:
    #1) Refactor
    #2) Add character level features
    #3) Sort vocab based on freq
    def __init__(self, fname, maxlen, langtype = 'source', chars = False):
        logging.info('Reading %s (%s language)' % (fname, langtype))
        self.fname = fname
        self.langtype = langtype
        self.unkcount = 0
        self.raw_data = util.fread(self.fname).split('\n')
        #char2idx is only useful for source language
        self.char2idx = {'PAD' : 0, 'UNK' : 1}
        self.idx2char = {0 : 'PAD', 1 : 'UNK'}
        self.n_chars = 2    
        self.chars = chars
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
        self.maxlen = maxlen
        self.langtype = langtype
        self.sentidx = 0

        self.build_vocab()
        L = len(self.raw_data)        

        for j, _ in enumerate(self.raw_data):
            sentence = _.strip().split(' ')[:self.maxlen]
            self.normalized_data.append(sentence)
            self.n_sentences += 1
            for word in sentence:
                self.addWord(word)
            sys.stdout.write('Processed: %d/%d(%.3f)    \r' % (j, L, float(j * 100) / L))
            sys.stdout.flush()
        print self.n_words
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

    def addChar(self, word):
        word = list(word.decode('utf-8'))
        for char in word:
            if char not in self.char2idx:
                self.char2idx[char] = self.n_chars
                self.idx2char[self.n_chars] = char
                self.n_chars += 1
            else:
                pass #increment char2count[char]
    
    def addWord(self, word):
        try:
            tmp = self._d[word]
        except KeyError:
            self.word2count[word] = 0
            return
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.word2count[word] = 1
            self.idx2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
        self.addChar(word) # Building character vocabulary
    
    def build_vocab(self):
        _d = {}
        self._d = {}
        for _ in self.raw_data:
            sentence = _.strip().split(' ')[:self.maxlen]
            for word in sentence:
                try:
                    _d[word] += 1
                except KeyError:
                    _d[word] = 1
        _sd = sorted(_d.items(), key=operator.itemgetter(1), reverse = True)
        print len(_sd)
        for i in _sd[:60000]:
            self._d[i[0]] = i[1]
        del _sd
        del _d
        
        
    def idx2sentence(self, idx):
        return self.normalized_data[idx]
    
    def sentence2Sequence(self, sentence, add_sos = True, add_eos = True, maxlen = 0, pad = False, max_word_len = 0):
        seq = []
        if(isinstance(sentence, str)):
            orgsentence = sentence.split(' ')
            sentence = sentence.split(' ')
        else:
            orgsentence = ' '.join(sentence)
            assert(isinstance(sentence, list))
        
        assert(maxlen > 0)

        if add_sos:
            if sentence[0] != 'SOS' :
                sentence = ['SOS'] + sentence
        else:
            if sentence[0] == 'SOS':
                sentence = sentence[1::]
                
        if add_eos:
            if sentence[-1] != 'EOS':
                sentence = sentence[:maxlen - 1]
                sentence = sentence + ['EOS']
        else:
            if sentence[-1] == 'EOS':
                sentence = sentence[:-1]
        
        if(self.chars):
            for word in sentence:
                seq.append(self.word2CharSequence(word, max_word_len))
            while len(seq) < maxlen:
                seq.append(self.word2CharSequence('', max_word_len))
        else:
            c = 0
            newsent = []
            for word in sentence:
                try:
                    tmp = self.word2idx[word]
                    if((word not in ['PAD', 'SOS', 'EOS']) and self.word2count[word] <= 1):
                        tmp = self.word2idx['UNK']
                        newsent.append('UNK')
                    else:
                        newsent.append(word)
                    seq.append(tmp)
                except KeyError:
                    seq.append(self.word2idx['UNK'])
                    newsent.append('UNK')
            if pad:
                while len(seq) < maxlen:
                    seq = seq + [self.word2idx['PAD']]
            #if self.langtype == 'target':
            #    print 'annotated:', ' '.join(newsent)
        return seq
    
    def word2CharSequence(self, word, max_word_len):
        seq = []
        chars = list(word.decode('utf-8'))
        
        for char in chars:
            try:
                seq.append(self.char2idx[char])
            except KeyError:
                #assert(False)
                seq.append(self.char2idx['UNK'])
        while len(seq) < max_word_len:
            seq.append(self.char2idx['PAD'])
        for i in seq:
            assert(i < self.n_chars)
        return seq

    def sequence2Sentence(self, seq):
        sentence = []
        for i in seq:
            if self.idx2word[i] in ['SOS', 'EOS', 'PAD']:
                continue
            try:
                sentence.append(self.idx2word[i])
            except KeyError:
                sentence.append('UNK')
        return sentence
    
    def idx2Sequence(self, idx):
        sentence = self.idx2sentence(idx)
        return self.sentence2Sequence(sentence)
    
    def saveVocab(self, filepath):
        print 'Saving vocab to %s' % filepath
        c1 = 0
        c2 = 0
        total = 0
        f = open(filepath, 'w')
        for i in xrange(self.n_words):
            #f.write("%s  %d\n" % (self.idx2word[i], self.word2count[self.idx2word[i]]))
            f.write("%s\n" % (self.idx2word[i]))
            cword = self.idx2word[i]
            if(self.word2count[cword] <= 1):
                c1 += 1
            if self.word2count[cword] <= 2:
                c2 += 1
            total += 1
        print c1, c2, total
        f.close()
    
class BatchData(object):
    '''
    A container for samples in a batch
    '''
    def __init__(self, src, tgt, src_len, tgt_len,
                 indices, src_raw, tgt_raw, batch_size):
        self.indices = indices
        self.src = src
        self.tgt = tgt
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.src_raw = src_raw
        self.tgt_raw = tgt_raw
        self.batch_size = batch_size

class dataIterator(object):
    '''
    Iterator for data.
    each call to .next() method returns a pair of sequences (source sequence, target dequence) 
    which can be fed to encoder and decoder respectively.
    '''
    #TODO: refactor
    def __init__(self, source_lang, target_lang, shuffle = True):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.shuffle = shuffle
        self.order = [i for i in xrange(self.source_lang.n_sentences)]
        self.source_lens = [len(i) for i in self.source_lang.normalized_data]
        self.n_samples = self.source_lang.n_sentences
        self.reset()

    def __iter__(self):
        return self
    
    def reset(self):
        self.sentidx = 0
        if self.shuffle:
            random.shuffle(self.order)
        
    def next(self, source_language_model = None, target_language_model = None, return_texts = False):
        #FIXME: This returns only texts. ideally, it should return texts and sequences
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

        max_word_len = 0

        source_lines_unsorted = []
        target_lines_unsorted = []
        sentences_indices_unsorted = []
        source_lens_unsorted = []
        target_lens_unsorted = []

        samples = 0
        for _ in xrange(batch_size):
            try:
                source_line, target_line, sentidx = self.next(source_language_model, target_language_model, return_texts = True)
                samples += 1
            except Exception as e:
                print e
                break
            for w in source_line:
                max_word_len = max(max_word_len, len(list(w.decode('utf-8'))))
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
            target_lens = [target_lens_unsorted[i] for i in indices]
        else:
            target_lines = []

        #Fixing base language model. (Here language model is different from actual LM in NLP)
        #In case of test set or validation set, we need to fix train language model as base language model.
        #because, we need to convert the texts from test/valid set into sequences according to training model.
        source_base_language = self.source_lang
        if source_language_model != None:
            source_base_language = source_language_model
        target_base_language = self.target_lang
        if target_language_model != None :
            target_base_language = target_language_model
        
        #Forming source and target sequences.
        source_lines = [source_base_language.sentence2Sequence(line, add_sos = False, add_eos = False, maxlen = source_max_len, pad = True, max_word_len = max_word_len) for line in source_lines]
        source_lines = Variable(torch.LongTensor(source_lines)).cuda()
        source_lens = torch.LongTensor(source_lens).cuda()
        sentences_indices = torch.LongTensor(sentences_indices).cuda()
        #transposing because, it's shape is currently batch size x seq len
        #But the required shape should be seq len x batch size
        source_lines = source_lines.transpose(0, 1)

        if len(target_lines) > 0:
            target_lines = [target_base_language.sentence2Sequence(line, add_sos = True, add_eos = True, maxlen = target_max_len, pad = True) for line in target_lines]
            target_lines = Variable(torch.LongTensor(target_lines)).cuda()
            target_lens = torch.LongTensor(target_lens).cuda()
            target_lines = target_lines.transpose(0, 1)
        else:
            target_lens = []
            target_raw = []
        
        #print target_lines[0]
        #exit(0)
        batch = BatchData(source_lines, target_lines, source_lens, target_lens,
                          sentences_indices, source_raw, target_raw, samples)
        
        return batch
