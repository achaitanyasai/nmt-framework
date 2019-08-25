#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Data iterators
'''

import os
import logging
import random

import numpy as np
import torch
from torch.autograd import Variable

import util
import sys
import operator
import tqdm
import base_config as config
import csv

logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('./data/training_logs.txt')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(pathname)s:%(lineno)d - %(message)s','%Y-%m-%d:%H:%M:%S')
fh.setFormatter(formatter)
logger.addHandler(fh)

class BatchData(object):
    '''
    A container for samples in a batch
    '''
    def __init__(self, src, tgt, src_len, tgt_len,
                 indices, src_raw, tgt_raw, batch_size):
        self.indices = indices
        self.src = src  # Source sequence of shape: seq_len x batch_size
        self.tgt = tgt  # Target sequence of shape: seq_len x batch_size or None
        self.src_len = src_len #lengths of the source sequence. Shape: batch_size
        self.tgt_len = tgt_len #lengths of the target sequence. Shape: batch_size or None
        self.src_raw = src_raw #Raw source sentences
        self.tgt_raw = tgt_raw #Raw target sentences or None
        self.batch_size = batch_size #Number of samples in current batch
        self.src_hashes = None
        self.tgt_hashes = None

class Field(object):
    def __init__(self, lang_type, max_vocab_size=-1, ignore_too_many_unknowns=False):
        assert lang_type in ['source', 'target']
        self.langType = lang_type
        if lang_type == 'source':
            # 'UNK': 0, 'PAD': 1
            self.word2idx = {'UNK': 0, 'PAD': 1}
            self.word2count = {'UNK': 1, 'PAD': 1}
            self.idx2word = {0: 'UNK', 1: 'PAD'}
            self.nWords = 2
        else:
            # 'UNK': 0, 'PAD': 1, 'SOS: 2, 'EOS': 3
            self.word2idx = {'SOS': 2, 'EOS': 3, 'UNK': 0, 'PAD': 1}
            self.word2count = {'SOS': 1, 'EOS': 1, 'UNK': 1, 'PAD': 1}
            self.idx2word = {0: 'UNK', 1: 'PAD', 2: 'SOS', 3: 'EOS'}
            self.nWords = 4
        self.maxVocabSize = max_vocab_size
        self.unkCount = -1
        self.ignoreTooManyUnk = ignore_too_many_unknowns


    def build_vocab(self, raw_sentences):
        assert len(raw_sentences) > 0

        _d = {}
        __d = {}
        for sentence in raw_sentences:
            # TODO: # WARN: we are not truncating the max length here. It should be done in the caller.
            if type(sentence) == type(''):
                sentence = sentence.split()
            for word in sentence:
                try:
                    _d[word] += 1
                except KeyError:
                    _d[word] = 1
        _sd = sorted(_d.items(), key=operator.itemgetter(1), reverse=True)
        num_unknown_tokens = max(0, len(_sd) - self.maxVocabSize)
        self.unkCount = num_unknown_tokens

        logger.info('Unknown tokens: %d (%.2f%%)' % (num_unknown_tokens, (num_unknown_tokens * 100.0) / len(_sd)))
        logger.info('Vocab size: %d (%.2f%%)' % (
        len(_sd) - num_unknown_tokens, 100.0 - (num_unknown_tokens * 100.0) / len(_sd)))

        if not self.ignoreTooManyUnk and (num_unknown_tokens * 100.0) / len(_sd) >= 20.00:
            if self.maxVocabSize >= 100000:
                logger.info('Ignoring too many unknowns option as vocabulary size is more than 100K')
            else:
                logger.error('Too many unknowns. Please increase vocabulary size')
                raise Exception('Too many unknowns. Please increase vocabulary size\n')

        if not self.ignoreTooManyUnk and (num_unknown_tokens * 100.0) / len(_sd) <= 2.00:
            logger.error('Very few unknowns. Please decrease vocabulary size')
            raise Exception('Very few unknowns. Please decrease vocabulary size\n')

        if self.langType == 'source':
            v = 2
        else:
            v = 4
        for i in _sd[:self.maxVocabSize - v]:
            __d[i[0]] = i[1]

        for sentence in raw_sentences:
            if type(sentence) == type(''):
                sentence = sentence.split()
            for word in sentence:
                try:
                    _ = __d[word]
                except KeyError:
                    continue
                if word not in self.word2idx:
                    self.word2idx[word] = self.nWords
                    self.word2count[word] = 1
                    self.idx2word[self.nWords] = word
                    self.nWords += 1
                else:
                    self.word2count[word] += 1
        del _sd
        del _d
        del __d

    def sentence2indices(self, sent, max_len=0):
        if type(sent) == type(''):
            sent = sent.lstrip().rstrip().split()
        assert type(sent) == type([])
        if len(sent) == 0:
            logger.warning('Empty sentence')
            return [], []
        assert type(sent[0]) == type('')
        resIdx = []
        resStr = []
        for _word in sent:
            if _word in self.word2idx:
                if _word == 'UNK':
                    logger.warning('UNK token found in the sentence: "%s"' % (' '.join(sent)))
                word =_word
            else:
                word = 'UNK'
            resStr.append(word)
            resIdx.append(self.word2idx[word])

        while len(resIdx) < max_len:
            word = 'PAD'
            resIdx.append(self.word2idx[word])
            resStr.append(word)

        return resIdx, resStr

    def vec2sentence(self, vec):
        pass

class DataIterator(object):
    '''
    Iterator for data.
    each call to .next() method returns a pair of sequences (source sequence, target dequence) 
    which can be fed to encoder and decoder respectively.
    '''
    def __init__(self, fields=None, fname=None, shuffle=True, data_type=None,
                 src_max_len=None, tgt_max_len=None, src_max_vocab_size=None,
                 tgt_max_vocab_size=None, ignore_too_many_unknowns=None,
                 cleanup=True):

        self.cleanup = cleanup
        assert fname is not None

        if shuffle:
            assert data_type == 'train'

        if data_type == 'train' and not shuffle:
            logger.warning('Please shuffle the training set')

        self.basefname = fname
        self.preprocessedfname = fname + '.tmp'
        self.copy_file(self.basefname, self.preprocessedfname)

        if fields is None:
            assert data_type in ['train']
            fields = self.preprocess(fname, src_max_len, tgt_max_len, src_max_vocab_size,
                                     tgt_max_vocab_size, ignore_too_many_unknowns)
        else:
            assert data_type in ['valid', 'test']

        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len

        self.sourceField = fields[0]
        self.targetField = fields[1]
        self.shuffle = shuffle
        self.dataType = data_type

        self.nSentences = util.count_lines(self.basefname)
        self.nSentencesPruned = util.count_lines(self.preprocessedfname)
        self.order = [i for i in range(self.nSentences)]
        self.fileDescriptor = None
        self.csvReader = None

        self.reset()
        del fields

    def __del__(self):
        if self.cleanup:
            try:
                logger.info('Removing %s' % self.preprocessedfname)
                os.remove(self.preprocessedfname)
            except (OSError, FileNotFoundError):
                pass
            except AttributeError:
                pass

    def __iter__(self):
        return self

    def copy_file(self, source_file, destination_file):
        f = open(source_file)
        x = f.read().strip()
        f.close()

        f = open(destination_file, 'w')
        f.write(x)
        f.close()
        del x

    def preprocess(self, fname, src_max_len, tgt_max_len, src_max_vocab_size, tgt_max_vocab_size, ignore_too_many_unknowns):

        sourceField = Field('source', max_vocab_size=src_max_vocab_size, ignore_too_many_unknowns=ignore_too_many_unknowns)
        targetField = Field('target', max_vocab_size=tgt_max_vocab_size, ignore_too_many_unknowns=ignore_too_many_unknowns)
        csvFile = open(fname)
        src = []
        tgt = []
        for row in csv.reader(csvFile, delimiter=",", quotechar='"'):
            cur_src = row[0].strip().split()
            cur_tgt = row[1].strip().split()
            # Note that this should only be done to training data.
            if (len(cur_src) <= src_max_len + 0) and (len(cur_tgt) <= tgt_max_len + 0):
                src.append(cur_src)
                tgt.append(cur_tgt)
        csvFile.close()

        sourceField.build_vocab(src)
        targetField.build_vocab(tgt)

        of = open(self.preprocessedfname, 'w')
        csvwriter = csv.writer(of, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i, j in zip(src, tgt):
            csvwriter.writerow([' '.join(i), ' '.join(j)])
        of.close()

        return (sourceField, targetField)

    def reset(self):
        self.sentidx = 0
        if self.shuffle:
            random.shuffle(self.order)
            csvFile = open(self.basefname)
            rows = []
            for row in csv.reader(csvFile, delimiter=",", quotechar='"'):
                rows.append(row)
            csvFile.close()

            of = open(self.preprocessedfname, 'w')
            csvwriter = csv.writer(of, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            L = len(self.order)
            for i in range(L):
                src = rows[self.order[i]][0]
                tgt = rows[self.order[i]][1]
                cur_src = src.strip().split()
                cur_tgt = tgt.strip().split()
                # Note that this should only be done to training data.
                if (len(cur_src) <= self.src_max_len + 0) and (len(cur_tgt) <= self.tgt_max_len + 0):
                    csvwriter.writerow([src, tgt])
            of.close()
        self.next = self.nextgen()
    
    def nextgen(self):
        with open(self.preprocessedfname) as fileDescriptor:
            self.csvReader = csv.reader(fileDescriptor)
            for row in self.csvReader:
                source_sentence = row[0]
                if self.dataType == 'train':
                    target_sentence = row[1]
                else:
                    target_sentence = None
                yield (source_sentence, target_sentence, self.sentidx)
                self.sentidx += 1
            # Uncomment this if you want to use update rather than batch
            # return None, None, None
    
    def next_batch(self, batch_size):
        src_lines_unsorted = []
        tgt_lines_unsorted = []
        sentences_indices_unsorted = []
        src_lens_unsorted = []
        tgt_lens_unsorted = []
        n_samples = 0

        i = 0
        while i < batch_size:
            try:
                src_line, tgt_line, sentidx = next(self.next)
                src_line = src_line.split()
                src_lines_unsorted.append(src_line)
                src_lens_unsorted.append(len(src_line))
                sentences_indices_unsorted.append(sentidx)

                if tgt_line:
                    tgt_line = tgt_line.split()
                    tgt_lines_unsorted.append(tgt_line)
                    tgt_lens_unsorted.append(len(tgt_line))
                n_samples += 1
                i += 1
            except StopIteration:
                # TODO: uncomment below line and remove break if you want to use updates instead of epochs.
                # self.reset()
                break
            except Exception as e:
                # FIXME: continue or break??
                i += 1
                continue

        if len(src_lines_unsorted) == 0:
            return None

        src_max_len = max(src_lens_unsorted)
        if len(tgt_lens_unsorted) > 0:
            tgt_max_len = max(tgt_lens_unsorted)

        indices = np.argsort(np.array(src_lens_unsorted))[::-1]

        sentences_indices = []
        src_lines = []
        src_raw = []
        src_lens = []

        tgt_lines = []
        tgt_raw = []
        tgt_lens = []

        for i in indices:
            x = self.sourceField.sentence2indices(src_lines_unsorted[i], max_len=src_max_len)
            src_lines.append(x[0])
            sentences_indices.append(sentences_indices_unsorted[i])
            src_raw.append(src_lines_unsorted[i])
            src_lens.append(src_lens_unsorted[i])

        src_lines = torch.LongTensor(src_lines)
        src_lens = torch.LongTensor(src_lens)
        if len(tgt_lens_unsorted) > 0:
            for i in indices:
                y = self.targetField.sentence2indices(tgt_lines_unsorted[i], max_len=tgt_max_len)
                tgt_lines.append(y[0])
                tgt_raw.append(tgt_lines_unsorted[i])
                tgt_lens.append(tgt_lens_unsorted[i])

        tgt_lines = torch.LongTensor(tgt_lines).cuda()
        tgt_lens = torch.LongTensor(tgt_lens).cuda()

        source_lines = []

        batch = BatchData(src_lines, tgt_lines, src_lens, tgt_lens,
                          sentences_indices, src_raw, tgt_raw, n_samples)
        return batch
