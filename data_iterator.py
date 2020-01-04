#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Data iterators
'''

import os
import secrets
import random

import numpy as np
import torch
from torch.autograd import Variable

import util
import sys
import operator
import tqdm
import csv
from structs import *

class BatchData(object):
    '''
    A container for samples in a batch
    '''
    def __init__(self, src, tgt, src_len, tgt_len,
                 indices, src_raw, tgt_raw, batch_size):
        self.indices = indices
        self.src = src  # Source sequence of shape: seq_len x batch_size
        self.tgt = tgt  # Target sequence of shape: seq_len x batch_size or None
        self.src_lens = src_len #lengths of the source sequence. Shape: batch_size
        self.tgt_lens = tgt_len #lengths of the target sequence. Shape: batch_size or None
        self.src_raw = src_raw #Raw source sentences
        self.tgt_raw = tgt_raw #Raw target sentences or None
        self.batch_size = batch_size #Number of samples in current batch

    def transpose(self):
        self.src = self.src.transpose(0, 1)
        try:
            self.tgt = self.tgt.transpose(0, 1)
        except Exception:
            pass

class Field(object):
    logger.error('Lowercase, check if it\'s correct: "# FIXME: applying lower() here:"')

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
            # TODO(Just a warning): # WARN: we are not truncating the max length here. It should be done in the caller.
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
                self.add_word(word)
                # Remove the below snippet
                # if word not in self.word2idx:
                #     self.word2idx[word] = self.nWords
                #     self.word2count[word] = 1
                #     self.idx2word[self.nWords] = word
                #     self.nWords += 1
                # else:
                #     self.word2count[word] += 1
        del _sd
        del _d
        del __d

    def build_vocab_V2(self, _d):
        assert len(_d) > 0
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
        __d = {}

        for i in _sd[:self.maxVocabSize - v]:
            __d[i[0]] = i[1]
        for word in _d:
            if word in __d:
                self.add_word(word)
        del _sd
        del _d
        del __d

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.nWords
            self.word2count[word] = 1
            self.idx2word[self.nWords] = word
            self.nWords += 1
        else:
            self.word2count[word] += 1


    def sentence2indices(self, sent, add_sos=False, add_eos=False, max_len=0):
        if type(sent) == type(''):
            sent = sent.lower().lstrip().rstrip().split()
        assert type(sent) == type([])
        if len(sent) == 0:
            logger.warning('Empty sentence')
            return [], []
        assert type(sent[0]) == type('')
        resIdx = []
        resStr = []
        if self.langType == 'target':
            if add_sos and add_eos:
                sent = ['SOS'] + sent + ['EOS']
            else:
                logger.warning('SOS and EOS tokens are not being added')

        for _word in sent:
            if _word in self.word2idx:
                if _word == 'UNK':
                    logger.warning('UNK token found in the sentence: "%s"' % (' '.join(sent)))
                word =_word
            else:
                word = 'UNK'
            resStr.append(word)
            resIdx.append(self.word2idx[word])

        while len(resIdx) < max_len + add_sos + add_eos:
            word = 'PAD'
            resIdx.append(self.word2idx[word])
            resStr.append(word)
        # assert len(resIdx) <= max_len + add_sos + add_eos
        return resIdx, resStr

    def vec2sentence(self, vec):
        assert type(vec) == type([])
        res = []
        for i in vec:
            res.append(self.idx2word[i])
        return ' '.join(res)

class DataIterator(object):
    '''
    Iterator for data.
    each call to .next() method returns a pair of sequences (source sequence, target dequence) 
    which can be fed to encoder and decoder respectively.
    '''
    def __init__(self, fields=None, fname=None, shuffle=True, data_type=None,
                 src_max_len=None, tgt_max_len=None, src_max_vocab_size=None,
                 tgt_max_vocab_size=None, ignore_too_many_unknowns=None,
                 cleanup=True, break_on_stop_iteration=True,
                 tie_embeddings=False):

        logger.warning('Add tests for tie embedding')
        self.cleanup = cleanup
        self.break_on_stop_iteration = break_on_stop_iteration
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
        self.tie_embeddings = tie_embeddings
        if tie_embeddings and data_type == 'train':
            for word in self.sourceField.word2idx:
                self.targetField.add_word(word)
            self.sourceField.nWords = self.targetField.nWords
            self.sourceField.word2idx = self.targetField.word2idx
            self.sourceField.idx2word = self.targetField.idx2word
            self.sourceField.word2count = self.targetField.word2count
        del fields

        # f = open('/tmp/vocab.en', 'w')
        # for word in self.sourceField.word2idx:
        #     f.write('%s\n' % word)
        # f.close()
        #
        # f = open('/tmp/vocab.hi', 'w')
        # for word in self.targetField.word2idx:
        #     f.write('%s\n' % word)
        # f.close()
        # exit(0)

    def __del__(self):
        if self.cleanup:
            try:
                try:
                    logger.info('Removing %s' % self.preprocessedfname)
                except NameError:
                    pass
                os.remove(self.preprocessedfname)
            except (OSError, FileNotFoundError):
                pass
            except AttributeError:
                pass

    def __iter__(self):
        return self

    def copy_file(self, source_file, destination_file):
        f = open(source_file)
        g = open(destination_file, 'w')
        for line in f.readlines():
            x = line.lower().strip()
            if x != '':
                g.write(x + '\n')

        f.close()
        g.close()

    def preprocess(self, fname, src_max_len, tgt_max_len, src_max_vocab_size, tgt_max_vocab_size, ignore_too_many_unknowns):

        sourceField = Field('source', max_vocab_size=src_max_vocab_size, ignore_too_many_unknowns=ignore_too_many_unknowns)
        targetField = Field('target', max_vocab_size=tgt_max_vocab_size, ignore_too_many_unknowns=ignore_too_many_unknowns)
        csvFile = open(fname)

        of = open(self.preprocessedfname, 'w')
        csvwriter = csv.writer(of, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

        src_dict = {}
        tgt_dict = {}
        for row in csv.reader(csvFile, delimiter=",", quotechar='"'):
            # FIXME: applying lower() here
            cur_src = row[0].lower().strip().split()
            cur_tgt = row[1].lower().strip().split()
            # Note that this should only be done to training data.
            if (len(cur_src) <= src_max_len + 0) and (len(cur_tgt) <= tgt_max_len + 0):
                csvwriter.writerow([' '.join(cur_src), ' '.join(cur_tgt)])
                for word in cur_src:
                    try:
                        src_dict[word] += 1
                    except KeyError:
                        src_dict[word] = 1
                for word in cur_tgt:
                    try:
                        tgt_dict[word] += 1
                    except KeyError:
                        tgt_dict[word] = 1

        sourceField.build_vocab_V2(src_dict)
        targetField.build_vocab_V2(tgt_dict)

        csvFile.close()
        of.close()

        return (sourceField, targetField)

    def reset(self):
        '''
        This method is deprecated because of too much memory usage. Please use self.resetV2()
        for optimized memory usage
        :return:
        '''

        self.resetV2()
        return

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
                cur_src = src.lower().strip().split()
                cur_tgt = tgt.lower().strip().split()
                # Note that this should only be done to training data.
                if (len(cur_src) <= self.src_max_len + 0) and (len(cur_tgt) <= self.tgt_max_len + 0):
                    csvwriter.writerow([src, tgt])
            of.close()
        self.next = self.nextgen()

    def resetV2(self):
        self.sentidx = 0
        if self.shuffle:
            salt = secrets.token_hex(8)
            src_tmp_file = 'data/src.%s.txt.tmp' % salt
            tgt_tmp_file = 'data/tgt.%s.txt.tmp' % salt

            random.shuffle(self.order)
            csvFile = open(self.basefname)
            rows_src = []
            for row in csv.reader(csvFile, delimiter=",", quotechar='"'):
                rows_src.append(row[0])
            csvFile.close()
            L = len(self.order)

            f = open(src_tmp_file, 'w')
            for i in range(L):
                src = rows_src[self.order[i]]
                f.write('%s\n' % src.strip())
            f.close()
            del rows_src

            csvFile = open(self.basefname)
            rows_tgt = []
            for row in csv.reader(csvFile, delimiter=",", quotechar='"'):
                rows_tgt.append(row[1])
            csvFile.close()
            L = len(self.order)
            f = open(tgt_tmp_file, 'w')
            for i in range(L):
                tgt = rows_tgt[self.order[i]]
                f.write('%s\n' % tgt.strip())
            f.close()
            del rows_tgt

            of = open(self.preprocessedfname, 'w')
            csvwriter = csv.writer(of, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

            sf = open(src_tmp_file)
            tf = open(tgt_tmp_file)
            for src, tgt in zip(sf.readlines(), tf.readlines()):
                cur_src = src.lower().strip().split()
                cur_tgt = tgt.lower().strip().split()
                if (len(cur_src) <= self.src_max_len + 0) and (len(cur_tgt) <= self.tgt_max_len + 0):
                    csvwriter.writerow([src.strip(), tgt.strip()])
            of.close()

            os.system('rm %s' % src_tmp_file)
            os.system('rm %s' % tgt_tmp_file)
        self.next = self.nextgen()

    def nextgen(self):
        with open(self.preprocessedfname) as fileDescriptor:
            self.csvReader = csv.reader(fileDescriptor)
            for row in self.csvReader:
                source_sentence = row[0]
                if self.dataType in ['train', 'valid', 'test']:
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
                #############################
                # Fixed by adding self.break_on_stop_iteration variable
                # TODO: uncomment below line and remove break if you want to use updates instead of epochs.
                # self.reset()
                # break
                #############################
                if self.break_on_stop_iteration:
                    break
                else:
                    self.reset()
            # TODO: Do we need to catch and ignore exception?
            # except Exception as e:
            #     i += 1
            #     continue

        if len(src_lines_unsorted) == 0:
            return None

        src_max_len = max(src_lens_unsorted)
        tgt_max_len = 0
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

        src_lines = torch.LongTensor(src_lines).cuda()
        src_lens = torch.LongTensor(src_lens).cuda()
        if len(tgt_lens_unsorted) > 0:
            for i in indices:
                y = self.targetField.sentence2indices(tgt_lines_unsorted[i], add_sos=True, add_eos=True, max_len=tgt_max_len)
                tgt_lines.append(y[0])
                tgt_raw.append(tgt_lines_unsorted[i])
                tgt_lens.append(tgt_lens_unsorted[i])

            tgt_lines = torch.LongTensor(tgt_lines).cuda()
            tgt_lens = torch.LongTensor(tgt_lens).cuda()

        batch = BatchData(src_lines, tgt_lines, src_lens, tgt_lens,
                          sentences_indices, src_raw, tgt_raw, n_samples)
        return batch
