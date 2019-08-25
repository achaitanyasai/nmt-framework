#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
tests for data_iterator.py
'''

import random
random.seed(8595)
import unittest
import data_iterator

class testField(unittest.TestCase):

    def setUp(self):
        pass

    def test___init__(self):
        with self.assertRaises(AssertionError) as context:
            f = data_iterator.Field(lang_type='some type')
        with self.assertRaises(AssertionError) as context:
            f = data_iterator.Field(lang_type=None)

    def test_basic_checks(self):
        f = data_iterator.Field(lang_type='source')
        with self.assertRaises(AssertionError) as context:
            f.build_vocab([])

    def test_build_vocab_source_nwords(self):
        f = data_iterator.Field('source', max_vocab_size=7, ignore_too_many_unknowns=True)
        raw_sentences = ['hello world', 'poqr', 'hello']
        f.build_vocab(raw_sentences)
        self.assertEqual(f.nWords, 5)
        a = {0: 'UNK', 1: 'PAD', 2: 'hello', 3: 'world', 4: 'poqr'}
        self.assertEqual(f.idx2word, a)
        self.assertEqual(f.nWords, len(f.word2idx))
        self.assertEqual(f.nWords, len(f.word2count))

    def test_build_vocab_source_nwords_ignore_too_many_unknowns(self):
        f = data_iterator.Field('source', max_vocab_size=2, ignore_too_many_unknowns=True)
        raw_sentences = ['hello world', 'poqr', 'hello']
        f.build_vocab(raw_sentences)
        self.assertEqual(f.nWords, 2)
        a = {0: 'UNK', 1: 'PAD'}
        self.assertEqual(f.idx2word, a)
        self.assertEqual(f.nWords, len(f.word2idx))
        self.assertEqual(f.nWords, len(f.word2count))

    def test_build_vocab_target_nwords(self):
        f = data_iterator.Field('target', max_vocab_size=100, ignore_too_many_unknowns=True)
        raw_sentences = ['hello world', 'poqr', 'hello']
        f.build_vocab(raw_sentences)
        self.assertEqual(f.nWords, 7)
        a = self.idx2word = {0: 'UNK', 1: 'PAD', 2: 'SOS', 3: 'EOS', 4: 'hello', 5: 'world', 6: 'poqr'}
        self.assertEqual(f.idx2word, a)
        self.assertEqual(f.nWords, len(f.word2idx))
        self.assertEqual(f.nWords, len(f.word2count))
        self.assertEqual(f.unkCount , 0)

    def test_build_vocab_target_nwords_ignore_too_many_unknowns(self):
        f = data_iterator.Field('target', max_vocab_size=4, ignore_too_many_unknowns=True)
        raw_sentences = ['hello world', 'poqr', 'hello', 'stop', 'worldss']
        f.build_vocab(raw_sentences)
        self.assertEqual(f.nWords, 4)
        a = {0: 'UNK', 1: 'PAD', 2: 'SOS', 3: 'EOS'}
        self.assertEqual(f.unkCount, 1)
        self.assertEqual(f.idx2word, a)
        self.assertEqual(f.nWords, len(f.word2idx))
        self.assertEqual(f.nWords, len(f.word2count))

    def test_sentence2indices(self):
        f = data_iterator.Field('source', max_vocab_size=100, ignore_too_many_unknowns=True)
        raw_sentences = ['hello world', 'poqr', 'hello']
        f.build_vocab(raw_sentences)
        a, b = f.sentence2indices('hello poqr')
        self.assertEqual(a, [2, 4])
        self.assertEqual(b, ['hello', 'poqr'])

        a, b = f.sentence2indices('hello rareword')
        self.assertEqual(a, [2, 0])
        self.assertEqual(b, ['hello', 'UNK'])

        a, b = f.sentence2indices('hello UNK')
        self.assertEqual(a, [2, 0])
        self.assertEqual(b, ['hello', 'UNK'])

    def test_sentence2indices_valid_types(self):
        f = data_iterator.Field('source', max_vocab_size=100, ignore_too_many_unknowns=True)
        _, _ = f.sentence2indices('asdf qwer')
        _, _ = f.sentence2indices(['asdf', 'qwer'])
        with self.assertRaises(AssertionError) as _:
            _, _ = f.sentence2indices(('asdf', 'qwer'))
        _, _ = f.sentence2indices([])

class testDataIterator(unittest.TestCase):

    def setUp(self):
        pass

    def test___init__(self):
        with self.assertRaises(AssertionError) as _:
            _ = data_iterator.DataIterator(fields=None, fname=None, shuffle=True, data_type=None,
                                                  src_max_len=None, tgt_max_len=None, src_max_vocab_size=None,
                                                  tgt_max_vocab_size=None, ignore_too_many_unknowns=None, cleanup=True)

        iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_very_small/data.csv', shuffle=True, data_type='train',
                 src_max_len=100, tgt_max_len=100, src_max_vocab_size=100,
                 tgt_max_vocab_size=100, ignore_too_many_unknowns=True, cleanup=False)

        self.assertEqual(iterator.nSentences, 4)
        f = open('./tests/toy_very_small/data.csv.tmp')
        x = f.read()
        f.close()
        self.assertEqual(x, 'hello world,hello world\nokrst,okrst\nasdf 22,17 qwe\n22,17\n')
        iterator.reset()
        iterator.reset()
        f = open('./tests/toy_very_small/data.csv.tmp')
        x = f.read()
        f.close()
        self.assertEqual(x, 'asdf 22,17 qwe\nhello world,hello world\nokrst,okrst\n22,17\n')

        iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_very_small/data.csv', shuffle=False,
                                              data_type='train',
                                              src_max_len=100, tgt_max_len=100, src_max_vocab_size=100,
                                              tgt_max_vocab_size=100, ignore_too_many_unknowns=True,
                                              cleanup=False)

        self.assertEqual(iterator.nSentences, 4)
        f = open('./tests/toy_very_small/data.csv.tmp')
        x = f.read()
        f.close()
        self.assertEqual(x, 'hello world,hello world\nokrst,okrst\n22,17\nasdf 22,17 qwe\n')

    def test_length_pruning(self):
        iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_very_small/data.csv', shuffle=True,
                                              data_type='train',
                                              src_max_len=1, tgt_max_len=1, src_max_vocab_size=100,
                                              tgt_max_vocab_size=100, ignore_too_many_unknowns=True)

        self.assertEqual(iterator.nSentences, 4)
        self.assertEqual(iterator.nSentencesPruned, 2)
        f = open('./tests/toy_very_small/data.csv.tmp')
        x = f.read()
        f.close()
        self.assertEqual(x, '22,17\nokrst,okrst\n')
        a = []
        for i in iterator.next:
            a.append(i)
        self.assertEqual(a, [('22', '17', 0), ('okrst', 'okrst', 1)])

    def test_testset_iteration(self):
        with self.assertRaises(AssertionError) as _:
            iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_very_small/data.csv', shuffle=True,
                                                  data_type='test',
                                                  src_max_len=2, tgt_max_len=2, src_max_vocab_size=100,
                                                  tgt_max_vocab_size=100, ignore_too_many_unknowns=True)

        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_very_small/data.csv', shuffle=True,
                                              data_type='train',
                                              src_max_len=100, tgt_max_len=100, src_max_vocab_size=100,
                                              tgt_max_vocab_size=100, ignore_too_many_unknowns=True)

        iterator = data_iterator.DataIterator(fields=(train_iterator.sourceField, train_iterator.targetField),
                                              fname='./tests/toy_very_small/data_test.csv', shuffle=False,
                                              data_type='test', src_max_len=2, tgt_max_len=2, src_max_vocab_size=100,
                                              tgt_max_vocab_size=100, ignore_too_many_unknowns=True)

        self.assertEqual(iterator.nSentences, 6)
        self.assertEqual(iterator.nSentencesPruned, 6)
        f = open('./tests/toy_very_small/data_test.csv.tmp')
        x = f.read()
        f.close()
        self.assertEqual(x, 'hello world,\nokrst,\n22,\nasdf 22,\nnot seen,\n22 ghsfghda,')
        a = []
        c = []
        for i in iterator.next:
            a.append(i)
            c.append(iterator.sourceField.sentence2indices(i[0]))
        b = [
                ('hello world', None, 0),
                ('okrst', None, 1),
                ('22', None, 2),
                ('asdf 22', None, 3),
                ('not seen', None, 4),
                ('22 ghsfghda', None, 5)
        ]
        d = [
                ([2, 3], ['hello', 'world']),
                ([4], ['okrst']),
                ([5], ['22']),
                ([6, 5], ['asdf', '22']),
                ([0, 0], ['UNK', 'UNK']),
                ([5, 0], ['22', 'UNK'])
        ]
        self.assertEqual(a, b)
        self.assertEqual(c, d)

    def test_next_batch(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_very_small/data.csv', shuffle=True,
                                              data_type='train',
                                              src_max_len=100, tgt_max_len=100, src_max_vocab_size=100,
                                              tgt_max_vocab_size=100, ignore_too_many_unknowns=True)
        a = train_iterator.next_batch(1)
        a = train_iterator.next_batch(1)
        a = train_iterator.next_batch(1)
        print(a.src)
        print(a.tgt)

if __name__ == '__main__':
   unittest.main()
