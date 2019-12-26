#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
tests for data_iterator.py
'''
import os
import random
random.seed(8595)
import torch
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

class testDataIteratorReproducability(unittest.TestCase):

    def setUp(self):
        pass

    def test_reset(self):
        iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_copy/train.csv', shuffle=True,
                                              data_type='train',
                                              src_max_len=100, tgt_max_len=100, src_max_vocab_size=100,
                                              tgt_max_vocab_size=100, ignore_too_many_unknowns=True, cleanup=True)

        iterator1 = data_iterator.DataIterator(fields=None, fname='./tests/toy_copy/train.csv', shuffle=True,
                                              data_type='train',
                                              src_max_len=100, tgt_max_len=100, src_max_vocab_size=100,
                                              tgt_max_vocab_size=100, ignore_too_many_unknowns=True, cleanup=True)

        for i in range(5):
            z = open(iterator.preprocessedfname).read()

            iterator.reset()
            iterator1.reset()

            x = open(iterator.preprocessedfname).read()
            y = open(iterator1.preprocessedfname).read()

            self.assertEqual(x, y)
            self.assertNotEqual(x, z)

        del iterator1
        del iterator

    def test_reset_length_pruning(self):
        iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_very_small/data.csv', shuffle=True,
                                              data_type='train',
                                              src_max_len=1, tgt_max_len=1, src_max_vocab_size=100,
                                              tgt_max_vocab_size=100, ignore_too_many_unknowns=True, cleanup=True)

        x = open(iterator.preprocessedfname).read()
        y = 'okrst,okrst\n22,17\n'
        self.assertEqual(''.join(sorted(x)), ''.join(sorted(y)))

        iterator1 = data_iterator.DataIterator(fields=(iterator.sourceField, iterator.targetField), fname='./tests/toy_very_small/data_test.csv', shuffle=False,
                                              data_type='test',
                                              src_max_len=None, tgt_max_len=None, src_max_vocab_size=None,
                                              tgt_max_vocab_size=None, ignore_too_many_unknowns=True, cleanup=True)

        x = open(iterator1.preprocessedfname).read()
        y = 'hello world,\nokrst,\n22,\nasdf 22,\nnot seen,\n22 ghsfghda,\n'
        self.assertEqual(''.join(sorted(x)), ''.join(sorted(y)))

        del iterator
        del iterator1

    def test_asserts(self):
        iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_very_small/data.csv', shuffle=True,
                                              data_type='train',
                                              src_max_len=1, tgt_max_len=1, src_max_vocab_size=100,
                                              tgt_max_vocab_size=100, ignore_too_many_unknowns=True, cleanup=False)
        with self.assertRaises(AssertionError) as context:
            iterator.__del__()
        os.remove(iterator.preprocessedfname)

class testDataIterator(unittest.TestCase):

    def setUp(self):
        pass

    def test___init__(self):
        import random
        random.seed(8595)
        with self.assertRaises(AssertionError) as _:
            _ = data_iterator.DataIterator(fields=None, fname=None, shuffle=True, data_type=None,
                                                  src_max_len=None, tgt_max_len=None, src_max_vocab_size=None,
                                                  tgt_max_vocab_size=None, ignore_too_many_unknowns=None, cleanup=True)

        iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_very_small/data.csv', shuffle=True, data_type='train',
                 src_max_len=100, tgt_max_len=100, src_max_vocab_size=100,
                 tgt_max_vocab_size=100, ignore_too_many_unknowns=True, cleanup=True)

        self.assertEqual(iterator.nSentences, 4)
        f = open(iterator.preprocessedfname)
        x = f.read()
        f.close()
        f = open('./tests/toy_very_small/data.csv')
        y = f.read()
        f.close()
        self.assertNotEqual(x, y)
        self.assertEqual(''.join(sorted(x)), ''.join(sorted(y)))
        # self.assertEqual(x, 'hello world,hello world\nokrst,okrst\nasdf 22,17 qwe\n22,17\n')
        iterator.reset()
        iterator.reset()
        iterator.reset()
        iterator.reset()
        f = open(iterator.preprocessedfname)
        x = f.read()
        f.close()
        # self.assertEqual(x, 'asdf 22,17 qwe\nhello world,hello world\nokrst,okrst\n22,17\n')
        self.assertNotEqual(x, y)
        self.assertEqual(''.join(sorted(x)), ''.join(sorted(y)))
        iterator.__del__()

        iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_very_small/data.csv', shuffle=False,
                                              data_type='train',
                                              src_max_len=100, tgt_max_len=100, src_max_vocab_size=100,
                                              tgt_max_vocab_size=100, ignore_too_many_unknowns=True,
                                              cleanup=True)

        self.assertEqual(iterator.nSentences, 4)
        f = open(iterator.preprocessedfname)
        x = f.read()
        f.close()
        self.assertEqual(''.join(sorted(x)), ''.join(sorted('hello world,hello world\nokrst,okrst\n22,17\nasdf 22,17 qwe\n')))
        iterator.__del__()

    def test_length_pruning(self):
        iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_very_small/data.csv', shuffle=True,
                                              data_type='train',
                                              src_max_len=1, tgt_max_len=1, src_max_vocab_size=100,
                                              tgt_max_vocab_size=100, ignore_too_many_unknowns=True)

        self.assertEqual(iterator.nSentences, 4)
        self.assertEqual(iterator.nSentencesPruned, 2)
        f = open(iterator.preprocessedfname)
        x = f.read()
        f.close()
        y = '22,17\nokrst,okrst\n'
        self.assertEqual(''.join(sorted(x)), ''.join(sorted(y)))
        # self.assertEqual(x, '22,17\nokrst,okrst\n')
        a = [
            ('22', '17'),
            ('okrst', 'okrst')
        ]
        b = [0, 1]
        for i in iterator.next:
            j = (i[0], i[1])
            k = i[2]

            idx = a.index(j)
            a.pop(idx)
            idx = b.index(k)
            b.pop(idx)

        self.assertEqual(a, [])
        self.assertEqual(b, [])
        iterator.__del__()

    def test_testset_iteration(self):
        with self.assertRaises(AssertionError) as _:
            iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_very_small/data.csv', shuffle=True,
                                                  data_type='test',
                                                  src_max_len=2, tgt_max_len=2, src_max_vocab_size=100,
                                                  tgt_max_vocab_size=100, ignore_too_many_unknowns=True)
            iterator.__del__()

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
        f = open(iterator.preprocessedfname)
        x = f.read()
        f.close()

        y = 'hello world,\nokrst,\n22,\nasdf 22,\nnot seen,\n22 ghsfghda,\n'

        self.assertEqual(''.join(sorted(x)), ''.join(sorted(y)))
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
        iterator.__del__()
        train_iterator.__del__()

    def test_next_batch(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_very_small/data.csv', shuffle=True,
                                              data_type='train',
                                              src_max_len=100, tgt_max_len=100, src_max_vocab_size=100,
                                              tgt_max_vocab_size=100, ignore_too_many_unknowns=True)
        all_samples_src_raw = [
            [['22']],
            [['hello', 'world']],
            [['okrst']],
            [['asdf', '22']]
        ]

        all_samples_src = [
            torch.tensor([[5]]).cuda(),
            torch.tensor([[2, 3]]).cuda(),
            torch.tensor([[4]]).cuda(),
            torch.tensor([[6, 5]]).cuda()
        ]

        all_samples_tgt_raw = [
            [['17']],
            [['hello', 'world']],
            [['okrst']],
            [['17', 'qwe']]
        ]

        all_samples_tgt = [
            torch.tensor([[2, 7, 3]]).cuda(),
            torch.tensor([[2, 4, 5, 3]]).cuda(),
            torch.tensor([[2, 6, 3]]).cuda(),
            torch.tensor([[2, 7, 8, 3]]).cuda()
        ]

        for i in range(4):
            a = train_iterator.next_batch(1)
            src_raw = a.src_raw
            idx = all_samples_src_raw.index(src_raw)
            self.assertTrue(torch.all(torch.eq(a.src, all_samples_src[idx])))
            self.assertTrue(torch.all(torch.eq(a.tgt, all_samples_tgt[idx])))
            self.assertEqual(a.tgt_raw, all_samples_tgt_raw[idx])

            all_samples_src_raw.pop(idx)
            all_samples_src.pop(idx)
            all_samples_tgt.pop(idx)
            all_samples_tgt_raw.pop(idx)

        a = train_iterator.next_batch(1)
        self.assertEqual(a, None)
        self.assertEqual(all_samples_src, [])
        self.assertEqual(all_samples_src_raw, [])
        self.assertEqual(all_samples_tgt, [])
        self.assertEqual(all_samples_tgt_raw, [])
        # =========================================
        train_iterator.__del__()

    def test_next_batch_pruned(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_very_small/data.csv', shuffle=True,
                                              data_type='train',
                                              src_max_len=1, tgt_max_len=1, src_max_vocab_size=100,
                                              tgt_max_vocab_size=100, ignore_too_many_unknowns=True)

        all_samples_src_raw = [
            [['22']],
            [['okrst']],
        ]

        all_samples_src = [
            torch.tensor([[3]]).cuda(),
            torch.tensor([[2]]).cuda(),
        ]

        all_samples_tgt_raw = [
            [['17']],
            [['okrst']],
        ]

        all_samples_tgt = [
            torch.tensor([[2, 5, 3]]).cuda(),
            torch.tensor([[2, 4, 3]]).cuda(),
        ]

        for i in range(2):
            a = train_iterator.next_batch(1)
            src_raw = a.src_raw
            idx = all_samples_src_raw.index(src_raw)
            print(a.src)
            print(all_samples_src[idx])
            self.assertTrue(torch.all(torch.eq(a.src, all_samples_src[idx])))
            self.assertTrue(torch.all(torch.eq(a.tgt, all_samples_tgt[idx])))
            self.assertEqual(a.tgt_raw, all_samples_tgt_raw[idx])

            all_samples_src_raw.pop(idx)
            all_samples_src.pop(idx)
            all_samples_tgt.pop(idx)
            all_samples_tgt_raw.pop(idx)

        a = train_iterator.next_batch(1)
        self.assertEqual(all_samples_src, [])
        self.assertEqual(all_samples_tgt_raw, [])
        self.assertEqual(all_samples_tgt, [])
        self.assertEqual(all_samples_src_raw, [])
        self.assertEqual(a, None)
        # =========================================
        train_iterator.__del__()

    def test_next_batch_multiple_samples(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_copy/train.csv', shuffle=False,
                                                    data_type='train',
                                                    src_max_len=100, tgt_max_len=100, src_max_vocab_size=10000,
                                                    tgt_max_vocab_size=10000, ignore_too_many_unknowns=True)
        a = train_iterator.next_batch(5)
        src = a.src.data.cpu().numpy().tolist()
        req_src_raw = [
            '16 3 6 15 13 13 笑 16 8 9 10 16 10 5 8 9 7 5',
            '14 5 4 1 笑 16 12 4 13 3 10 18 7 17 8 6 18',
            '9 7 9 14 1 17 8 2 17 9 笑 15 2 12 18 12',
            '4 16 3 笑 10 笑 18 17 14 6 8 13 5 笑 11 10',
            '0 18 7 笑 8 7 14 1 12 12 10 18 3 11'
        ]
        req_src = [
            '16 3 6 15 13 13 笑 16 8 9 10 16 10 5 8 9 7 5',
            '14 5 4 1 笑 16 12 4 13 3 10 18 7 17 8 6 18 PAD',
            '9 7 9 14 1 17 8 2 17 9 笑 15 2 12 18 12 PAD PAD',
            '4 16 3 笑 10 笑 18 17 14 6 8 13 5 笑 11 10 PAD PAD',
            '0 18 7 笑 8 7 14 1 12 12 10 18 3 11 PAD PAD PAD PAD'
        ]
        for i, j, k, l in zip(src, req_src, a.src_raw, req_src_raw):
            self.assertEqual(j, train_iterator.sourceField.vec2sentence(i))
            self.assertEqual(k, l.split())
            self.assertEqual(len(i), 18)
        # =========================================
        tgt = a.tgt.data.cpu().numpy().tolist()
        req_tgt_raw = [
            '16 3 6 15 13 13 笑 16 8 9 10 16 10 5 8 9 7 5',
            '14 5 4 1 笑 16 12 4 13 3 10 18 7 17 8 6 18',
            '9 7 9 14 1 17 8 2 17 9 笑 15 2 12 18 12',
            '4 16 3 笑 10 笑 18 17 14 6 8 13 5 笑 11 10',
            '0 18 7 笑 8 7 14 1 12 12 10 18 3 11'
        ]
        req_tgt = [
            'SOS 16 3 6 15 13 13 笑 16 8 9 10 16 10 5 8 9 7 5 EOS',
            'SOS 14 5 4 1 笑 16 12 4 13 3 10 18 7 17 8 6 18 EOS PAD',
            'SOS 9 7 9 14 1 17 8 2 17 9 笑 15 2 12 18 12 EOS PAD PAD',
            'SOS 4 16 3 笑 10 笑 18 17 14 6 8 13 5 笑 11 10 EOS PAD PAD',
            'SOS 0 18 7 笑 8 7 14 1 12 12 10 18 3 11 EOS PAD PAD PAD PAD'
        ]
        for i, j, k, l in zip(tgt, req_tgt, a.tgt_raw, req_tgt_raw):
            self.assertEqual(j, train_iterator.targetField.vec2sentence(i))
            self.assertEqual(k, l.split())
            self.assertEqual(len(i), 20)
        train_iterator.__del__()

    def test_next_batch_testset(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_copy/train.csv', shuffle=False,
                                                    data_type='train',
                                                    src_max_len=100, tgt_max_len=100, src_max_vocab_size=10000,
                                                    tgt_max_vocab_size=10000, ignore_too_many_unknowns=True)

        valid_iterator = data_iterator.DataIterator(fields=(train_iterator.sourceField, train_iterator.targetField), fname='./tests/toy_copy/valid.csv', shuffle=False,
                                                    data_type='valid',
                                                    src_max_len=100, tgt_max_len=100, src_max_vocab_size=10000,
                                                    tgt_max_vocab_size=10000, ignore_too_many_unknowns=True)

        a = valid_iterator.next_batch(2)
        src = a.src.data.cpu().numpy().tolist()
        req_src_raw = [
            ['9', '18', '15', '笑', '3', '16', '2', '笑', '5', '15', '4', '4', '17', '10', '2', '1', '1', '3'],
            ['6', '13', '4', '14', '10', '4', '2', '6', '5', '4', '14']
        ]
        req_src = [
            ['9', '18', '15', '笑', '3', '16', '2', '笑', '5', '15', '4', '4', '17', '10', '2', '1', '1', '3'],
            ['6', '13', '4', '14', '10', '4', '2', '6', '5', '4', '14', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD']
        ]
        for i, j, k, l in zip(src, req_src, a.src_raw, req_src_raw):
            self.assertEqual(j, train_iterator.sourceField.vec2sentence(i).split())
            self.assertEqual(k, l)
            self.assertEqual(len(i), 18)

        train_iterator.__del__()
        valid_iterator.__del__()
#        Not required to test the tgt.


class testDataIterator1(unittest.TestCase):

    def setUp(self):
        pass

    def test_break_on_stop_iteration(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_very_small/data.csv', shuffle=False,
                                              data_type='train',
                                              src_max_len=100, tgt_max_len=100, src_max_vocab_size=100,
                                              tgt_max_vocab_size=100, ignore_too_many_unknowns=True,
                                              break_on_stop_iteration=False)
        R = random.randint(10, 20)
        for i in range(R):
            a = train_iterator.next_batch(1)
            self.assertTrue(torch.all(torch.eq(a.src, torch.tensor([[2, 3]]).cuda())))
            self.assertEqual(a.src_raw, [['hello', 'world']])
            self.assertTrue(torch.all(torch.eq(a.tgt, torch.tensor([[2, 4, 5, 3]]).cuda())))
            self.assertEqual(a.tgt_raw, [['hello', 'world']])
            # =========================================
            a = train_iterator.next_batch(1)
            self.assertTrue(torch.all(torch.eq(a.src, torch.tensor([[4]]).cuda())))
            self.assertEqual(a.src_raw, [['okrst']])
            self.assertTrue(torch.all(torch.eq(a.tgt, torch.tensor([[2, 6, 3]]).cuda())))
            self.assertEqual(a.tgt_raw, [['okrst']])
            # =========================================
            a = train_iterator.next_batch(1)

            self.assertTrue(torch.eq(a.src, torch.tensor([[5]]).cuda()))
            self.assertEqual(a.src_raw, [['22']])
            self.assertTrue(torch.all(torch.eq(a.tgt, torch.tensor([[2, 7, 3]]).cuda())))
            self.assertEqual(a.tgt_raw, [['17']])
            # =========================================
            a = train_iterator.next_batch(1)
            self.assertTrue(torch.all(torch.eq(a.src, torch.tensor([[6, 5]]).cuda())))
            self.assertEqual(a.src_raw, [['asdf', '22']])
            self.assertTrue(torch.all(torch.eq(a.tgt, torch.tensor([[2, 7, 8, 3]]).cuda())))
            self.assertEqual(a.tgt_raw, [['17', 'qwe']])
            # =========================================
        train_iterator.__del__()

    def test_break_on_stop_iteration_shuffle(self):
        train_iterator = data_iterator.DataIterator(fields=None, fname='./tests/toy_very_small/data.csv', shuffle=True,
                                              data_type='train',
                                              src_max_len=100, tgt_max_len=100, src_max_vocab_size=100,
                                              tgt_max_vocab_size=100, ignore_too_many_unknowns=True,
                                              break_on_stop_iteration=False)
        R = random.randint(10, 20)
        X = []
        Y = list([[[5]]] * R)
        Y +=  list([[[2, 3]]] * R)
        Y += list([[[4]]] * R)
        Y += list([[[6, 5]]] * R)

        for i in range(R):
            a = train_iterator.next_batch(1)
            X.append(a.src.data.cpu().numpy().tolist())
            # =========================================
            a = train_iterator.next_batch(1)
            X.append(a.src.data.cpu().numpy().tolist())
            # =========================================
            a = train_iterator.next_batch(1)
            X.append(a.src.data.cpu().numpy().tolist())
            # =========================================
            a = train_iterator.next_batch(1)
            X.append(a.src.data.cpu().numpy().tolist())
            # =========================================
        X.sort()
        Y.sort()
        self.assertEqual(X, Y)

        train_iterator.__del__()

if __name__ == '__main__':
   unittest.main()
