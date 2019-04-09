#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
tests for data_iterator.py
'''

import unittest
import data_iterator

class testLang(unittest.TestCase):

    def setUp(self):
        pass
    
    def test_vocab(self):
        src_lang = data_iterator.Lang('./tests/test_data.src', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang = data_iterator.Lang('./tests/test_data.tgt', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)
        self.assertEqual(src_lang.n_words, 4 + 2)
        self.assertEqual(tgt_lang.n_words, 3 + 4)
        self.assertEqual(src_lang.n_hashes, 16 + 8)
        self.assertEqual(tgt_lang.n_hashes, 11 + 8)

        src_lang = data_iterator.Lang('./tests/test_data.src', 10, max_word_len_allowed=1, max_vocab_size=10000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang = data_iterator.Lang('./tests/test_data.tgt', 10, max_word_len_allowed=1, max_vocab_size=10000, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)
        self.assertEqual(src_lang.n_words, 4 + 2)
        self.assertEqual(tgt_lang.n_words, 3 + 4)
        self.assertEqual(src_lang.n_hashes, 8 + 8)
        self.assertEqual(tgt_lang.n_hashes, 6 + 8)

        src_lang = data_iterator.Lang('./tests/test_data.src', 1, max_word_len_allowed=10000, max_vocab_size=10000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang = data_iterator.Lang('./tests/test_data.tgt', 1, max_word_len_allowed=10000, max_vocab_size=10000, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)
        self.assertEqual(src_lang.n_words, 2 + 2)
        self.assertEqual(tgt_lang.n_words, 2 + 4)
        self.assertEqual(src_lang.n_hashes, 10 + 8)
        self.assertEqual(tgt_lang.n_hashes, 9 + 8)

        src_lang = data_iterator.Lang('./tests/test_data.src', 10, max_word_len_allowed=10000, max_vocab_size=3, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang = data_iterator.Lang('./tests/test_data.tgt', 10, max_word_len_allowed=10000, max_vocab_size=5, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)
        self.assertEqual(src_lang.n_words, 1 + 2)
        self.assertEqual(tgt_lang.n_words, 1 + 4)
        self.assertEqual(src_lang.n_hashes, 16 + 8) #16 + 7 because, in case of source, even though the word is UNK, it's character features will be considered.
        self.assertEqual(tgt_lang.n_hashes, 5 + 8)
    
    def test_hash_sequence(self):
        src_lang = data_iterator.Lang('./tests/test_data.src', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang = data_iterator.Lang('./tests/test_data.tgt', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)
        
        # Testing word "kihog"
        got = tuple(src_lang.word2HashSequence('kihog', None, 2))
        actual = []
        cur = ['SOC'] + list('kihog'.decode('utf-8')) + ['EOC']
        for i in range(len(cur)):
            if i + 1 < len(cur):
                actual.append(src_lang.hash2idx[''.join(cur[i : i + 2])])
        while len(actual) < src_lang.max_word_len + 1:
            actual.append(src_lang.hash2idx['CPAD'])
        self.assertEqual(tuple(actual), got)

        word = ' '.join(src_lang.sequence2Hashes(list(got)))
        actual = u'SOCk ki ih ho og gEOC'
        self.assertEqual(actual, word)

        # Testing word "a"
        got = tuple(src_lang.word2HashSequence('a', None, 2))
        actual = []
        cur = ['SOC'] + list('a'.decode('utf-8')) + ['EOC']
        for i in range(len(cur)):
            if i + 1 < len(cur):
                actual.append(src_lang.hash2idx[''.join(cur[i : i + 2])])
        while len(actual) < src_lang.max_word_len + 1:
            actual.append(src_lang.hash2idx['CPAD'])
        self.assertEqual(tuple(actual), got)

        word = ' '.join(src_lang.sequence2Hashes(list(got)))
        actual = u'SOCa aEOC CPAD CPAD CPAD CPAD'
        self.assertEqual(actual, word)

        # Testing UNK token
        got = tuple(tgt_lang.word2HashSequence('UNK', None, 2))
        actual = []
        cur = ['UNK']
        actual.append(tgt_lang.hash2idx['UNK'])
        for i in range(len(cur)):
            if i + 1 < len(cur):
                actual.append(tgt_lang.hash2idx[''.join(cur[i : i + 2])])
        while len(actual) < tgt_lang.max_word_len + 1:
            actual.append(tgt_lang.hash2idx['CPAD'])
        self.assertEqual(tuple(actual), got)

        word = ' '.join(tgt_lang.sequence2Hashes(list(got)))
        actual = u'UNK CPAD CPAD CPAD CPAD CPAD'
        self.assertEqual(actual, word)

        # Testing word 'abcd'
        got = tuple(tgt_lang.word2HashSequence('abcd', None, 2))
        actual = []
        cur = ['SOC'] + list('abcd'.decode('utf-8')) + ['EOC']        
        for i in range(len(cur)):
            if i + 1 < len(cur):
                actual.append(tgt_lang.hash2idx[''.join(cur[i : i + 2])])
        while len(actual) < tgt_lang.max_word_len + 1:
            actual.append(tgt_lang.hash2idx['CPAD'])
        self.assertEqual(tuple(actual), got)

        word = ' '.join(tgt_lang.sequence2Hashes(list(got)))
        actual = u'SOCa ab bc cd dEOC CPAD'
        self.assertEqual(actual, word)

        # Testing word 'aqrd'
        got = tuple(tgt_lang.word2HashSequence('aqrd', None, 2))
        actual = []
        cur = ['SOC'] + list('aqrd'.decode('utf-8')) + ['EOC']        
        for i in range(len(cur)):
            if i + 1 < len(cur):
                try:
                    actual.append(tgt_lang.hash2idx[''.join(cur[i : i + 2])])
                except KeyError:
                    actual.append(tgt_lang.hash2idx['CUNK'])
        while len(actual) < tgt_lang.max_word_len + 1:
            actual.append(tgt_lang.hash2idx['CPAD'])
        self.assertEqual(tuple(actual), got)

        word = ' '.join(tgt_lang.sequence2Hashes(list(got)))
        actual = u'SOCa CUNK CUNK CUNK dEOC CPAD'
        self.assertEqual(actual, word)

    def test_sentence_sequence(self):
        tgt_lang = data_iterator.Lang('./tests/test_data.tgt', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)
        
        seq_words, seq_hashes = tgt_lang.sentence2Sequence('pbbbd f', add_sos = True, add_eos = True, maxlen = 0, pad = False, max_word_len = 0)
        
        #Extra CPAD at the end of below 4 sequences is because of max_length of 'abcd'
        self.assertEqual('SOS CPAD CPAD CPAD CPAD CPAD',
                        ' '.join(tgt_lang.sequence2Hashes(seq_hashes[0])))
        
        self.assertEqual('SOCp pb bb bb bd dEOC',
                        ' '.join(tgt_lang.sequence2Hashes(seq_hashes[1])))
        
        self.assertEqual('SOCf fEOC CPAD CPAD CPAD CPAD',
                        ' '.join(tgt_lang.sequence2Hashes(seq_hashes[2])))
        
        self.assertEqual('EOS CPAD CPAD CPAD CPAD CPAD',
                        ' '.join(tgt_lang.sequence2Hashes(seq_hashes[3])))
                                
        self.assertEqual('SOS pbbbd f EOS', ' '.join(tgt_lang.sequence2Sentence(seq_words)))

        seq_words, seq_hashes = tgt_lang.sentence2Sequence('pbbbd sdhfgsjf', add_sos = True, add_eos = True, maxlen = 10, pad = False, max_word_len = 0)
        self.assertEqual('SOS pbbbd UNK EOS', ' '.join(tgt_lang.sequence2Sentence(seq_words)))


class testdataIterator(unittest.TestCase):

    def setUp(self):
        pass
    
    def test_next(self):
        src_lang = data_iterator.Lang('./tests/test_data.src', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang = data_iterator.Lang('./tests/test_data.tgt', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)
        
        iterator = data_iterator.dataIterator(src_lang, tgt_lang, shuffle = True)
        j = 1
        all_sentences = []
        while True:
            try:
                all_sentences.append(iterator.next()[2])
            except Exception as e:
                self.assertEqual(str(e), 'Stop')
                break
            j += 1
        all_sentences.sort()
        self.assertEqual(j, 3)
        self.assertEqual(range(iterator.n_samples), all_sentences)
    
    def test_batch(self):
        src_lang = data_iterator.Lang('./tests/test_data.src', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang = data_iterator.Lang('./tests/test_data.tgt', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)
        
        iterator = data_iterator.dataIterator(src_lang, tgt_lang, shuffle = False)

        cur_batch = iterator.next_batch(3, None, source_language_model = src_lang, target_language_model = tgt_lang)
        self.assertEqual(cur_batch.batch_size, 2)

        cur_batch = iterator.next_batch(1, None, source_language_model = src_lang, target_language_model = tgt_lang)
        self.assertEqual(cur_batch.batch_size, 1)

        #Testing src_sentence
        ret_src = cur_batch.src.transpose(0, 1).data.cpu().numpy().tolist()[0]
        actual_src = [2, 3, 4, 5]
        self.assertEqual(ret_src, actual_src)
        self.assertEqual('kihog nbr hihoq a', ' '.join(src_lang.sequence2Sentence(ret_src)))

        #Testing src_hashes
        ret_src_hashes = cur_batch.src_hashes
        self.assertEqual(ret_src_hashes.shape, (1, 4, 6)) #batch_size x num_words x max_hash_len
        actual_hashes_src_seq = [
            [8, 9, 10, 11, 12, 13],
            [14, 15, 16, 17, 1, 1],
            [18, 19, 10, 11, 20, 21],
            [22, 23, 1, 1, 1, 1]
        ]
        for j, i in enumerate(ret_src_hashes[0]):
            self.assertEqual(actual_hashes_src_seq[j], i.data.cpu().numpy().tolist())
        
        actual_hashes_src = [
            'SOCk ki ih ho og gEOC',
            'SOCn nb br rEOC CPAD CPAD',
            'SOCh hi ih ho oq qEOC',
            'SOCa aEOC CPAD CPAD CPAD CPAD'
        ]
        for j, i in enumerate(ret_src_hashes[0]):
            self.assertEqual(actual_hashes_src[j], ' '.join(src_lang.sequence2Hashes(i.data.cpu().numpy().tolist())))

        #Testing tgt_sentence
        ret_tgt = cur_batch.tgt.transpose(0, 1).data.cpu().numpy().tolist()[0]
        actual_tgt = [2, 4, 3]
        self.assertEqual(ret_tgt, actual_tgt)
        self.assertEqual('SOS abcd EOS', ' '.join(tgt_lang.sequence2Sentence(ret_tgt)))

        #Testing tgt_hashes
        ret_tgt_hashes = cur_batch.tgt_hashes
        self.assertEqual(ret_tgt_hashes.shape, (1, 3, 6)) #batch_size x num_words x max_hash_len
        actual_hashes_tgt_seq = [
            [4, 1, 1, 1, 1, 1],
            [8, 9, 10, 11, 12, 1],
            [5, 1, 1, 1, 1, 1],
        ]
        for j, i in enumerate(ret_tgt_hashes[0]):
            self.assertEqual(actual_hashes_tgt_seq[j], i.data.cpu().numpy().tolist())
        
        actual_hashes_tgt = [
            'SOS CPAD CPAD CPAD CPAD CPAD',
            'SOCa ab bc cd dEOC CPAD',
            'EOS CPAD CPAD CPAD CPAD CPAD',
        ]
        for j, i in enumerate(ret_tgt_hashes[0]):
            self.assertEqual(actual_hashes_tgt[j], ' '.join(tgt_lang.sequence2Hashes(i.data.cpu().numpy().tolist())))
        
        self.assertEqual(cur_batch.src_len[0], 4)
        self.assertEqual(cur_batch.tgt_len[0], 3)

        self.assertEqual(' '.join(cur_batch.src_raw[0]), 'kihog nbr hihoq a')
        self.assertEqual(' '.join(cur_batch.tgt_raw[0]), 'abcd')
    
    def test_lang_not_many_unknowns(self):
        src_lang = data_iterator.Lang('./tests/train.top1000.te', 50, max_word_len_allowed=30, max_vocab_size=30000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang = data_iterator.Lang('./tests/train.top1000.hn', 50, max_word_len_allowed=30, max_vocab_size=30000, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)

        src_lang_valid = data_iterator.Lang('./tests/valid.top100.te', 50, max_word_len_allowed=30, max_vocab_size=30000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang_valid = data_iterator.Lang('./tests/valid.top100.hn', 50, max_word_len_allowed=30, max_vocab_size=30000, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)

        iterator = data_iterator.dataIterator(src_lang, tgt_lang, shuffle = False)
        
        iterator_valid = data_iterator.dataIterator(src_lang_valid, tgt_lang_valid, shuffle = False)

        batch = iterator.next_batch(1, None, source_language_model = src_lang, target_language_model = tgt_lang)
        src_got = src_lang.sequence2Sentence(batch.src.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('వారు చాలా రోజుల వరకు తక్కువ గుణంగల నూనెను UNK ఉంటారు .', ' '.join(src_got))
        self.assertEqual(batch.src_len[0], 10)
        
        req = [
            'SOCవా వారు రుEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCచా చాలా లాEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCరో రోజు జుల లEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCవ వర రకు కుEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCత తక్ క్కు కువ వEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCగు గుణం ణంగ గల లEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCనూ నూనె నెను నుEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCవా వాడు డుతూ తూEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCఉం ఉంటా టారు రుEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOC. .EOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
        ]
        for j, i in enumerate(batch.src_hashes[0]):
            ret = src_lang.sequence2Hashes(i.data.cpu().numpy().tolist())
            self.assertEqual(len(ret), src_lang.max_word_len + 1)
            self.assertLessEqual(len(ret), src_lang.max_word_len_allowed + 1)
            self.assertEqual(req[j], ' '.join(ret).encode('utf-8'))

        tgt_got = tgt_lang.sequence2Sentence(batch.tgt.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('SOS वे कई दिन तक निम्न श्रेणी के तेल बरतते रहते हैं । EOS', ' '.join(tgt_got))
        self.assertEqual(batch.tgt_len[0], 14)
        
        req = [
            'SOS CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCवे वेEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCक कई ईEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCदि दिन नEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCत तक कEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCनि निम् म्न नEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCश् श्रे रेणी णीEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCके केEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCते तेल लEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCब बर रत तते तेEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCर रह हते तेEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCहैं हैंEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOC। ।EOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'EOS CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
        ]
        for j, i in enumerate(batch.tgt_hashes[0]):
            ret = tgt_lang.sequence2Hashes(i.data.cpu().numpy().tolist())
            self.assertEqual(len(ret), tgt_lang.max_word_len + 1)
            self.assertLessEqual(len(ret), tgt_lang.max_word_len_allowed + 1)
            self.assertEqual(req[j], ' '.join(ret).encode('utf-8'))

        batch = iterator.next_batch(1, None, source_language_model = src_lang, target_language_model = tgt_lang)
        src_got = src_lang.sequence2Sentence(batch.src.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('విభిన్న లక్షణాల అనుగుణంగా హోమియోపతి వైద్యంలో ఈ రోగానికి ఔషధాలు ఈ విధంగా ఉన్నాయి .', ' '.join(src_got))
        self.assertEqual(batch.src_len[0], 12)

        tgt_got = tgt_lang.sequence2Sentence(batch.tgt.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('SOS विभिन्न लक्षणों के अनुसार होम्योपैथिक चिकित्सा में इस रोग की औषधियाँ इस प्रकार हैं । EOS', ' '.join(tgt_got))
        self.assertEqual(batch.tgt_len[0], 17)

        batch = iterator_valid.next_batch(1, None, source_language_model = src_lang, target_language_model = tgt_lang)
        src_got = src_lang.sequence2Sentence(batch.src.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('మానస జాతీయ ఉద్యానవనంలో పార్క్ 1952 లో చేయబడిన పరిశోధన ప్రకారం దట్టమైన అడవులకు నాలుగు వైపులా దాదాపు 77 గ్రాములు నెలకొని ఉన్నాయి .', ' '.join(src_got))
        self.assertEqual(batch.src_len[0], 19)

        tgt_got = tgt_lang.sequence2Sentence(batch.tgt.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('SOS मानस राष्‍ट्रीय उद्यान में वर्ष 1952 में किए गए सर्वेक्षण के अनुसार घने जंगलों के चारों और करीब 77 गाँव बसे हुए हैं । EOS', ' '.join(tgt_got))
        self.assertEqual(batch.tgt_len[0], 26)

        batch = iterator_valid.next_batch(1, None, source_language_model = src_lang, target_language_model = tgt_lang)
        src_got = src_lang.sequence2Sentence(batch.src.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('జలుబు కూడా ఒక సాధారణ సమస్య మరియు ఒకవేళ సమయానికి దానిని UNK దీనితో UNK మరియు టాన్సిల్స్ ఏర్పడతాయి .', ' '.join(src_got))
        self.assertEqual(batch.src_len[0], 16)
        
        req = [
            'SOCజ జలు లుబు బుEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCకూ కూడా డాEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCఒ ఒక కEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCసా సాధా ధార రణ ణEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCస సమ మస్ స్య యEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCమ మరి రియు యుEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCఒ ఒక కవే వేళ ళEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCస సమ మయా యాని నికి కిEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCదా దాని నిని నిEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCని నియం యంత్ త్రిం రించ చక కపో పోతే తేEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCదీ దీని నితో తోEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCన్ న్యూ యూమో మోని నియా యాEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCమ మరి రియు యుEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCటా టాన్ న్సి సిల్ ల్స్ స్EOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCఏ ఏర్ ర్ప పడ డతా తాయి యిEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOC. .EOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
        ]
        for j, i in enumerate(batch.src_hashes[0]):
            ret = src_lang.sequence2Hashes(i.data.cpu().numpy().tolist())
            self.assertEqual(len(ret), src_lang.max_word_len + 1)
            self.assertLessEqual(len(ret), src_lang.max_word_len_allowed + 1)
            self.assertEqual(req[j], ' '.join(ret).encode('utf-8'))

        tgt_got = tgt_lang.sequence2Sentence(batch.tgt.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('SOS ज़ुकाम भी एक आम समस्या है और यदि समय रहते इसे नियंत्रित नहीं किया जाता , तो उससे न्यूमोनिया और UNK हो सकते है | EOS', ' '.join(tgt_got))
        self.assertEqual(batch.tgt_len[0], 27)
        
        req = [
            'SOS CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCज़ु ज़ुका काम मEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCभी भीEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCए एक कEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCआ आम मEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCस सम मस् स्या याEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCहै हैEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCऔ और रEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCय यदि दिEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCस सम मय यEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCर रह हते तेEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCइ इसे सेEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCनि नियं यंत् त्रि रित तEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCन नहीं हींEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCकि किया याEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCजा जाता ताEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOC, ,EOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCतो तोEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCउ उस ससे सेEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCन् न्यू यूमो मोनि निया याEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCऔ और रEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'UNK CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCहो होEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCस सक कते तेEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCहै हैEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOC| |EOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'EOS CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD'
        ]
        for j, i in enumerate(batch.tgt_hashes[0]):
            ret = tgt_lang.sequence2Hashes(i.data.cpu().numpy().tolist())
            self.assertEqual(len(ret), tgt_lang.max_word_len + 1)
            self.assertLessEqual(len(ret), tgt_lang.max_word_len_allowed + 1)
            self.assertEqual(req[j], ' '.join(ret).encode('utf-8'))

    def test_lang_very_high_unknowns(self):
        src_lang = data_iterator.Lang('./tests/train.top1000.te', 50, max_word_len_allowed=30, max_vocab_size=10, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang = data_iterator.Lang('./tests/train.top1000.hn', 50, max_word_len_allowed=30, max_vocab_size=10, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)

        src_lang_valid = data_iterator.Lang('./tests/valid.top100.te', 50, max_word_len_allowed=30, max_vocab_size=30000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang_valid = data_iterator.Lang('./tests/valid.top100.hn', 50, max_word_len_allowed=30, max_vocab_size=30000, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)

        iterator = data_iterator.dataIterator(src_lang, tgt_lang, shuffle = False)
        
        iterator_valid = data_iterator.dataIterator(src_lang_valid, tgt_lang_valid, shuffle = False)

        batch = iterator.next_batch(1, None, source_language_model = src_lang, target_language_model = tgt_lang)
        src_got = src_lang.sequence2Sentence(batch.src.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('UNK UNK UNK UNK UNK UNK UNK UNK UNK .', ' '.join(src_got))
        
        req = [
            'SOCవా వారు రుEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCచా చాలా లాEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCరో రోజు జుల లEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCవ వర రకు కుEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCత తక్ క్కు కువ వEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCగు గుణం ణంగ గల లEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCనూ నూనె నెను నుEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCవా వాడు డుతూ తూEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOCఉం ఉంటా టారు రుEOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
            'SOC. .EOC CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD CPAD',
        ]
        for j, i in enumerate(batch.src_hashes[0]):
            ret = src_lang.sequence2Hashes(i.data.cpu().numpy().tolist())
            self.assertEqual(len(ret), src_lang.max_word_len + 1)
            self.assertLessEqual(len(ret), src_lang.max_word_len_allowed + 1)
            self.assertEqual(req[j], ' '.join(ret).encode('utf-8'))
        
        tgt_got = tgt_lang.sequence2Sentence(batch.tgt.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('SOS UNK UNK UNK UNK UNK UNK के UNK UNK UNK UNK । EOS', ' '.join(tgt_got))
        req = [
            'SOS CPAD',
            'UNK CPAD',
            'UNK CPAD',
            'UNK CPAD',
            'UNK CPAD',
            'UNK CPAD',
            'UNK CPAD',
            'SOCके केEOC',
            'UNK CPAD',
            'UNK CPAD',
            'UNK CPAD',
            'UNK CPAD',
            'SOC। ।EOC',
            'EOS CPAD',
        ]
        for j, i in enumerate(batch.tgt_hashes[0]):
            ret = tgt_lang.sequence2Hashes(i.data.cpu().numpy().tolist())
            self.assertEqual(len(ret), tgt_lang.max_word_len + 1)
            self.assertLessEqual(len(ret), tgt_lang.max_word_len_allowed + 1)
            self.assertEqual(req[j], ' '.join(ret).encode('utf-8'))
    
        batch = iterator.next_batch(1, None, source_language_model = src_lang, target_language_model = tgt_lang)
        src_got = src_lang.sequence2Sentence(batch.src.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('UNK UNK UNK UNK UNK ఈ UNK UNK ఈ UNK UNK .', ' '.join(src_got))

        tgt_got = tgt_lang.sequence2Sentence(batch.tgt.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('SOS UNK UNK के UNK UNK UNK में UNK UNK UNK UNK UNK UNK UNK । EOS', ' '.join(tgt_got))

        batch = iterator_valid.next_batch(1, None, source_language_model = src_lang, target_language_model = tgt_lang)
        src_got = src_lang.sequence2Sentence(batch.src.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK .', ' '.join(src_got))

        tgt_got = tgt_lang.sequence2Sentence(batch.tgt.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('SOS UNK UNK UNK में UNK UNK में UNK UNK UNK के UNK UNK UNK के UNK UNK UNK UNK UNK UNK UNK UNK । EOS', ' '.join(tgt_got))

        batch = iterator_valid.next_batch(1, None, source_language_model = src_lang, target_language_model = tgt_lang)
        src_got = src_lang.sequence2Sentence(batch.src.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('UNK కూడా ఒక UNK UNK మరియు UNK UNK UNK UNK UNK UNK మరియు UNK UNK .', ' '.join(src_got))

        tgt_got = tgt_lang.sequence2Sentence(batch.tgt.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('SOS UNK UNK UNK UNK UNK है UNK UNK UNK UNK UNK UNK UNK UNK UNK , UNK UNK UNK UNK UNK UNK UNK है UNK EOS', ' '.join(tgt_got))
    
    def test_lang_no_unknowns(self):
        src_lang = data_iterator.Lang('./tests/train.top1000.te', 50, max_word_len_allowed=30, max_vocab_size=100000000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang = data_iterator.Lang('./tests/train.top1000.hn', 50, max_word_len_allowed=30, max_vocab_size=100000000, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)

        src_lang_valid = data_iterator.Lang('./tests/valid.top100.te', 50, max_word_len_allowed=30, max_vocab_size=30000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang_valid = data_iterator.Lang('./tests/valid.top100.hn', 50, max_word_len_allowed=30, max_vocab_size=30000, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)

        iterator = data_iterator.dataIterator(src_lang, tgt_lang, shuffle = False)
        
        iterator_valid = data_iterator.dataIterator(src_lang_valid, tgt_lang_valid, shuffle = False)

        batch = iterator.next_batch(1, None, source_language_model = src_lang, target_language_model = tgt_lang)
        src_got = src_lang.sequence2Sentence(batch.src.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('వారు చాలా రోజుల వరకు తక్కువ గుణంగల నూనెను వాడుతూ ఉంటారు .', ' '.join(src_got))

        tgt_got = tgt_lang.sequence2Sentence(batch.tgt.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('SOS वे कई दिन तक निम्न श्रेणी के तेल बरतते रहते हैं । EOS', ' '.join(tgt_got))

        batch = iterator.next_batch(1, None, source_language_model = src_lang, target_language_model = tgt_lang)
        src_got = src_lang.sequence2Sentence(batch.src.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('విభిన్న లక్షణాల అనుగుణంగా హోమియోపతి వైద్యంలో ఈ రోగానికి ఔషధాలు ఈ విధంగా ఉన్నాయి .', ' '.join(src_got))

        tgt_got = tgt_lang.sequence2Sentence(batch.tgt.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('SOS विभिन्न लक्षणों के अनुसार होम्योपैथिक चिकित्सा में इस रोग की औषधियाँ इस प्रकार हैं । EOS', ' '.join(tgt_got))

        batch = iterator_valid.next_batch(1, None, source_language_model = src_lang, target_language_model = tgt_lang)
        src_got = src_lang.sequence2Sentence(batch.src.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('మానస జాతీయ ఉద్యానవనంలో పార్క్ 1952 లో చేయబడిన పరిశోధన ప్రకారం దట్టమైన అడవులకు నాలుగు వైపులా దాదాపు 77 గ్రాములు నెలకొని ఉన్నాయి .', ' '.join(src_got))

        tgt_got = tgt_lang.sequence2Sentence(batch.tgt.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('SOS मानस राष्‍ट्रीय उद्यान में वर्ष 1952 में किए गए सर्वेक्षण के अनुसार घने जंगलों के चारों और करीब 77 गाँव बसे हुए हैं । EOS', ' '.join(tgt_got))

        batch = iterator_valid.next_batch(1, None, source_language_model = src_lang, target_language_model = tgt_lang)
        src_got = src_lang.sequence2Sentence(batch.src.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('జలుబు కూడా ఒక సాధారణ సమస్య మరియు ఒకవేళ సమయానికి దానిని UNK దీనితో న్యూమోనియా మరియు టాన్సిల్స్ ఏర్పడతాయి .', ' '.join(src_got))

        tgt_got = tgt_lang.sequence2Sentence(batch.tgt.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('SOS ज़ुकाम भी एक आम समस्या है और यदि समय रहते इसे नियंत्रित नहीं किया जाता , तो उससे न्यूमोनिया और UNK हो सकते है | EOS', ' '.join(tgt_got))
    
    def test_lang_unknowns_raise_exception(self):
        with self.assertRaises(Exception) as context: 
            _ = data_iterator.Lang('./tests/train.top1000.te', 50, max_word_len_allowed=30, max_vocab_size=100000000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=False)
        self.assertTrue('Very few unknowns. Please decrease vocabulary size' in str(context.exception))
        
        with self.assertRaises(Exception) as context:
            _ = data_iterator.Lang('./tests/train.top1000.hn', 50, max_word_len_allowed=30, max_vocab_size=1, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=False)
        self.assertTrue('Too many unknowns. Please increase vocabulary size' in str(context.exception))

        try:
            src_lang_valid = data_iterator.Lang('./tests/valid.top100.te', 50, max_word_len_allowed=30, max_vocab_size=30000000000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
            tgt_lang_valid = data_iterator.Lang('./tests/valid.top100.hn', 50, max_word_len_allowed=30, max_vocab_size=1, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)
        except Exception, error:
            self.fail('Exception raised with %s' % error)
        
if __name__ == '__main__':
   unittest.main()
