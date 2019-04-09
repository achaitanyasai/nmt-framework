#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
tests for data_iterator_optimized.py
'''

import tqdm
import unittest
import data_iterator_optimized

class testLang(unittest.TestCase):

    def setUp(self):
        pass
    
    def test_vocab(self):
        src_lang = data_iterator_optimized.Lang('./tests/test_data.src', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang = data_iterator_optimized.Lang('./tests/test_data.tgt', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)
        self.assertEqual(src_lang.n_words, 4 + 2)
        self.assertEqual(tgt_lang.n_words, 3 + 4)

        # src_lang = data_iterator_optimized.Lang('/tmp/train.en', 10, max_word_len_allowed=20, max_vocab_size=30000, langtype='source', chars = False, verbose=True, ignore_too_many_unknowns=True)
        # tgt_lang = data_iterator_optimized.Lang('/tmp/train.de', 10, max_word_len_allowed=20, max_vocab_size=30000, langtype='target', chars = False, verbose=True, ignore_too_many_unknowns=True)
        # self.assertEqual(src_lang.n_words, 30000)
        # self.assertEqual(tgt_lang.n_words, 30000)
        # a = [i for i in range(src_lang.n_sentences)]
        # import random
        # random.shuffle(a)
        # b = {}
        # k = 0
        # for i, j in tqdm.tqdm(zip(src_lang.file_iterator, tgt_lang.file_iterator)):
        #     try:
        #         b[i].append(j)
        #     except KeyError:
        #         b[i] = [j]
        # src_lang.reset(a)
        # tgt_lang.reset(a)
        # for i, j in tqdm.tqdm(zip(src_lang.file_iterator, tgt_lang.file_iterator)):
        #     self.assertIn(j, b[i])
        
        # src_lang = data_iterator_optimized.Lang('/tmp/train.de', 10, max_word_len_allowed=20, max_vocab_size=30000, langtype='source', chars = False, verbose=True, ignore_too_many_unknowns=True)
        # tgt_lang = data_iterator_optimized.Lang('/tmp/train.en', 10, max_word_len_allowed=20, max_vocab_size=30000, langtype='target', chars = False, verbose=True, ignore_too_many_unknowns=True)
        # self.assertEqual(src_lang.n_words, 30000)
        # self.assertEqual(tgt_lang.n_words, 30000)
        # a = [i for i in range(src_lang.n_sentences)]
        # import random
        # random.shuffle(a)
        # b = {}
        # k = 0
        # for i, j in tqdm.tqdm(zip(src_lang.file_iterator, tgt_lang.file_iterator)):
        #     try:
        #         b[i].append(j)
        #     except KeyError:
        #         b[i] = [j]
        # src_lang.reset(a)
        # tgt_lang.reset(a)
        # for i, j in tqdm.tqdm(zip(src_lang.file_iterator, tgt_lang.file_iterator)):
        #     self.assertIn(j, b[i])

    def test_sentence_sequence(self):
        tgt_lang = data_iterator_optimized.Lang('./tests/test_data.tgt', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)
        
        seq_words = tgt_lang.sentence2Sequence('pbbbd f', add_sos = True, add_eos = True, maxlen = 0, pad = False)
                                
        self.assertEqual('SOS pbbbd f EOS', ' '.join(tgt_lang.sequence2Sentence(seq_words)))

        seq_words = tgt_lang.sentence2Sequence('pbbbd sdhfgsjf', add_sos = True, add_eos = True, maxlen = 10, pad = False)
        self.assertEqual('SOS pbbbd UNK EOS', ' '.join(tgt_lang.sequence2Sentence(seq_words)))
        pass


class testdataIterator(unittest.TestCase):

    def setUp(self):
        pass
    
    def test_next(self):
        src_lang = data_iterator_optimized.Lang('./tests/test_data.src', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang = data_iterator_optimized.Lang('./tests/test_data.tgt', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)
        
        iterator = data_iterator_optimized.dataIterator(src_lang, tgt_lang, shuffle = True)
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
        src_lang = data_iterator_optimized.Lang('./tests/test_data.src', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang = data_iterator_optimized.Lang('./tests/test_data.tgt', 10, max_word_len_allowed=20, max_vocab_size=10000, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)
        
        iterator = data_iterator_optimized.dataIterator(src_lang, tgt_lang, shuffle = False)

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

        #Testing tgt_sentence
        ret_tgt = cur_batch.tgt.transpose(0, 1).data.cpu().numpy().tolist()[0]
        actual_tgt = [2, 4, 3]
        self.assertEqual(ret_tgt, actual_tgt)
        self.assertEqual('SOS abcd EOS', ' '.join(tgt_lang.sequence2Sentence(ret_tgt)))

        self.assertEqual(cur_batch.src_len[0], 4)
        self.assertEqual(cur_batch.tgt_len[0], 3)

        self.assertEqual(' '.join(cur_batch.src_raw[0]), 'kihog nbr hihoq a')
        self.assertEqual(' '.join(cur_batch.tgt_raw[0]), 'abcd')
    
    def test_lang_not_many_unknowns(self):
        src_lang = data_iterator_optimized.Lang('./tests/train.top1000.te', 50, max_word_len_allowed=30, max_vocab_size=30000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang = data_iterator_optimized.Lang('./tests/train.top1000.hn', 50, max_word_len_allowed=30, max_vocab_size=30000, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)

        src_lang_valid = data_iterator_optimized.Lang('./tests/valid.top100.te', 50, max_word_len_allowed=30, max_vocab_size=30000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang_valid = data_iterator_optimized.Lang('./tests/valid.top100.hn', 50, max_word_len_allowed=30, max_vocab_size=30000, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)

        iterator = data_iterator_optimized.dataIterator(src_lang, tgt_lang, shuffle = False)
        
        iterator_valid = data_iterator_optimized.dataIterator(src_lang_valid, tgt_lang_valid, shuffle = False)

        batch = iterator.next_batch(1, None, source_language_model = src_lang, target_language_model = tgt_lang)
        src_got = src_lang.sequence2Sentence(batch.src.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('వారు చాలా రోజుల వరకు తక్కువ గుణంగల నూనెను UNK ఉంటారు .', ' '.join(src_got))
        self.assertEqual(batch.src_len[0], 10)
        
        tgt_got = tgt_lang.sequence2Sentence(batch.tgt.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('SOS वे कई दिन तक निम्न श्रेणी के तेल बरतते रहते हैं । EOS', ' '.join(tgt_got))
        self.assertEqual(batch.tgt_len[0], 14)
        
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
        
        tgt_got = tgt_lang.sequence2Sentence(batch.tgt.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('SOS ज़ुकाम भी एक आम समस्या है और यदि समय रहते इसे नियंत्रित नहीं किया जाता , तो उससे न्यूमोनिया और UNK हो सकते है | EOS', ' '.join(tgt_got))
        self.assertEqual(batch.tgt_len[0], 27)
        
    def test_lang_very_high_unknowns(self):
        src_lang = data_iterator_optimized.Lang('./tests/train.top1000.te', 50, max_word_len_allowed=30, max_vocab_size=10, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang = data_iterator_optimized.Lang('./tests/train.top1000.hn', 50, max_word_len_allowed=30, max_vocab_size=10, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)

        src_lang_valid = data_iterator_optimized.Lang('./tests/valid.top100.te', 50, max_word_len_allowed=30, max_vocab_size=30000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang_valid = data_iterator_optimized.Lang('./tests/valid.top100.hn', 50, max_word_len_allowed=30, max_vocab_size=30000, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)

        iterator = data_iterator_optimized.dataIterator(src_lang, tgt_lang, shuffle = False)
        
        iterator_valid = data_iterator_optimized.dataIterator(src_lang_valid, tgt_lang_valid, shuffle = False)

        batch = iterator.next_batch(1, None, source_language_model = src_lang, target_language_model = tgt_lang)
        src_got = src_lang.sequence2Sentence(batch.src.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('UNK UNK UNK UNK UNK UNK UNK UNK UNK .', ' '.join(src_got))
        
        tgt_got = tgt_lang.sequence2Sentence(batch.tgt.transpose(0, 1).data.cpu().numpy().tolist()[0])
        self.assertEqual('SOS UNK UNK UNK UNK UNK UNK के UNK UNK UNK UNK । EOS', ' '.join(tgt_got))
        
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
        src_lang = data_iterator_optimized.Lang('./tests/train.top1000.te', 50, max_word_len_allowed=30, max_vocab_size=100000000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang = data_iterator_optimized.Lang('./tests/train.top1000.hn', 50, max_word_len_allowed=30, max_vocab_size=100000000, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)

        src_lang_valid = data_iterator_optimized.Lang('./tests/valid.top100.te', 50, max_word_len_allowed=30, max_vocab_size=30000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
        tgt_lang_valid = data_iterator_optimized.Lang('./tests/valid.top100.hn', 50, max_word_len_allowed=30, max_vocab_size=30000, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)

        iterator = data_iterator_optimized.dataIterator(src_lang, tgt_lang, shuffle = False)
        
        iterator_valid = data_iterator_optimized.dataIterator(src_lang_valid, tgt_lang_valid, shuffle = False)

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
            _ = data_iterator_optimized.Lang('./tests/train.top1000.te', 50, max_word_len_allowed=30, max_vocab_size=100000000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=False)
        self.assertTrue('Very few unknowns. Please decrease vocabulary size' in str(context.exception))
        
        with self.assertRaises(Exception) as context:
            _ = data_iterator_optimized.Lang('./tests/train.top1000.hn', 50, max_word_len_allowed=30, max_vocab_size=1, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=False)
        self.assertTrue('Too many unknowns. Please increase vocabulary size' in str(context.exception))

        try:
            src_lang_valid = data_iterator_optimized.Lang('./tests/valid.top100.te', 50, max_word_len_allowed=30, max_vocab_size=30000000000, langtype='source', chars = False, verbose=False, ignore_too_many_unknowns=True)
            tgt_lang_valid = data_iterator_optimized.Lang('./tests/valid.top100.hn', 50, max_word_len_allowed=30, max_vocab_size=1, langtype='target', chars = False, verbose=False, ignore_too_many_unknowns=True)
        except Exception, error:
            self.fail('Exception raised with %s' % error)
        
if __name__ == '__main__':
   unittest.main()
