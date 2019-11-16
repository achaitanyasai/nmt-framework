import torch
from torch.autograd import Variable

from . import Beam
from . import utils
from structs import *


class Translator(object):
    def __init__(self, model, source_data, target_data, test_data_iterator,
                 beam_size, n_best=1, max_length=50, global_scorer=None,
                 copy_attn=False, cuda=True, beam_trace=False, min_length=0):
        self.model = model
        self.source_data = source_data
        self.target_data = target_data
        self.test_data_iterator = test_data_iterator
        self.beam_size = beam_size
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.cuda = cuda
        self.beam_trace = beam_trace
        self.min_length = min_length
    
    # def get_char_features(self, inp):
    #     res = []
    #     res_len = []
    #
    #     inp = inp.transpose(0, 1)
    #     for i in inp:
    #         idx = i.data.cpu().numpy()[0]
    #         word = self.target_data.idx2word[idx]
    #         if(word == ''):
    #             word = 'UNK'
    #         ret = self.target_data.word2HashSequence(word, None, 2)
    #         res.append(ret)
    #
    #     res = torch.LongTensor(res, requires_grad=False).cuda()
    #     res = res.unsqueeze(0)
    #     res = res.transpose(0, 1)
    #     return res

    def translate_batch(self, batch):
        batch.transpose()
        src, src_lengths = batch.src, batch.src_lens
        lines_src_hashes = None

        beam_size = self.beam_size
        batch_size = batch.batch_size
        assert(batch_size == src.size(1)) # seq len x batch size
        vocab = self.target_data.word2idx
        beam = [Beam.Beam(beam_size, n_best=self.n_best, cuda=self.cuda,
                     global_scorer=self.global_scorer, pad=vocab['PAD'],
                     eos=vocab['EOS'], bos=vocab['SOS'], min_length=self.min_length)
                for _ in range(batch_size)]

        enc_hidden, context, penalty, encoder_embeddings = self.model.encoder(src, None, src_lengths)
        encoderOutputs = EncoderOutputs(enc_hidden, context, encoder_embeddings)
        dec_states = self.model.decoder.init_decoder_state(src, context, enc_hidden)

        # dec_states = self.model.decoder.init_decoder_state(src, context, enc_states)
        
        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(context.data).long().fill_(context.size(0))
                
        context = utils.rvar(context.data, beam_size)
        enc_rnn_input = utils.rvar(encoder_embeddings.data, beam_size)
        context_lengths = src_lengths.repeat(beam_size)
        dec_states.repeat_beam_size_times(beam_size)

        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            inp = utils.var(torch.stack([b.get_current_state() for b in beam]).t().contiguous().view(1, -1))
            
            if self.copy_attn:
                inp = inp.masked_fill(inp.gt(len(vocab) - 1), 0)
                raise Exception('Unimplemented')
                #Unimplemented
            
            char_inp = None#self.get_char_features(inp)

            dec_out, dec_states, attn, _ = self.model.decoder(inp, None, context,
                                                              enc_rnn_input, dec_states,
                                                              context_lengths=context_lengths)
            # dec_out, dec_states, attn, _ = self.model.decoder(inp, char_inp, enc_rnn_input, context, dec_states, context_lengths = context_lengths)
            dec_out = dec_out.squeeze(0)

            if not self.copy_attn:
                out = self.model.decoder.decoder2vocab.forward(dec_out).data
                out = utils.unbottle(out, beam_size, batch_size)
            else:
                #Unimplemented
                raise Exception('Unimplemented')
            
            for j, b in enumerate(beam):
                b.advance(out[:, j], utils.unbottle(attn['std'], beam_size, batch_size).data[:, j, :context_lengths[j]])
                dec_states.beam_update(j, b.get_current_origin(), beam_size)
        
        ret = self._from_beam(beam)
        ret['gold_score'] = [0] * batch_size
        return ret
    
    def _from_beam(self, beam):
        ret = {
            'predictions': [],
            'scores': [],
            'attention':[]
        }
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[: n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret['predictions'].append(hyps)
            ret['scores'].append(scores)
            ret['attention'].append(attn)
        return ret

class Translation(object):
    def __init__(self, src, src_raw, pred_sents, attn, pred_scores, tgt_sent, gold_score):
        self.src = src
        self.src_raw = src_raw
        self.src_raw = self.src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent =tgt_sent
        self.gold_score = gold_score

class TranslationBuilder(object):
    def __init__(self, source_data, target_data, test_data_iterator,
                 n_best, replace_unk):
        self.source_data = source_data
        self.target_data = target_data
        self.test_data_iterator = test_data_iterator
        self.n_best = n_best
        self.replace_unk = replace_unk
    
    def _build_target_tokens(self, src, src_raw, pred, attn):
        vocab = self.target_data.idx2word
        tokens = []
        for tok in pred:
            try:
                tokens.append(vocab[tok.item()])
            except KeyError:
                #Can be done better (??)
                #Refer: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/Translation.py#L36
                tokens.append('UNK')
            if tokens[-1] == 'EOS':
                tokens = tokens[:-1]
                break
        if self.replace_unk:
            assert(attn is not None)
            assert(src is not None)
            for i in range(len(tokens)):
                if tokens[i] == 'UNK':
                    _, max_index = attn[i].max(0)
                    tokens[i] = src_raw[max_index.item()]
        return tokens
    
    def from_batch(self, translation_batch, batch):
        assert(len(translation_batch['gold_score']) == len(translation_batch['predictions']))
        
        src, src_lengths = batch.src, batch.src_lens

        batch_size = batch.batch_size
        assert(src.size(1) == batch_size)

        preds, pred_score, attn, gold_score, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["gold_score"],
                        batch.indices),
                    key=lambda x: x[-1])))
        inds, perm = torch.sort(torch.tensor(batch.indices).cuda())
        src = src.data.index_select(1, perm)
        translations = []
        
        for b in range(batch_size):
            src_raw = batch.src_raw[perm[b]]
            pred_sents = [self._build_target_tokens(src[:, b], src_raw, preds[b][n], attn[b][n]) for n in range(self.n_best)]

            translation = Translation(src[:, b], src_raw, pred_sents, attn[b], pred_score[b], None, 0)

            translations.append(translation)
        return translations
