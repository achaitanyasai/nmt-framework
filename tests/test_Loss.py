import unittest
from modules import Loss
import torch
import torch.nn.functional as F
import data_iterator
import structs
torch.set_printoptions(precision=10)

class TestLossAndAccuracy(unittest.TestCase):
    def test_loss_with_zero_accuracy(self):
        _loss = Loss.NMTLoss(target_vocabulary_len=7, target_padding_idx=1,
                               reduction='sum', perform_dimension_checks=True).cuda()

        # src shape: batch_size x seq_len
        src = torch.tensor(
            [
                [2, 3, 4, 5],
                [3, 4, 1, 1]
            ], dtype=torch.int64, requires_grad=False
        ).cuda()

        # tgt shape: batch_size x seq_len
        tgt = torch.tensor(
            [
                [2, 4, 5, 3],
                [2, 4, 4, 3],
            ], dtype=torch.int64, requires_grad=False
        ).cuda()

        # predicted shape: batch_size x seq_len x tgt_vocab_size.
        # Note that (seq_len = tgt_seq_len - 1) because there is no SOS in predictions.

        predicted = torch.tensor(
            [
                [
                    [0.27578739731450835, 0.28015944483463034, 0.1142575644358387, 0.10971578192026918, 0.05760354961409472, 0.0958341363446243, 0.06664212553603428],
                    [0.11599556338750745, 0.20765450527315263, 0.17882063754508296, 0.09451736301664149, 0.21279142684705998, 0.010456875336914199, 0.17976362859364128],
                    [0.16634002261487138, 0.05109615271682168, 0.2130510680478736, 0.04501177173241767, 0.09622936919613069, 0.03387612952910475, 0.3943954861627801]
                ],
                [
                    [0.2367089044466947, 0.3362183043774197, 0.052103545792205254, 0.029796893619137815, 0.16104667186265817, 0.09548056745840004, 0.08864511244348429],
                    [0.18377649152812464, 0.09355287186688058, 0.1333162530259955, 0.1720949411050521, 0.09800385744309532, 0.05851594270914497, 0.260739642321707],
                    [0.16201767024447655, 0.17549039629714336, 0.051250868614221234, 0.140254556612448, 0.006261768243946401, 0.22618242337226518, 0.23854231661549935]
                ],
            ], dtype=torch.float32, requires_grad=False
        ).cuda()

        self.assertEqual(src.shape[0], 2)
        self.assertEqual(src.shape[1], 4)

        self.assertEqual(tgt.shape[0], 2)
        self.assertEqual(tgt.shape[1], 4)

        self.assertEqual(predicted.shape[0], 2)
        self.assertEqual(predicted.shape[1], 3)
        self.assertEqual(predicted.shape[2], 7)

        predicted = F.log_softmax(predicted, dim=2)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        predicted = predicted.transpose(0, 1)
        batch = data_iterator.BatchData(src, tgt, None, None, None, None, None, batch_size=2)
        modelOutputs = structs.ModelOutputs(predictions=predicted, penalty=None)
        loss, stats = _loss.compute_loss(batch, modelOutputs, cur_step=1)

        self.assertEqual(stats.accuracy(), 0.0)
        x = predicted[0][0][4] + predicted[0][1][4] # 4, 4
        x += predicted[1][0][5] + predicted[1][1][4] # 5, 5
        x += predicted[2][0][3] + predicted[2][1][3] # 3, 3
        x = x.item()

        self.assertEqual(stats._loss(), - x / 6.0)
        self.assertEqual(stats.n_words, 6)

    def test_loss_with_100_accuracy(self):
        lossObj = Loss.NMTLoss(target_vocabulary_len=7, target_padding_idx=1,
                               reduction='sum', perform_dimension_checks=True).cuda()

        # src shape: batch_size x seq_len
        src = torch.tensor(
            [
                [2, 3, 4, 5],
                [3, 4, 1, 1]
            ], dtype=torch.int64, requires_grad=False
        ).cuda()

        # tgt shape: batch_size x seq_len
        tgt = torch.tensor(
            [
                [2, 4, 5, 3],
                [2, 4, 4, 3],
            ], dtype=torch.int64, requires_grad=False
        ).cuda()

        # predicted shape: batch_size x seq_len x tgt_vocab_size.
        # Note that (seq_len = tgt_seq_len - 1) because there is no SOS in predictions.

        predicted = torch.tensor(
            [
                [
                    [0.27578739731450835, 0.28015944483463034, 0.1142575644358387, 0.10971578192026918, 0.55760354961409472, 0.0958341363446243, 0.06664212553603428],
                    [0.11599556338750745, 0.20765450527315263, 0.17882063754508296, 0.09451736301664149, 0.21279142684705998, 0.810456875336914199, 0.17976362859364128],
                    [0.16634002261487138, 0.05109615271682168, 0.2130510680478736, 0.94501177173241767, 0.09622936919613069, 0.03387612952910475, 0.3943954861627801]
                ],
                [
                    [0.2367089044466947, 0.3362183043774197, 0.052103545792205254, 0.029796893619137815, 0.96104667186265817, 0.09548056745840004, 0.08864511244348429],
                    [0.18377649152812464, 0.09355287186688058, 0.1333162530259955, 0.1720949411050521, 0.99800385744309532, 0.05851594270914497, 0.260739642321707],
                    [0.16201767024447655, 0.17549039629714336, 0.051250868614221234, 0.940254556612448, 0.006261768243946401, 0.22618242337226518, 0.23854231661549935]
                ],
            ], dtype=torch.float32, requires_grad=False
        ).cuda()

        self.assertEqual(src.shape[0], 2)
        self.assertEqual(src.shape[1], 4)

        self.assertEqual(tgt.shape[0], 2)
        self.assertEqual(tgt.shape[1], 4)

        self.assertEqual(predicted.shape[0], 2)
        self.assertEqual(predicted.shape[1], 3)
        self.assertEqual(predicted.shape[2], 7)

        predicted = F.log_softmax(predicted, dim=2)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        predicted = predicted.transpose(0, 1)
        batch = data_iterator.BatchData(src, tgt, None, None, None, None, None, batch_size=2)
        modelOutputs = structs.ModelOutputs(predictions=predicted, penalty=None)
        loss, stats = lossObj.compute_loss(batch, modelOutputs, cur_step=1)

        self.assertEqual(stats.accuracy(), 100.0)
        a = torch.tensor(
            [
                predicted[0][0][4], predicted[0][1][4],
                predicted[1][0][5], predicted[1][1][4],
                predicted[2][0][3], predicted[2][1][3]

            ], dtype=torch.float32, requires_grad=False
        )
        x = torch.sum(a).item()

        self.assertEqual(stats._loss(), - x / 6.0)
        self.assertEqual(stats.n_words, 6)

    def test_loss_with_pad_100_accuracy(self):
        lossObj = Loss.NMTLoss(target_vocabulary_len=7, target_padding_idx=1,
                               reduction='sum', perform_dimension_checks=True).cuda()

        # src shape: batch_size x seq_len
        src = torch.tensor(
            [
                [2, 3, 4, 5],
                [3, 4, 1, 1]
            ], dtype=torch.int64, requires_grad=False
        ).cuda()

        # tgt shape: batch_size x seq_len
        tgt = torch.tensor(
            [
                [2, 4, 5, 6, 3],
                [2, 4, 4, 3, 1],
            ], dtype=torch.int64, requires_grad=False
        ).cuda()

        # predicted shape: batch_size x seq_len x tgt_vocab_size.
        # Note that (seq_len = tgt_seq_len - 1) because there is no SOS in predictions.

        predicted = torch.tensor(
            [
                [
                    [0.27578739731450835, 0.28015944483463034, 0.1142575644358387, 0.10971578192026918, 0.55760354961409472, 0.0958341363446243, 0.06664212553603428],
                    [0.11599556338750745, 0.20765450527315263, 0.17882063754508296, 0.09451736301664149, 0.21279142684705998, 0.810456875336914199, 0.17976362859364128],
                    [0.16634002261487138, 0.05109615271682168, 0.2130510680478736, 0.14501177173241767, 0.09622936919613069, 0.03387612952910475, 0.9943954861627801],
                    [0.16634002261487138, 0.05109615271682168, 0.2130510680478736, 0.94501177173241767, 0.09622936919613069, 0.03387612952910475, 0.3943954861627801]
                ],
                [
                    [0.2367089044466947, 0.3362183043774197, 0.052103545792205254, 0.029796893619137815, 0.96104667186265817, 0.09548056745840004, 0.08864511244348429],
                    [0.18377649152812464, 0.09355287186688058, 0.1333162530259955, 0.1720949411050521, 0.99800385744309532, 0.05851594270914497, 0.260739642321707],
                    [0.16201767024447655, 0.17549039629714336, 0.051250868614221234, 0.940254556612448, 0.006261768243946401, 0.22618242337226518, 0.23854231661549935],
                    [0.16201767024447655, 0.12549039629714336, 0.951250868614221234, 0.040254556612448, 0.006261768243946401, 0.22618242337226518, 0.23854231661549935]
                ],
            ], dtype=torch.float32, requires_grad=False
        ).cuda()

        self.assertEqual(src.shape[0], 2)
        self.assertEqual(src.shape[1], 4)

        self.assertEqual(tgt.shape[0], 2)
        self.assertEqual(tgt.shape[1], 5)

        self.assertEqual(predicted.shape[0], 2)
        self.assertEqual(predicted.shape[1], 4)
        self.assertEqual(predicted.shape[2], 7)

        predicted = F.log_softmax(predicted, dim=2)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        predicted = predicted.transpose(0, 1)
        batch = data_iterator.BatchData(src, tgt, None, None, None, None, None, batch_size=2)
        modelOutputs = structs.ModelOutputs(predictions=predicted, penalty=None)
        loss, stats = lossObj.compute_loss(batch, modelOutputs, cur_step=1)

        self.assertEqual(stats.accuracy(), 100.0)
        a = torch.tensor(
            [
                predicted[0][0][4], predicted[0][1][4],
                predicted[1][0][5], predicted[1][1][4],
                predicted[2][0][6], predicted[2][1][3],
                predicted[3][0][3]

            ], dtype=torch.float32, requires_grad=False
        )
        x = torch.sum(a).item()

        self.assertEqual(stats._loss(), - x / 7.0)
        self.assertEqual(stats.n_words, 7)

    def test_loss_with_pad_100_accuracy1(self):
        lossObj = Loss.NMTLoss(target_vocabulary_len=7, target_padding_idx=1,
                               reduction='sum', perform_dimension_checks=True).cuda()

        # src shape: batch_size x seq_len
        src = torch.tensor(
            [
                [2, 3, 4, 5],
                [3, 4, 1, 1]
            ], dtype=torch.int64, requires_grad=False
        ).cuda()

        # tgt shape: batch_size x seq_len
        tgt = torch.tensor(
            [
                [2, 4, 5, 6, 3],
                [2, 4, 4, 3, 1],
            ], dtype=torch.int64, requires_grad=False
        ).cuda()

        # predicted shape: batch_size x seq_len x tgt_vocab_size.
        # Note that (seq_len = tgt_seq_len - 1) because there is no SOS in predictions.

        predicted = torch.tensor(
            [
                [
                    [0.27578739731450835, 0.28015944483463034, 0.1142575644358387, 0.10971578192026918, 0.55760354961409472, 0.0958341363446243, 0.06664212553603428],
                    [0.11599556338750745, 0.20765450527315263, 0.17882063754508296, 0.09451736301664149, 0.21279142684705998, 0.810456875336914199, 0.17976362859364128],
                    [0.16634002261487138, 0.05109615271682168, 0.2130510680478736, 0.14501177173241767, 0.09622936919613069, 0.03387612952910475, 0.9943954861627801],
                    [0.16634002261487138, 0.05109615271682168, 0.2130510680478736, 0.94501177173241767, 0.09622936919613069, 0.03387612952910475, 0.3943954861627801]
                ],
                [
                    [0.2367089044466947, 0.3362183043774197, 0.052103545792205254, 0.029796893619137815, 0.96104667186265817, 0.09548056745840004, 0.08864511244348429],
                    [0.18377649152812464, 0.09355287186688058, 0.1333162530259955, 0.1720949411050521, 0.99800385744309532, 0.05851594270914497, 0.260739642321707],
                    [0.16201767024447655, 0.17549039629714336, 0.051250868614221234, 0.940254556612448, 0.006261768243946401, 0.22618242337226518, 0.23854231661549935],
                    [0.16201767024447655, 0.92549039629714336, 0.001250868614221234, 0.040254556612448, 0.006261768243946401, 0.22618242337226518, 0.23854231661549935]
                ],
            ], dtype=torch.float32, requires_grad=False
        ).cuda()

        self.assertEqual(src.shape[0], 2)
        self.assertEqual(src.shape[1], 4)

        self.assertEqual(tgt.shape[0], 2)
        self.assertEqual(tgt.shape[1], 5)

        self.assertEqual(predicted.shape[0], 2)
        self.assertEqual(predicted.shape[1], 4)
        self.assertEqual(predicted.shape[2], 7)

        predicted = F.log_softmax(predicted, dim=2)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        predicted = predicted.transpose(0, 1)
        batch = data_iterator.BatchData(src, tgt, None, None, None, None, None, batch_size=2)
        modelOutputs = structs.ModelOutputs(predictions=predicted, penalty=None)
        loss, stats = lossObj.compute_loss(batch, modelOutputs, cur_step=1)

        self.assertEqual(stats.accuracy(), 100.0)
        a = torch.tensor(
            [
                predicted[0][0][4], predicted[0][1][4],
                predicted[1][0][5], predicted[1][1][4],
                predicted[2][0][6], predicted[2][1][3],
                predicted[3][0][3]

            ], dtype=torch.float32, requires_grad=False
        )
        x = torch.sum(a).item()

        self.assertEqual(stats._loss(), - x / 7.0)
        self.assertEqual(stats.n_words, 7)



if __name__ == '__main__':
    unittest.main()
