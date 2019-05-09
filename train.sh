#!/bin/bash

check_and_exit () {
    if [ $? -ne 0 ]; then
        exit 0
    fi
}

unittests="no"
clean="yes"
train="yes"
test="yes"
bleu="yes"

dataDir="/home/chaitanya/Research/datasets/english-turkish-preprocessed-data"
src="en"
trg="tr"

srcFile="$dataDir/train.$src"
trgFile="$dataDir/train.bpe.$trg"

srcValFile="$dataDir/dev/newsdev2016.tc.$src"
trgValFile="$dataDir/dev/newsdev2016.tc.bpe.$trg"

srcTestFile="$dataDir/dev/newstest2016.tc.$src"
trgTestFile="$dataDir/dev/newstest2016.tc.$trg"

if [ "$unittests" = "yes" ]; then
echo "Running unittests"
python2.7 -W ignore -m unittest discover -v -f
fi

check_and_exit

if [ "$clean" = "yes" ]; then
echo "Cleaning ./data/"
rm -rf ./data/*
fi 
check_and_exit

# Use CUDA_LAUNCH_BLOCKING=1 for debugging
if [ "$train" = "yes" ]; then
python nmt.py \
   --datasets "$srcFile $trgFile" \
   --datasets_valid "$srcValFile $trgValFile" \
   --src_maxlen 50 \
   --tgt_maxlen 100 \
   --batch_size 40 \
   --bidirectional \
   --enc_depth 2 \
   --dec_depth 2 \
   --dropout 0.4 \
   --lrate 1.0 \
   --optimizer sgd \
   --dispFreq 1000 \
   --sampleFreq 0 \
   --max_epochs 18 \
   --sampleFreq 100000 \
   --evaluateFreq 1 \
   --src_max_word_len 30 \
   --tgt_max_word_len 30 \
   --src_max_vocab_size 50000 \
   --tgt_max_vocab_size 29500 \
   --start_decay_at 10 \
   --saveTo ./data/model.pt \
   --max_number_of_sentences_allowed 300000000
fi
check_and_exit

if [ "$test" = "yes" ]; then
echo "Predicting"
python2.7 translate.py \
    --input $srcTestFile \
    --output ./data/predicted.bpe.txt \
    --model ./data/model.pt \
    --beam_size 10 \
    --replace_unk \
    --batch_size 1
fi
check_and_exit

if [ "$bleu" = "yes" ]; then
sed -r 's/(@@ )|(@@ ?$)//g' < ./data/predicted.bpe.txt > ./data/predicted.txt
./scripts/bleu-1.04.pl $trgTestFile < ./data/predicted.txt
fi
