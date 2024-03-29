#guild run train -y source_max_vocab_size=36000 target_max_vocab_size=25000 start_decay_at=10000 optimizer='adam' lrate=0.001


#export http_proxy=http://proxy.iiit.ac.in; export https_proxy=http://proxy.iiit.ac.in:8080; cp ./shuffle_file.sh /tmp/; guild run train -y -l "[en-hi] baseline with patience 10" model_type=seq2seq_baseline expt_name="baseline"
#export http_proxy=http://proxy.iiit.ac.in; export https_proxy=http://proxy.iiit.ac.in:8080; cp ./shuffle_file.sh /tmp/; guild run train -y -l "[en-hi] wordattn with patience 10" model_type=seq2seq_baseline_word_attn expt_name="baseline + dec dep word_attn"
export http_proxy=http://proxy.iiit.ac.in; export https_proxy=http://proxy.iiit.ac.in:8080;
#cp ./shuffle_file.sh /tmp/;
#guild run train -y -l "[en-hi] baseline seed=346 patience=1" model_type=seq2seq_baseline expt_name="seq2seq_baseline" patience_steps=1

#guild run train -y -l "[en-hi] baseline seed=346 patience=10" model_type=seq2seq_baseline expt_name="seq2seq_baseline" patience_steps=10

#guild run train -y -l "[en-hi] baseline_word_attn seed=346 patience=10" model_type=seq2seq_baseline_word_attn expt_name="seq2seq_baseline_word_attn" patience_steps=10

#guild run train -y -l "[de-en] seq2seq_multivec_word_attn seed=346 patience=5" model_type=seq2seq_multivec_word_attn expt_name="seq2seq_multivec_word_attn" patience_steps=5 steps=40000
#guild run train -y \\n print-cmd -l "[de-en] seq2seq_multivec_word_attn seed=346 patience=5" model_type=seq2seq_multivec_word_attn expt_name="seq2seq_multivec_word_attn" patience_steps=5 steps=30000

#guild runs rm -p -T -y


#python nmt.py --adagram_embeddings_dir /home/chaitanya/Research/ShataAnuvadak/_pytorch/all_indian_languages/data/monolingual/german/ --adam_beta1 0.9 --adam_beta2 0.999 --batch_size 64 --bidirectional 1 --decay_lrate_steps 1000 --decoder_attention_type general --decoder_dropout 0.4 --decoder_embedding_dim 500 --decoder_hidden_dim 500 --decoder_num_layers 2 --encoder_dropout 0.4 --encoder_embedding_dim 500 --encoder_hidden_dim 500 --encoder_num_layers 2 --expt_name test --ignore_too_many_unknowns 1 --lrate 1.0 --lrate_decay 0.5 --max_grad_norm 5.0 --model_type seq2seq_multivec_word_attn --norm_method sents --optimizer sgd --path_to_logs data/training_logs.txt --patience_steps 5 --save_freq 10000000 --save_to data/model.pt --source_max_len 50 --source_max_vocab_size 60000 --start_decay_at 15000 --steps 30000 --target_max_len 50 --target_max_vocab_size 46000 --test_dataset /home/chaitanya/Datasets/german-english/test.de-en.csv --testset_target /home/chaitanya/Datasets/german-english/test.en --train_dataset /home/chaitanya/Datasets/german-english/train.de-en.csv --valid_dataset /home/chaitanya/Datasets/german-english/valid.de-en.csv --valid_steps 250 --warmup_steps 0

#./scripts/bleu-1.04.pl /home/chaitanya/Datasets/english-hindi/test.hn < ./data/predicted.txt

python nmt.py \
--adagram_embeddings_dir /home/chaitanya/Research/ShataAnuvadak/_pytorch/all_indian_languages/data/monolingual/english/wiki/ \
--adam_beta1 0.9 \
--adam_beta2 0.999 \
--batch_size 80 \
--bidirectional 1 \
--decay_lrate_steps 1000 \
--decoder_attention_type general \
--decoder_dropout 0.4 \
--decoder_embedding_dim 500 \
--decoder_hidden_dim 500 \
--decoder_num_layers 2 \
--encoder_dropout 0.4 \
--encoder_embedding_dim 500 \
--encoder_hidden_dim 500 \
--encoder_num_layers 2 \
--expt_name "dev" \
--ignore_too_many_unknowns 0 \
--lrate 1.0 \
--lrate_decay 0.5 \
--max_grad_norm 5.0 \
--model_type seq2seq_multivec_word_attn \
--norm_method sents \
--optimizer sgd \
--path_to_logs data/training_logs.txt \
--patience_steps 10 \
--save_freq 10000000 \
--save_to data/model.pt \
--source_max_len 50 \
--source_max_vocab_size 26000 \
--start_decay_at 15000 \
--steps 20000 \
--target_max_len 50 \
--target_max_vocab_size 36000 \
--test_dataset /home/chaitanya/Datasets/english-hindi/test.en-hn.csv \
--testset_target /home/chaitanya/Datasets/english-hindi/test.hn \
--train_dataset /home/chaitanya/Datasets/english-hindi/train.en-hn.csv \
--valid_dataset /home/chaitanya/Datasets/english-hindi/valid.en-hn.csv \
--valid_steps 250 \
--warmup_steps 0