in_file="../dataset/test.en"
out_file="../dataset/test.en.lbl"

python preprocess_adagram_labels.py $in_file
bash /home/chaitanya/Research/Adagram_data/AdaGram.jl/run.sh a.jl > /tmp/word-context-probs.txt
python postprocess_adagram_labels.py /tmp/word-context-probs.txt $out_file