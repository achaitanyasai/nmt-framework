####Unittests: 
- `py.test -v --no-print-logs`
- `pytest -v --show-capture=no --cov-report html --cov=.` to run with coverage

Few codes to look for:
- `~/Research/ShataAnuvadak/_pytorch/all_indian_languages/multivec-adagram/nmt-framework/nmt.py`
- `/home/chaitanya/Research/ShataAnuvadak/_pytorch/all_indian_languages/data`

####Guild commands: 
- `guild run train -y -l <label>`
- `guild runs delete` - To delete the runs
- `guild runs purge` - To delete the deleted runs completely.
- `guild runs rm --started 'last 2 hours'` - To delete the runs made in last 2 hours.
- `guild runs rm --started 'today' -p` - To _permenantly_ delete the runs which were triggered on the same day.

####Adagram commands:
- To generate dictionary: `bash ../../AdaGram.jl/utils/dictionary.sh ./monolingual.en ./monolingual.en.dict`
- To train the model: bash `../../AdaGram.jl/train.sh ./monolingual.en ./monolingual.en.dict ./monolingual.en.model --workers 16 --dim 500 --prototypes 3 --epochs 2`
- To get the embeddings: `../../AdaGram.jl/run.sh ./scripts/a.jl > embeddings.txt`. **Before running**: 
  - Make sure you change the model path in `a.jl`
  - Make sure you modify `print(vm.In[i, <vector space in 1..k>, v])` in `write_word2vec` in `~/.julia/v0.4/AdaGram/src/util.jl` to the desired vector space.
  - Note that the word embeddings will be printed to stdout.
