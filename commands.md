####Ada commands:
- `squeue --user chaitanyasai.alaparthi`
- `sbatch train.sh` to submit batch job
- `sinteractive -c 10 -p long -A research -g 1` to run interactively
- Add the following environment modules/job configs in train.sh while submitting with sbatch
```
#!/bin/bash

#SBATCH -A research
#SBATCH -n 32
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
module load cuda/10.0
module load cudnn/7.6-cuda-10.0
module load python/3.7.4

source ../../venv/bin/activate
 ```
- Visit hpc.iiit.ac.in/wiki/index.php/Ada_User_Guide for more information

####Unittests: 
- `py.test -v --no-print-logs`
- `py.test -v --no-print-logs -x` to stop on first failure
- `pytest -v --show-capture=no --cov-report html --cov=.` to run with coverage

####Few codes to look for:
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
