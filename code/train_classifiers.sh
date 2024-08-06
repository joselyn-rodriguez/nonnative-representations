#!/bin/bash
#
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --job-name="train_classifer"
#SBATCH --mem=264G
#SBATCH --time=1-00:00:00
#SBATCH --err="%j_train_classifier.txt"

set -exu
      
export PATH=/fs/clip-realspeech/software/conda/envs/wav2vec2/bin:$PATH

set +u

echo $CONDA_DEFAULT_ENV
which python

python -c "import sklearn"

load_dir=$1
language=$2
data_path=$3

python train_classifiers.py $load_dir $language $data_path