#!/bin/bash
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --job-name="get_embeds"
#SBATCH --gres=gpu:2
#SBATCH --mem=264G
#SBATCH --time=2-00:00:00
#SBATCH --err="%j_extract_embeds.txt"


set -exu
      
export PATH=/fs/clip-realspeech/software/conda/bin:$PATH


source deactivate

set +u
source activate wav2vec

echo $CONDA_DEFAULT_ENV
echo $nvidia_smi
which python

python -c "import fairseq"

data_path=$1
out_dir=$2
out_name=$3
cp=$4
file_type=$5

python extract_all_reps.py save_rep $data_path $out_dir $out_name $cp $file_type
 