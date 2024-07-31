#!/bin/bash
set +e
# Installation and download IPA dictionaries

#SBATCH -c 12
#SBATCH --mem=12GB
#SBATCH -G 1
#SBATCH -p gpu-preempt
#SBATCH --time 01:00:00
#SBATCH -o scripts/out/generate_images_%j.out
#SBATCH --mail-type END

module load miniconda/22.11.1-1
conda activate sinr_icml

python /home/oyilmazel_umass_edu/inaturalist-sinr/viz_map.py  --name new --model_path /home/oyilmazel_umass_edu/inaturalist-sinr/fine-tuned/demo/fine_tune_xsmall_lr.pt --taxa_id 5165
python /home/oyilmazel_umass_edu/inaturalist-sinr/viz_map.py  --name new --model_path /home/oyilmazel_umass_edu/inaturalist-sinr/fine-tuned/demo/fine_tune_xsmall_lr.pt --taxa_id 47518
python /home/oyilmazel_umass_edu/inaturalist-sinr/viz_map.py  --name new --model_path /home/oyilmazel_umass_edu/inaturalist-sinr/fine-tuned/demo/fine_tune_xsmall_lr.pt --taxa_id 741901
python /home/oyilmazel_umass_edu/inaturalist-sinr/viz_map.py  --name new --model_path /home/oyilmazel_umass_edu/inaturalist-sinr/fine-tuned/demo/fine_tune_xsmall_lr.pt --taxa_id 130714