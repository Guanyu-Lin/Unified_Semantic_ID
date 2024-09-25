#!/bin/bash
#
#SBATCH --time=24:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
##SBATCH --mem-per-cpu=1GB
#SBATCH --job-name="cluster"
##SBATCH --partition=secondary
#SBATCH --partition=eng-research-gpu
##SBATCH --partition=IllinoisComputes-GPU
##SBATCH --account=jiaxuan-ic
#SBATCH --account=jiaxuan-cs-eng
#SBATCH --gres=gpu:A10:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j.o
#SBATCH --error=logs/%j.e
#SBATCH --mail-user=729309290@qq.com
#SBATCH --mail-type=BEGIN,END,ALL
#End of embedded SBATCH options
#


python train_sample.py --hidden_size 64 --semantic_dim_size 64 --codebook_n_layer 3 --id_dim_size 8 --reshape_size 72 --codebook_size 256 --rq_loss_weight 1 --is_text --distance_type "hybrid" --is_reconstruction --is_cluster --data_name "Beauty"

python train_sample.py --hidden_size 64 --semantic_dim_size 64 --codebook_n_layer 3 --id_dim_size 8 --reshape_size 72 --codebook_size 256 --rq_loss_weight 1 --is_text --distance_type "hybrid" --is_reconstruction --is_cluster --add_cluster --data_name "Beauty"