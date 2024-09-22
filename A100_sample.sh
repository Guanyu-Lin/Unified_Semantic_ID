#!/bin/bash
#
#SBATCH --time=24:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=1GB
#SBATCH --job-name="id_semantic"
##SBATCH --partition=secondary
##SBATCH --partition=eng-research-gpu
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jiaxuan-ic
##SBATCH --account=jiaxuan-cs-eng
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j.o
#SBATCH --error=logs/%j.e
#SBATCH --mail-user=729309290@qq.com
#SBATCH --mail-type=BEGIN,END,ALL
#End of embedded SBATCH options
#

echo "hello from S$SLURM JOB ID"
python train_sample.py --hidden_size 64 --semantic_dim_size 32 --codebook_n_layer 8 --id_dim_size 64 --reshape_size 64 --codebook_size 512 --rq_loss_weight 1 --is_text --distance_type "hybrid" --is_cluster --only_semantic
python train_sample.py --hidden_size 64 --semantic_dim_size 32 --codebook_n_layer 8 --id_dim_size 64 --reshape_size 64 --codebook_size 1024 --rq_loss_weight 1 --is_text --distance_type "hybrid" --is_cluster --only_semantic
python train_sample.py --hidden_size 64 --semantic_dim_size 32 --codebook_n_layer 6 --id_dim_size 64 --reshape_size 64 --codebook_size 1024 --rq_loss_weight 1 --is_text --distance_type "hybrid" --is_cluster --only_semantic
