#!/bin/bash
#
#SBATCH --time=24:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
##SBATCH --mem-per-cpu=1GB
#SBATCH --job-name="plug"
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

echo "hello from S$SLURM JOB ID"
python train_semantic_full.py --embedding_type 'semantic_id_concat' --hidden_size 64 --codebook_size 3 --semantic_dim_size 64 --id_dim_size 32
python train_semantic_full.py --embedding_type 'semantic_id_concat' --hidden_size 64 --codebook_size 3 --semantic_dim_size 64 --id_dim_size 64
python train_semantic_full.py --embedding_type 'semantic_id_concat' --hidden_size 64 --codebook_size 3 --semantic_dim_size 64 --id_dim_size 128
