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
python train_semantic_full.py --embedding_type 'id' --hidden_size 64 --codebook_size 3 --semantic_dim_size 0 --id_dim_size 16 --attribute_size 512
# python train_semantic_full.py --embedding_type 'semantic_id_concat' --hidden_size 64 --codebook_size 3 --semantic_dim_size 32 --id_dim_size 16 --attribute_size 512
