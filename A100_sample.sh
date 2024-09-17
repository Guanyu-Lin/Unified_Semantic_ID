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
# python train_sample.py --embedding_type 'concat_reshape' --hidden_size 64 --codebook_size 3 --semantic_dim_size 32 --id_dim_size 64 --reshape_size 64 --attribute_size 512
# python train_sample.py --embedding_type 'concat_reshape' --hidden_size 64 --codebook_size 3 --semantic_dim_size 32 --id_dim_size 64 --reshape_size 64 --attribute_size 1024
# python train_sample.py --hidden_size 64 --codebook_size 3 --semantic_dim_size 32 --id_dim_size 64 --reshape_size 64 --codebook_size 512 --is_cos
python train_sample.py --hidden_size 64 --codebook_size 3 --semantic_dim_size 32 --id_dim_size 64 --reshape_size 64 --codebook_size 512 --is_cos --rq_loss_weight 1

# python train_sample.py --embedding_type 'add_reshape' --hidden_size 64 --codebook_size 3 --semantic_dim_size 32 --id_dim_size 64 --reshape_size 64 --attribute_size 1024

