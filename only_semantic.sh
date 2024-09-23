python train_sample.py --hidden_size 64 --semantic_dim_size 32 --codebook_n_layer 4 --id_dim_size 64 --reshape_size 64 --codebook_size 512 --rq_loss_weight 1 --is_text --distance_type "hybrid" --is_cluster --only_semantic
python train_sample.py --hidden_size 64 --semantic_dim_size 32 --codebook_n_layer 8 --id_dim_size 64 --reshape_size 64 --codebook_size 512 --rq_loss_weight 1 --is_text --distance_type "hybrid" --is_cluster --only_semantic
python train_sample.py --hidden_size 64 --semantic_dim_size 32 --codebook_n_layer 6 --id_dim_size 64 --reshape_size 64 --codebook_size 1024 --rq_loss_weight 1 --is_text --distance_type "hybrid" --is_cluster --only_semantic
python train_sample.py --hidden_size 64 --semantic_dim_size 32 --codebook_n_layer 8 --id_dim_size 64 --reshape_size 64 --codebook_size 1024 --rq_loss_weight 1 --is_text --distance_type "hybrid" --is_cluster --only_semantic

python train_sample.py --hidden_size 64 --semantic_dim_size 64 --codebook_n_layer 1 --id_dim_size 64 --reshape_size 64 --codebook_size 13000 --rq_loss_weight 1 --is_text --distance_type "hybrid" --only_semantic



python train_sample.py --hidden_size 64 --semantic_dim_size 32 --codebook_n_layer 4 --id_dim_size 64 --reshape_size 64 --codebook_size 512 --rq_loss_weight 1 --is_text --distance_type "Cosine" --is_cluster --only_semantic
python train_sample.py --hidden_size 64 --semantic_dim_size 32 --codebook_n_layer 8 --id_dim_size 64 --reshape_size 64 --codebook_size 512 --rq_loss_weight 1 --is_text --distance_type "Cosine" --is_cluster --only_semantic
python train_sample.py --hidden_size 64 --semantic_dim_size 32 --codebook_n_layer 6 --id_dim_size 64 --reshape_size 64 --codebook_size 1024 --rq_loss_weight 1 --is_text --distance_type "Cosine" --is_cluster --only_semantic
python train_sample.py --hidden_size 64 --semantic_dim_size 32 --codebook_n_layer 8 --id_dim_size 64 --reshape_size 64 --codebook_size 1024 --rq_loss_weight 1 --is_text --distance_type "Cosine" --is_cluster --only_semantic
