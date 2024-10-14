
# python train_sample.py --hidden_size 64 --semantic_dim_size 64 --codebook_n_layer 3 --id_dim_size 64 --reshape_size 64 --codebook_size 256 --rq_loss_weight 1 --token_type "id" --is_text --distance_type "hybrid" --is_reconstruction --data_name "Sports_and_Outdoors"

# python train_sample.py --hidden_size 64 --semantic_dim_size 64 --codebook_n_layer 3 --id_dim_size 64 --reshape_size 64 --codebook_size 256 --rq_loss_weight 1 --token_type "semantic" --is_text --distance_type "hybrid" --is_reconstruction --data_name "Sports_and_Outdoors"

nohup python train_sample.py --hidden_size 64 --semantic_dim_size 64 --codebook_n_layer 3 --id_dim_size 8 --reshape_size 72 --codebook_size 256 --rq_loss_weight 1 --token_type "hybrid" --is_text --distance_type "cosine" --is_reconstruction --data_name "Sports_and_Outdoors" > sport_cosine.log 2>&1 &

python train_sample.py --hidden_size 64 --semantic_dim_size 64 --codebook_n_layer 3 --id_dim_size 4 --reshape_size 68 --codebook_size 256 --rq_loss_weight 1 --token_type "hybrid" --is_text --distance_type "hybrid" --is_reconstruction --data_name "Sports_and_Outdoors" --do_eval

python train_sample.py --hidden_size 64 --semantic_dim_size 64 --codebook_n_layer 3 --id_dim_size 16 --reshape_size 80 --codebook_size 256 --rq_loss_weight 1 --token_type "hybrid" --is_text --distance_type "hybrid" --is_reconstruction --data_name "Sports_and_Outdoors"

