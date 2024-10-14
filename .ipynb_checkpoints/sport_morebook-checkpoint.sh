nohup python train_sample.py --hidden_size 64 --semantic_dim_size 64 --codebook_n_layer 3 --id_dim_size 8 --reshape_size 72 --codebook_size 512 --rq_loss_weight 1 --token_type "hybrid" --is_text --distance_type "hybrid" --is_reconstruction --data_name "Sports_and_Outdoors" --do_eval > sport_512.log 2>&1 &

nohup python train_sample.py --hidden_size 64 --semantic_dim_size 64 --codebook_n_layer 3 --id_dim_size 8 --reshape_size 72 --codebook_size 1024 --rq_loss_weight 1 --token_type "hybrid" --is_text --distance_type "hybrid" --is_reconstruction --data_name "Sports_and_Outdoors"  > sport_1024.log 2>&1 &

nohup python train_sample.py --hidden_size 64 --semantic_dim_size 64 --codebook_n_layer 3 --id_dim_size 8 --reshape_size 72 --codebook_size 128 --rq_loss_weight 1 --token_type "hybrid" --is_text --distance_type "hybrid" --is_reconstruction --data_name "Sports_and_Outdoors" > sport_128.log 2>&1 &

nohup python train_sample.py --hidden_size 64 --semantic_dim_size 64 --codebook_n_layer 3 --id_dim_size 8 --reshape_size 72 --codebook_size 256 --rq_loss_weight 1 --token_type "hybrid" --is_text --distance_type "hybrid" --is_reconstruction --data_name "Sports_and_Outdoors" > sport_256.log 2>&1 &
