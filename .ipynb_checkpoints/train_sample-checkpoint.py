# -*- coding: utf-8 -*-
# @Time    : 2020/11/5 21:11
# @Author  : Hui Wang

import os
import numpy as np
import random
import torch
import argparse
import seaborn as sns
from sklearn.cluster import KMeans

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
from sklearn.manifold import TSNE

from datasets import SASRecDataset, SemanticDataset
from trainers import FinetuneTrainer, Initializer
from models import S3RecModel
from utils import EarlyStopping, get_user_seqs_and_sample, check_path, set_seed

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='Beauty', type=str)
    parser.add_argument('--token_type', type=str, default="hybrid", help="token type")
    parser.add_argument('--text_embedding_file', default='text_emb.pt', type=str)


    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=10, type=int, help="pretrain epochs 10, 20, 30...")

    # model args
    parser.add_argument("--model_name", default='Finetune_sample', type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--id_dim_size", type=int, default=64, help="dim size of id embedding")
    parser.add_argument("--reshape_size", type=int, default=64, help="reshape size of id embedding")
    parser.add_argument("--semantic_dim_size", type=int, default=32, help="dim size of semantic embedding")
    parser.add_argument("--codebook_n_layer", type=int, default=3, help="codebook size of transformer model")
    parser.add_argument("--cluster_size", type=int, default=50, help="cluster size of transformer model")
    parser.add_argument("--codebook_size", type=int, default=256, help="number of each codebook")
    parser.add_argument("--hard_gate", type=float, default=1.0)

    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_cluster_layers", type=int, default=1, help="number of cluster layers")

    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)

    parser.add_argument('--w', type=float, default=1)
    parser.add_argument('--rq_loss_weight', type=float, default=1)
    parser.add_argument('--distance_type', default="hybrid", type=str)
    parser.add_argument('--is_cluster', action='store_true')
    parser.add_argument('--add_cluster', default='add', type=str)

    parser.add_argument('--is_text', action='store_true')
    parser.add_argument('--is_reconstruction', action='store_true')

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)
    args.text_embedding_file =  args.data_name + 'text_emb.pt'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + args.data_name + '.txt'
    args.sample_file = args.data_dir + args.data_name + '_sample.txt'



    user_seq, max_item, max_user, max_seq_length_all, sample_seq = \
        get_user_seqs_and_sample(args.data_file, args.sample_file)



    args.item_size = max_item + 2
    args.user_size = max_user + 1
    args.mask_id = max_item + 1
    args.max_seq_length_all = max_seq_length_all - 3

    # save model args
    cluster_str = "no_cluster"
    if args.is_cluster:
        cluster_str = "cluster"
        
    reconstruction_str = "no_reconstruction"
    if args.is_reconstruction:
        reconstruction_str = "reconstruction"
        

    args_str = f'{args.num_hidden_layers}-{args.hidden_size}-{args.data_name}-{args.id_dim_size}-{args.codebook_n_layer}-{args.codebook_size}-{args.semantic_dim_size}-{args.distance_type}-{reconstruction_str}-{cluster_str}-{args.add_cluster}-{args.token_type}-{args.num_cluster_layers}-{args.max_seq_length}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # args.item2attribute = item2attribute
    # args.attribute2item = attribute2item


    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    train_dataset = SASRecDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = SASRecDataset(args, user_seq, test_neg_items=sample_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = SASRecDataset(args, user_seq, test_neg_items=sample_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)




    model = S3RecModel(args=args)
    # initializer = Initializer(model, semantic_dataloader, train_dataloader, eval_dataloader,
    #                           test_dataloader, args)
    # initializer.initialize()
    trainer = FinetuneTrainer(model, train_dataloader, eval_dataloader,
                              test_dataloader, args)

    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        emb_layer = []
        
        emb_layer.append(model.rq_model.rvq.layers[0]._codebook.embed[0])
        
        second_layer = model.rq_model.rvq.layers[1]._codebook.embed
        emb_layer.append(second_layer[0])
        final_layer = model.rq_model.rvq_Euclidean.layers[0]._codebook.embed
        emb_layer.append(final_layer[0])
        unique = model.unique_item_embeddings.weight
        emb_layer.append(unique)
        
        p_size = [80, 80, 80, 10]
        name = ["first_layer", "second_layer", "third_layer", "unique"]
        for i in range(len(emb_layer)):
            plt.rcParams['font.sans-serif'] = ['Times New Roman']
            font = 'Times New Roman'
            plt.rcParams['axes.unicode_minus'] = False
            matplotlib.rcParams['pdf.fonttype'] = 42
            matplotlib.rcParams['ps.fonttype'] = 42
            plt.rcParams['savefig.dpi'] = 300
            plt.rcParams['figure.dpi'] = 300
            font3 = {'family': font,
                     'weight': 'normal',
                     'size': 36,
                     }
            font1 = {'family': font,
                'weight': 'normal',
                'size': 24,
                }   
            k =args.codebook_size
            if i < 3:
                dff_pred = range(k)
            else:
                kmeans = KMeans(n_clusters=k, random_state=42)
                dff_pred = kmeans.fit_predict(emb_layer[i].cpu().detach().numpy())

            sns.set(rc={'figure.figsize':(8,8)})
            sns.set_style("white", {"axes.edgecolor": ".0", "axes.facecolor":"none"})
            figure, ax = plt.subplots()

            sns.despine(top=True, right=True, left=True, bottom=True)
            palette = sns.hls_palette(k, l=.4, s=.9)

            tsne = TSNE(verbose=1, perplexity=21, random_state=42, early_exaggeration=3)
            dff_tsne = tsne.fit_transform(emb_layer[i].cpu().detach().numpy())

            # import pdb
            # pdb.set_trace()
            sns.scatterplot(x = dff_tsne[:,0], y = dff_tsne[:,1], hue=dff_pred, palette=palette, s=p_size[i])
            plt.xticks([])
            plt.yticks([])
            plt.legend()
            plt.gca().get_legend().remove()
    
    
            plt.tight_layout()
            
            
            plt.savefig('emb/' + name[i] + str(args.codebook_size) + args.data_name + '.png')
        scores, result_info = trainer.test(0, full_sort=False)

    else:

        early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            scores, _ = trainer.valid(epoch, full_sort=False)
            # evaluate on MRR
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('---------------Sample 99 results-------------------')
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=False)
        # scores, result_info = trainer.test(0, full_sort=False)

    print(args_str)
    print(result_info)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')
main()