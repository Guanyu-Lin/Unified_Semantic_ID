# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 10:57
# @Author  : Hui Wang

import torch
from rq_modules.rqvae import RqVae
import torch.nn as nn
from modules import Encoder, Encoder_Cluster, LayerNorm
class Embedding(nn.Module):
    def __init__(self, embeddings):
        super(Embedding, self).__init__()
        self.weight = embeddings
    def forward(self, input_ids):
        return self.weight[input_ids]

class S3RecModel(nn.Module):
    def __init__(self, args):
        super(S3RecModel, self).__init__()
        self.args = args

        
        if not(self.args.only_semantic):
            self.unique_item_embeddings = nn.Embedding(self.args.item_size, self.args.id_dim_size, padding_idx=0)
            self.user_embeddings = nn.Embedding(self.args.user_size, self.args.id_dim_size, padding_idx=0)


        self.position_embeddings = nn.Embedding(args.max_seq_length, args.reshape_size)
        self.position_embeddings_all = nn.Embedding(args.max_seq_length_all, args.reshape_size)

        self.LayerNorm = LayerNorm(args.reshape_size, eps=1e-12)

  
        self.item_encoder = Encoder(args)
        self.item_encoder_cluster = Encoder_Cluster(args)

        self.rq_model = RqVae(
            input_dim=768,
            embed_dim=args.semantic_dim_size,
            hidden_dim=256,
            codebook_size=args.codebook_size,
            n_layers=args.codebook_n_layer,
            loss_weight=args.w,
            distance_type=args.distance_type,
            is_cluster=args.is_cluster,
            reshape_dim = args.reshape_size,
            is_reconstruction = args.is_reconstruction,
            cluster_size = args.cluster_size
        )


        # self.rq_model_user = RqVae(
        #     input_dim=768,
        #     embed_dim=args.semantic_dim_size,
        #     hidden_dim=256,
        #     codebook_size=args.codebook_size,
        #     n_layers=args.codebook_n_layer,
        #     distance_type=args.distance_type,
        #     loss_weight=args.w,
        #     reshape_dim = args.reshape_size
        # )

        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.criterion = nn.BCELoss(reduction='none')
        if not(self.args.is_text):
            self.text_item_embeddings = nn.Embedding(self.args.item_size, 768, padding_idx=0)


        self.apply(self.init_weights)
        if self.args.is_text:
            print(args.text_embedding_file)
            self.text_item_tensors = torch.load(args.data_dir + args.text_embedding_file)
            self.text_item_embeddings = nn.Embedding(self.args.item_size, 768, padding_idx=0).from_pretrained(torch.cat([self.text_item_tensors, torch.rand(1, 768)], 0))

    def add_position_embedding_cluster(self, sequence, all_attention_mask):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)

        text_item_embeddings = self.text_item_embeddings(sequence)


        rq_loss, rq_index, rq_item_embeddings = self.rq_model(text_item_embeddings)

        labels = rq_index[:, :, 0]
        if self.args.only_semantic:
            item_embeddings = rq_item_embeddings
        else:
            unique_item_embeddings = self.unique_item_embeddings(sequence)
            item_embeddings = torch.cat([rq_item_embeddings, unique_item_embeddings], -1)

        position_embeddings = self.position_embeddings_all(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        # # import pdb
        # # pdb.set_trace()

        # label_count = extended_cluster_mask.sum(dim=2) # b, c
        # weight = torch.nn.functional.normalize(extended_cluster_mask.float(), p=1, dim=2) # l1 normalization
        # cluster_emb = torch.mm(weight, sequence_emb) # b, c, d
        cluster_mask_ls = []

        cluster_emb_ls = []

        for sample, label, mask in zip(sequence_emb, labels, all_attention_mask):
            # import pdb
            # pdb.set_trace()
            M = torch.zeros(self.args.cluster_size, sample.shape[0], device=sequence.device)
            M[label, torch.arange(sample.shape[0])] = 1
            M = (M.long() & mask.view(-1, 1, self.args.max_seq_length_all)).squeeze(0) # c, s
            cluster_mask_ls.append(M)
    
            # M = torch.nn.functional.normalize(M.float(), p=1, dim=1)
            cluster_emb_ls.append(torch.mm(M.float(), sample))
        # import pdb
        # pdb.set_trace()
        cluster_emb = torch.stack(cluster_emb_ls, 0)
        cluster_mask = torch.stack(cluster_mask_ls, 0)
        del cluster_emb_ls[:]
        del cluster_mask_ls[:]
   

        return rq_loss, cluster_emb, cluster_mask, text_item_embeddings


    #
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        text_item_embeddings = self.text_item_embeddings(sequence)
            


        rq_loss, rq_index, rq_item_embeddings = self.rq_model(text_item_embeddings)
        
        if self.args.only_semantic:
            item_embeddings = rq_item_embeddings
        else:
            unique_item_embeddings = self.unique_item_embeddings(sequence)
            item_embeddings = torch.cat([rq_item_embeddings, unique_item_embeddings], -1)
            
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings


   
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return rq_loss, sequence_emb

    # Fine tune
    # same as SASRec
    def finetune(self, input_ids, all_ids, user_ids):

        attention_mask = (input_ids > 0).long()
        all_attention_mask = (all_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64

        max_len = attention_mask.size(-1)
        # all_max_len = all_attention_mask.size(-1)

        attn_shape = (1, max_len, max_len)

        # cluster2his_mask = torch.ones(self.args.codebook_size).long()
        # extended_cluster2his_mask = cluster2his_mask.view(-1, self.args.codebook_size, 1) & all_attention_mask.view(-1, 1, all_max_len) # b, c, s_all

        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        rq_loss, sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)
        
        # sequence_text_emb = self.look_embedding(input_ids)
        # sequence_rq_emb = self.convert_rq_embedding(sequence_text_emb)
        encoded_sequence = item_encoded_layers[-1]
        # if self.args.is_cluster:
        if self.args.add_cluster:
            rq_loss_cluster, cluster_emb, cluster_mask, text_emb = self.add_position_embedding_cluster(all_ids, all_attention_mask)
            # rq_loss += rq_loss_cluster
            cluster_mask = (torch.sum(cluster_mask, 2) > 0).long()
            extended_target2cluster_mask = attention_mask.view(-1, max_len, 1) & cluster_mask.view(-1, 1, self.args.cluster_size) # b, s, c
            # import pdb
            # pdb.set_trace()
            item_encoded_layers_cluster = self.item_encoder_cluster(sequence_emb, cluster_emb,
                                                extended_target2cluster_mask.unsqueeze(1),
                                                output_all_encoded_layers=True)
            encoded_sequence += item_encoded_layers_cluster[-1]

    
        # rq_loss_user, _, rq_user_embeddings = self.rq_model_user(text_emb)
        # if self.args.only_semantic:
        #     user_emb = torch.tile(torch.mean(rq_user_embeddings, 1, True), (1, max_len, 1))
        # else:
        #     unique_user_emb = self.user_embeddings(user_ids)
        #     user_emb = torch.tile(torch.unsqueeze(unique_user_emb, 1), (1, max_len, 1)) + torch.tile(torch.mean(rq_user_embeddings, 1, True), (1, max_len, 1))
        
        sequence_output = encoded_sequence
        return rq_loss, sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()