# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 10:57
# @Author  : Hui Wang

import torch
from rq_modules.rqvae import RqVae
import torch.nn as nn
from modules import Encoder, LayerNorm, Intermediate_cat, Intermediate_gate, Encoder_Cluster
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

        
        if self.args.token_type == "hybrid" or self.args.token_type == "id":
            self.unique_item_embeddings = nn.Embedding(self.args.item_size, self.args.id_dim_size, padding_idx=0)
        
        
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.reshape_size)

        self.LayerNorm = LayerNorm(args.reshape_size, eps=1e-12)

            
            
        
        self.item_encoder = Encoder(args)
        if self.args.token_type == "hybrid" or self.args.token_type == "semantic":
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


        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.criterion = nn.BCELoss(reduction='none')
        if not(self.args.is_text):
            self.text_item_embeddings = nn.Embedding(self.args.item_size, 768, padding_idx=0)


        self.apply(self.init_weights)
        if self.args.is_text:
            self.text_item_tensors = torch.load(args.data_dir + args.text_embedding_file)
            self.text_item_embeddings = nn.Embedding(self.args.item_size, 768, padding_idx=0).from_pretrained(torch.cat([self.text_item_tensors, torch.rand(1, 768)], 0))

    #
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        text_item_embeddings = self.text_item_embeddings(sequence)
            


        
        if self.args.token_type == "semantic":
            rq_loss, rq_index, rq_item_embeddings = self.rq_model(text_item_embeddings)

            item_embeddings = rq_item_embeddings
        elif self.args.token_type == "hybrid":
            rq_loss, rq_index, rq_item_embeddings = self.rq_model(text_item_embeddings)
            unique_item_embeddings = self.unique_item_embeddings(sequence)
            item_embeddings = torch.cat([rq_item_embeddings, unique_item_embeddings], -1)
        else:
            rq_loss = 0
            unique_item_embeddings = self.unique_item_embeddings(sequence)
            item_embeddings = unique_item_embeddings
            
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings


   
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return rq_loss, sequence_emb

    # Fine tune
    # same as SASRec
    def finetune(self, input_ids, all_ids, user_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64

        max_len = attention_mask.size(-1)
        # all_max_len = all_attention_mask.size(-1)

        attn_shape = (1, max_len, max_len)

        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        rq_loss, sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask,
                                            output_all_encoded_layers=True)
    
        sequence_output = item_encoded_layers[-1]

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