import torch
from vector_quantize_pytorch import ResidualVQ

from torch import nn
from typing import NamedTuple

from .encoder import MLP
from .decoder import Decode_MLP

from .loss import ReconstuctionLoss


class RqVaeOutput(NamedTuple):
    embeddings: torch.Tensor
    sem_ids: torch.Tensor
    commit_loss: torch.Tensor


class RqVae(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dim: int,
        codebook_size: int,
        n_layers: int = 1,
        commitment_weight: float = 0.25,
        loss_weight: float = 10.0,
        n_res_block: int = 2,
        n_res_channel: int = 32,
        distance_type: str = "Cosine",
        is_cluster: bool = False,
        reshape_dim = 64,
        is_reconstruction = False
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.loss_weight = loss_weight
        self.is_cluster = is_cluster
        self.distance_type = distance_type
        self.is_reconstruction = is_reconstruction
        addional_n_layer = 0
        if self.is_cluster:
            self.rvq_cluster = ResidualVQ(
                dim = self.embed_dim,
                codebook_size = 50,
                num_quantizers = 1,
                kmeans_init = True,   # set to True
                use_cosine_sim = True,
                kmeans_iters = 100,     # number of kmeans iterations to calculate the centroids for the codebook on init
                threshold_ema_dead_code=2
            )
            # addional_n_layer += 1
        if self.distance_type == "hybrid":
            self.rvq_Euclidean = ResidualVQ(
                dim = self.embed_dim,
                codebook_size = self.codebook_size,
                num_quantizers = 1,
                kmeans_init = True,   # set to True
                use_cosine_sim = False,
                kmeans_iters = 100,     # number of kmeans iterations to calculate the centroids for the codebook on init
                threshold_ema_dead_code=2
            )
            addional_n_layer += 1
        if self.n_layers - addional_n_layer > 0:
            self.rvq = ResidualVQ(
                dim = self.embed_dim,
                codebook_size = self.codebook_size,
                num_quantizers = self.n_layers - addional_n_layer,
                kmeans_init = True,   # set to True
                use_cosine_sim = True,
                kmeans_iters = 100,     # number of kmeans iterations to calculate the centroids for the codebook on init
                threshold_ema_dead_code=2
            )
        # self.layers = nn.ModuleList(modules=[
        #     Quantize(embed_dim=embed_dim // 2, n_embed=codebook_size)
        #     for _ in range(n_layers)
        # ])

        self.encoder = MLP(input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=embed_dim)
        # self.encoder = Encoder(self.input_dim, self.embed_dim, n_res_block, n_res_channel)
        # self.decoder = Decoder(self.embed_dim, self.input_dim, n_res_block, n_res_channel)

        # self.relu = 
        self.decoder = Decode_MLP(input_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=input_dim)
           
        self.decoder_rec = Decode_MLP(input_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=reshape_dim, shallow=True)



    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def get_semantic_ids(self,
                         x: torch.Tensor,
                         gumbel_t=0.001) -> torch.Tensor:
        # import pdb
        # pdb.set_trace()
        res = self.encode(x)

        # embs, residuals, sem_ids = [], [], []
        rq_embs = None
        rq_semantic_id = []
        rq_loss = []
        if self.is_cluster:
            embs_cluster, sem_ids_cluster, commit_loss_cluster = self.rvq_cluster(res)
            rq_embs = embs_cluster
            rq_loss.append(commit_loss_cluster)
            rq_semantic_id.append(sem_ids_cluster)
            # embs, sem_ids, commit_loss = self.rvq(res - embs_cluster)
            # rq_embs += embs
            # rq_loss.append(commit_loss)
            # rq_semantic_id.append(sem_ids)
            
        if (self.distance_type == "hybrid" and self.n_layers > 1) or (not(self.distance_type == "hybrid")):
            if rq_embs is not None:
                embs, sem_ids, commit_loss = self.rvq(res - rq_embs)
                rq_embs += embs
            else:
                embs, sem_ids, commit_loss = self.rvq(res)
                rq_embs = embs
            rq_embs = embs
            rq_loss.append(commit_loss)
            rq_semantic_id.append(sem_ids)


        if self.distance_type == "hybrid":
            if rq_embs is not None:
                embs_Euclidean, sem_ids_Euclidean, commit_loss_Euclidean = self.rvq_Euclidean(res - rq_embs)
                rq_embs += embs_Euclidean
            else:
                embs_Euclidean, sem_ids_Euclidean, commit_loss_Euclidean = self.rvq_Euclidean(res)
                rq_embs = embs_Euclidean
            rq_loss.append(commit_loss_Euclidean)
            rq_semantic_id.append(sem_ids_Euclidean)

        # for layer in self.layers:
        #     residuals.append(res)
        #     quantized = layer(res, temperature=gumbel_t)
        #     emb, id = quantized.embeddings, quantized.ids
        #     res = res - emb
        #     sem_ids.append(id)
        #     embs.append(emb)
        rq_semantic_id = torch.cat(rq_semantic_id, -1)
        rq_loss = torch.cat(rq_loss, -1)
        return RqVaeOutput(
            embeddings=rq_embs,
            sem_ids=rq_semantic_id,
            commit_loss = rq_loss,
        )

    def forward(self, x: torch.Tensor, stop = False) -> torch.Tensor:
        quantized = self.get_semantic_ids(x)
        embs, sem_ids, rqvae_loss = quantized.embeddings, quantized.sem_ids, quantized.commit_loss
        # import pdb
        # pdb.set_trace()
        if self.is_reconstruction:
            x_hat = self.decode(embs)
            reconstuction_loss = ReconstuctionLoss()(x_hat, x)
            loss = (reconstuction_loss.mean() + self.loss_weight * rqvae_loss.sum())

        else:
            loss = (self.loss_weight * rqvae_loss.sum())
        return loss, sem_ids, embs