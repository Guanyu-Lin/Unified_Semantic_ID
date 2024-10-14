import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.amazon import AmazonData
from sentence_transformers import SentenceTransformer
import pandas as pd
from torch.utils.data import DataLoader

def save_emb(loader, device, batch_size, ST5, data_name):
    emb_ls = []
    id_ls = []
    for i, text in loader:
        i = i.to(device)
        id_ls.append(i)
        input_emb = ST5.encode(text, convert_to_tensor=True, batch_size=batch_size).clone().detach().unsqueeze(1).to(device)


        emb_ls.append(input_emb)
    emb_ls = torch.cat(emb_ls, dim=0).cpu()
    id_ls = torch.cat(id_ls, dim=0).cpu()
    id2emb = torch.zeros(emb_ls.shape[0] + 1,emb_ls.shape[-1])
    for id, emb in zip(id_ls, emb_ls):

        id2emb[id] = emb[0]

    print(id2emb[0])
    print(id2emb)
    torch.save(id2emb, "data/" + data_name + "text_emb.pt")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--data_name', type=str, default="Beauty")

    args = parser.parse_args()

    device = 'cuda'
    dataset = AmazonData(data_name=args.data_name)

    ST5 = SentenceTransformer('../sentenceT5/sentence-t5-large', device=device)


    text_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    save_emb(text_loader, device, args.batch_size, ST5, args.data_name)

