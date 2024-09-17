import torch
samples = torch.Tensor([
                     [0.1, 0.1],    #-> group / class 1
                     [0.2, 0.2],    #-> group / class 2
                     [0.4, 0.4],    #-> group / class 2
                     [0.0, 0.0]     #-> group / class 0
              ])

labels = torch.LongTensor([1, 2, 2, 0])
import pdb
pdb.set_trace()
M = torch.zeros(labels.max()+1, len(samples))
M[labels, torch.arange(4)] = 1
M = torch.nn.functional.normalize(M, p=1, dim=1)
torch.mm(M, samples)
