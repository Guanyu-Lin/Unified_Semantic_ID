import torch
samples = torch.Tensor([
                     [[0.1, 0.1],    #-> group / class 1
                     [0.2, 0.2],    #-> group / class 2
                     [0.4, 0.4],],    #-> group / class 2
                     [[0.1, 0.1],    #-> group / class 1
                     [0.2, 0.2],    #-> group / class 2
                     [0.4, 0.4],]
              ])

labels = torch.LongTensor([[1, 2, 2], [1, 2, 0]])
# import pdb
# pdb.set_trace()
# labels = labels.view(labels.size(0), labels.size(1), 1).expand(-1, labels.size(1), samples.size(1))

# unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
# res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, samples)
ls = []
for b, l in zip(samples, labels):
    import pdb
    pdb.set_trace()
    M = torch.zeros(4, b.shape[0])
    M[l, torch.arange(b.shape[0])] = 1
    M = torch.nn.functional.normalize(M, p=1, dim=1)
    ls.append(torch.mm(M, b))

    
# M = torch.zeros(samples.shape[0], labels.max()+1, samples.shape[1])
# M[labels, :, torch.arange(samples.shape[1])] = 1
# M = torch.nn.functional.normalize(M, p=1, dim=1)
# torch.mm(M, samples)
