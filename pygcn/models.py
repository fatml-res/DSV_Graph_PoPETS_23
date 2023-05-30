import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution, DP_layer, perturb_adj
import torch
from utils import normalize_adj
import scipy.sparse as sp


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nhead=0,
                 FairDefense=False, gamma=torch.inf,
                 gpu=True):
        super(GCN, self).__init__()
        self.gpu=gpu

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.ptb_adj = perturb_adj(gamma)
        self.adj_ptb = None
        self.fix_adj = None

        if gamma == torch.inf:
            self.Ptb = False
        else:
            print("Model will go with FairDefense\nGamma={}\n".format(gamma))
            self.FairDefense = FairDefense

    def forward(self, x, adj):
        adj_copy = adj.clone()
        if self.Ptb:
            adj_copy = self.ptb_adj(adj)

        self.adj_ptb = adj_copy
        '''if self.gpu:
            adj_copy = normalize_adj(adj_copy + torch.eye(adj.shape[1]).cuda())
        else:
            adj_copy = normalize_adj(adj_copy + torch.eye(adj.shape[1]))'''

        x = F.relu(self.gc1(x, adj_copy))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj_copy)
        return F.log_softmax(x, dim=1)
