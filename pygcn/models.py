import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution, DP_layer, perturb_adj
import torch
from utils import normalize_adj
import scipy.sparse as sp


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nhead=0,
                 DP=False, lbd=0,
                 Ptb=False, gamma=torch.inf,
                 gpu=True,
                 ptb_time=-1):
        super(GCN, self).__init__()
        self.gpu=gpu

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.DP_layer = DP_layer(lbd)
        self.ptb_adj = perturb_adj(gamma)
        self.adj_ptb = None
        self.ptb_time = ptb_time
        self.fix_adj = None
        if lbd <= 0:
            self.DP = False
        else:
            print("Model will go with baseline\nlambda={}\n".format(lbd))
            self.DP = DP

        if gamma == torch.inf:
            self.Ptb = False
        else:
            print("Model will go with M-In\nGamma={}\n".format(gamma))
            self.Ptb = Ptb
            if self.ptb_time:
                print("Testing! The Adj will only be changed for once")

    def forward(self, x, adj):
        adj_copy = adj.clone()
        if self.Ptb:
            adj_copy = self.ptb_adj(adj)
            if self.ptb_time == 1:
                if self.fix_adj is None:
                    self.fix_adj = adj_copy
            elif self.ptb_time == 2:
                adj_copy = self.ptb_adj(self.adj_ptb) if self.adj_ptb is not None else adj_copy
            else:
                adj_copy = self.fix_adj

        self.adj_ptb = adj_copy
        '''if self.gpu:
            adj_copy = normalize_adj(adj_copy + torch.eye(adj.shape[1]).cuda())
        else:
            adj_copy = normalize_adj(adj_copy + torch.eye(adj.shape[1]))'''

        x = F.relu(self.gc1(x, adj_copy))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj_copy)
        if self.DP:
            x = self.DP_layer(x)
        return F.log_softmax(x, dim=1)
