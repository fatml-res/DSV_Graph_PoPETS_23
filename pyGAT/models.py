import torch
import torch.nn as nn
import torch.nn.functional as F
from pyGAT.layers import GraphAttentionLayer, SpGraphAttentionLayer, DP_layer, perturb_adj



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nhead, alpha=0.1, DP=False, lbd=0, Ptb=False, gamma=torch.inf, ptb_time=-1):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nhead)]
        self.DP_layer = DP_layer(lbd=lbd)
        self.ptb_adj = perturb_adj(gamma)
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.lbd = lbd
        self.adj_ptb = None
        self.ptb_time = ptb_time
        self.fix_adj = None
        if lbd > 0:
            self.DP = DP
        else:
            self.DP = False

        if gamma == torch.inf:
            self.Ptb = False
        else:
            self.Ptb = Ptb
            self.ptb_time = ptb_time
            self.fix_adj = None


        self.out_att = GraphAttentionLayer(nhid * nhead, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        if self.Ptb:
            adj_copy = self.ptb_adj(adj)
            if self.ptb_time == 1:
                if self.fix_adj is None:
                    self.fix_adj = adj_copy
            elif self.ptb_time == 2:
                adj_copy = self.ptb_adj(self.adj_ptb)
            else:
                adj_copy = self.fix_adj
        else:
            adj_copy = adj.clone()
        self.adj_ptb = adj_copy.clone()
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj_copy) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj_copy))
        if self.DP:
            x = self.DP_layer(x)
        return F.log_softmax(x, dim=1)

    def get_attentions(self, ft, adj):
        attention = 0 * torch.ones([len(ft), len(ft)])
        for l in self.attentions:
            attention += l.get_attention(ft, adj)
        attention = attention / len(self.attentions)
        return attention



class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

