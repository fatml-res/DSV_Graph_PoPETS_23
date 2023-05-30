import torch
import torch.nn as nn
import torch.nn.functional as F
from pyGAT.layers import GraphAttentionLayer, perturb_adj


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nhead, alpha=0.1, FairDefense=False, gamma=torch.inf):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nhead)]
        self.ptb_adj = perturb_adj(gamma)
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.adj_ptb = None

        if gamma == torch.inf:
            self.FairDefense = False
        else:
            self.FairDefense = FairDefense

        self.out_att = GraphAttentionLayer(nhid * nhead, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        if self.FairDefense:
            adj_copy = self.ptb_adj(adj)
        else:
            adj_copy = adj.clone()
        self.adj_ptb = adj_copy.clone()
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj_copy) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj_copy))
        return F.log_softmax(x, dim=1)

    def get_attentions(self, ft, adj):
        attention = 0 * torch.ones([len(ft), len(ft)])
        for l in self.attentions:
            attention += l.get_attention(ft, adj)
        attention = attention / len(self.attentions)
        return attention

