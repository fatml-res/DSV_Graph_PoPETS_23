import igraph
from utils import load_data
from stealing_link.partial_graph_generation import get_link
import numpy as np


if __name__ == "__main__":
    datapath = "dataset/"
    dataset = "facebook"
    ego_user = "107"

    adj, ft, gender, labels = load_data(datapath, dataset, ego_user, dropout=0)
    node_num = len(ft)
    link, unlink, g_link, g_unlink = get_link(adj, node_num, gender)
    print(np.unique(gender, return_counts=True))
    print(np.unique(g_link, return_counts=True))
    print(np.unique(g_unlink, return_counts=True))
    g = igraph.Graph.Adjacency((adj > 0).tolist())
    g.density