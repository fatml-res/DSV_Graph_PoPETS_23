import pandas as pd
import numpy as np
import torch
import pickle as pkl


if __name__ == "__main__":
    i = "20"
    node_file = "dataset/tagged/sample_{}.csv".format(i)
    edge_file = "dataset/tagged/edge_{}.csv".format(i)
    array = pd.read_csv(node_file, header=None, sep='\t').to_numpy()
    ids = array[:, 0]
    gender = np.array([1 if x == 'F' else 2 for x in array[:, 1]])
    ft = np.hstack([gender.reshape(-1, 1), array[:, 2:-1]])
    ft = ft.astype(float)
    labels = array[:, -1]
    labels = labels.astype(int)

    adj = np.zeros([len(ids), len(ids)])
    #edge_reader = open(edge_file, 'r')
    #edges = edge_reader.readlines()

    edges = pd.read_csv(edge_file, header=None, sep='\t').to_numpy()
    for edge in edges:
        #id1, id2 = edge.split("\n")[0].split(" ")
        id1, id2 = edge[0].split(" ")
        id1 = int(id1)
        id2 = int(id2)
        adj[np.where(ids == id1), np.where(ids == id2)] = 1
        adj[np.where(ids == id2), np.where(ids == id1)] = 1
    adj = torch.LongTensor(adj)
    ft = torch.FloatTensor(ft)
    labels = torch.LongTensor(labels)
    saving_path = "dataset/tagged/"
    dataset = "tagged_{}".format(i)

    with open('{}/ind.{}.ft'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(ft, f)

    with open('{}/ind.{}.labels'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(labels, f)

    with open('{}/ind.{}.gender'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(gender, f)

    with open('{}/ind.{}.adj'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(adj, f)

    pass

