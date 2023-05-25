import numpy as np
import pickle as pkl
import torch
import json


if __name__ == "__main__":
    datasets = ["facebook", "facebook", "pokec", "pokec", "tagged_40", "tagged_40"]
    model_types = ["GAT", "gcn", "GAT", "gcn", "GAT", "gcn"]
    deltas = [0.1, 0.05, 0.1, 0.1, 0, 0]

    for i in range(6):
        if deltas[i] > 0:
            adj_loc = "{}/CNR/Group/Reduce/Delta={}".format(model_types[i], deltas[i])
            adj_file = "{}/ind.{}.adj".format(adj_loc, datasets[i])
        else:
            adj_loc = model_types[i]
            adj_file = "{}/ind.{}.adj".format(adj_loc, datasets[i])

        adj = pkl.loads(open(adj_file, "rb").read()).detach().numpy()
        '''tr_edge_file = "{}/partial/t=0/{}_train_ratio_0.2_train_fair.json".format(adj_loc, datasets[i])
        for row in open(tr_edge_file).readlines():
            row = json.loads(row)
            if row['label']:
                ids_pair = row['id_pair']
                adj[ids_pair] = 1
                adj[ids_pair[::-1]] = 1'''
        gender = pkl.loads(open(adj_file.replace('adj', 'gender'), "rb").read())
        if torch.is_tensor(gender):
            gender = gender.detach().numpy()

        # O = 1/2m sum((Aij- di*dj/2m)cicj)
        didj = adj.sum(axis=1).reshape(-1, 1).dot(adj.sum(axis=1).reshape(1, -1))
        m = adj.sum() / 2
        cicj = 2 * (gender.reshape(-1, 1) == gender.reshape(1, -1)) - 1

        homophily = ((adj - didj / 2 / m) * cicj / 2 / m).sum()

        print("Homoghily of {} dataset under {} model is {:.2}".format(datasets[i], model_types[i], homophily))

    for i in range(6):
        for target_cicj in range(3):
            if deltas[i] > 0:
                adj_loc = "{}/CNR/Group/Reduce/Delta={}".format(model_types[i], deltas[i])
                adj_file = "{}/ind.{}.adj".format(adj_loc, datasets[i])
            else:
                adj_loc = model_types[i]
                adj_file = "{}/ind.{}.adj".format(adj_loc, datasets[i])

            adj = pkl.loads(open(adj_file, "rb").read()).detach().numpy() * 0
            tr_edge_file = "{}/partial/t=0/{}_train_ratio_0.2_train_fair.json".format(adj_loc, datasets[i])
            for row in open(tr_edge_file).readlines():
                row = json.loads(row)
                if row['label']:
                    ids_pair = row['id_pair']
                    adj[ids_pair] = 1
                    adj[ids_pair[::-1]] = 1
            gender = pkl.loads(open(adj_file.replace('adj', 'gender'), "rb").read())
            if torch.is_tensor(gender):
                gender = gender.detach().numpy()

            # O = 1/2m sum((Aij- di*dj/2m)cicj)
            didj = adj.sum(axis=1).reshape(-1, 1).dot(adj.sum(axis=1).reshape(1, -1))
            m = adj.sum() / 2
            if target_cicj == 0: # different gender
                cicj = 2 - gender.reshape(-1, 1) != gender.reshape(1, -1)
            else: # same gender
                cicj = 2 - (gender.reshape(-1, 1) == gender.reshape(1, -1)) * (gender == target_cicj)


            homophily = ((adj - didj / 2 / m) * cicj / 2 / m).sum()

            print("Homoghily of {} dataset, Group {} under {} model is {:.2}".format(datasets[i],target_cicj, model_types[i], homophily))