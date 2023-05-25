import igraph
import numpy as np
import pandas as pd
import pickle as pkl
import os
from tqdm import tqdm
import argparse


def get_5_weights(g, node1, node2):

    res = []
    edge_bts = g.edge_betweenness()
    for i in tqdm(range(len(node1))):
        n1 = node1[i]
        n2 = node2[i]
        try:
            bc_edge = edge_bts[g.get_eid(n1, n2)]
            mem = 1
        except:
            g.add_edge(n1, n2)
            eid = g.get_eid(n1, n2)
            bc_edge = g.edge_betweenness()[eid]
            g.delete_edges((n1, n2))
            mem = 0
        res.append([n1, n2, mem,
                    bc_edge])
    return np.array(res)


def get_weights_with_batch(g, node1, node2, batch_size=20):
    res = []
    edge_bts = g.edge_betweenness()
    idx_edges = np.arange(len(node1))
    np.random.shuffle(idx_edges)

    num_of_batch = np.ceil(len(idx_edges)/batch_size)
    for batch in tqdm(range(int(num_of_batch))):
        mem_list_batch = []
        # check edge existence, if not then add
        batch_max_idx = min((batch + 1) * batch_size, len(idx_edges))
        for i in range(batch * batch_size, batch_max_idx):
            n1 = node1[idx_edges[i]]
            n2 = node2[idx_edges[i]]
            try:
                g.get_eid(n1, n2)
                mem_list_batch.append(1)
            except:
                g.add_edge(n1, n2)
                mem_list_batch.append(0)

        # get edge betweenness
        edge_bts = g.edge_betweenness()

        # append res and delete edge
        for i in range(batch * batch_size, batch_max_idx):
            n1 = node1[idx_edges[i]]
            n2 = node2[idx_edges[i]]
            try:
                eid = g.get_eid(n1, n2)
                bc_edge = edge_bts[eid]
                res.append([n1, n2, mem_list_batch[i - batch * batch_size], bc_edge])
                if mem_list_batch[i - batch * batch_size] == 0:
                    g.delete_edges((n1, n2))
            except:
                print("Did not find edge {}".format((n1, n2)))

    return np.array(res)


if __name__ == "__main__":
    # facebook-GAT: d=0.1
    # pokec-GAT: d=0.1
    # facebook-GCN: d=0.05
    # pokec-GCN: d=0.1
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="GAT", help='Model Type, GAT or gcn')
    parser.add_argument('--dataset', type=str, default="facebook", help='dataset, facebook or cora')
    args = parser.parse_args()

    dataset = args.dataset
    model_type = args.model
    fair_sample = True
    Graph_name = "{}_0.2{}_attack3".format(dataset, "_fair" if fair_sample else "")
    delta = 0

    if delta > 0:
        datasource = "{}/CNR/Group/Reduce/Delta={}/".format(model_type, delta)
    else:
        datasource = "{}/".format(model_type)

    attack_res = datasource + "MIA_res/t=0/{}.csv".format(Graph_name)
    adj_file = datasource + "ind.{}.adj".format(dataset)
    adj = pkl.load(open(adj_file, 'rb')).detach().numpy()

    attack_df = pd.read_csv(attack_res)
    mem_ind = attack_df["Label"] == 1
    node1_mem = attack_df[mem_ind]["Node1"].to_numpy()
    node2_mem = attack_df[mem_ind]["Node2"].to_numpy()
    node1_nonmem = attack_df[~mem_ind]["Node1"].to_numpy()
    node2_nonmem = attack_df[~mem_ind]["Node2"].to_numpy()
    g = igraph.Graph.Adjacency((adj > 0).tolist())
    #res = get_5_weights(g, node1, node2)
    mem_res = get_5_weights(g, node1_mem, node2_mem)
    nonmem_res = get_weights_with_batch(g, node1_nonmem, node2_nonmem, 40)
    res = np.vstack([mem_res, nonmem_res])
    res_df = pd.DataFrame(res, columns=['node1', 'node2',
                                        'member', #'dgr_dot',
                                        #'bc_dot', 'cc_dot',
                                        #'ec_dot',
                                        'bc_edge'])
    if not os.path.exists("{}/weight".format(model_type)):
        os.makedirs("{}/weight".format(model_type))
    res_df.to_csv("{}/weight/{}_all_weights.csv".format(model_type,
                                                        dataset))
