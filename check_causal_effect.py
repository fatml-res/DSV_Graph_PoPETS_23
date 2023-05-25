# 1. sample group
## 1.1 sample nodes with ratio
## 1.2 extract edges from original graph
## 1.3 check density:
### 1.3.1 if density is larger than average density, remove edge to achieve 0.5*density
### 1.3.2 if density is smaller than average density, add edge to achieve 2 * density

## 1.4 save two graphs in sample-graph/graph_i.pkl and sample-graph/graph_i_ptb.pkl
### 1.4.1: each .pkl file include:
#### - node list: List start from 1, length=Nv
#### - edge list: array, [2, Ne]
#### - ft-matrix: array, [ft, Nv]
#### - labels: array, length=Nv
#### - edge properties: array, [2*Ne] (NS, EBC)
#### - density: float
#### - density_label: bool, 1 means high density

# 2. target experiment
# 3. partial generation: 20%
# 4. attack experiment
import json

import pandas as pd

from utils import *
from get_all_weight import get_5_weights, get_weights_with_batch
from stealing_link.partial_graph_generation import get_partial
import igraph
from common_Neighbor import cnr_link
from run_target import run_target
from GCN_dense import train_model
from attack import attack_main
import argparse
from tqdm import tqdm


def get_ori_graph(dataset, model_type):
    datapath = "dataset/"
    ego_user = "107"
    with open('model_config.json', 'r') as f:
        config = json.load(f)[dataset][model_type]
    delta = config["delta"]
    adj, ft, gender, labels = load_data(datapath, dataset, ego_user, dropout=0)
    if torch.is_tensor(labels):
        labels = torch.LongTensor(labels.numpy().astype(int))
    if delta > 0:
        adj = pkl.load(open(config["adj_location"], "rb"))
    return np.array(adj), np.array(ft), np.array(gender), np.array(labels)


def get_edge_properties(adj):
    g = igraph.Graph.Adjacency((adj > 0).tolist())
    n1, n2 = adj.nonzero()
    n1, n2 = n1[n1<n2], n2[n1<n2]
    n1n, n2n = (1-adj).nonzero()
    n1n, n2n = n1n[n1n < n2n], n2n[n1n < n2n]
    ind_sample = np.random.choice(np.arange(len(n1n)), len(n1), replace=False)
    n1n = n1n[ind_sample]
    n2n = n2n[ind_sample]
    mem_res = get_5_weights(g, n1, n2)
    if len(adj) > 2000:
        nonmem_res = get_weights_with_batch(g, n1n, n2n, batch_size=40)
    else:
        nonmem_res = get_5_weights(g, n1n, n2n)

    cnr_list = []
    # get cnrs
    ''' for i in tqdm(range(len(mem_res))):
        cnr_list.append(cnr_link(n1[i],
                                 n2[i],
                                 adj))
        for i in tqdm(range(len(nonmem_res))):
        cnr_list.append(cnr_link(n1n[i],
                                 n2n[i],
                                 adj))'''
    cnr_mem = np.nan_to_num(((adj[n1] * adj[n2])>0).sum(axis=1) / ((adj[n1] + adj[n2])>0).sum(axis=1), 0)
    cnr_nmem = np.nan_to_num(((adj[n1n] * adj[n2n]) > 0).sum(axis=1) / ((adj[n1n] + adj[n2n]) > 0).sum(axis=1), 0)
    all_arr = np.hstack([np.vstack([mem_res, nonmem_res]),
                         np.hstack([cnr_mem, cnr_nmem]).reshape(-1, 1)])
    return all_arr


def counterGraph(dict_ori_graph, density_all):
    if dict_ori_graph["density-label"]: # high density, reduce
        reduce = True
    else:
        reduce = False
    num_node = len(dict_ori_graph['labels'])
    total_all = num_node * (num_node - 1) / 2
    change_ratio = np.random.uniform()

    if reduce:
        target_density = density_all * (1 - change_ratio)
        target_edges = int(target_density * total_all)
        edges_to_reduce = int(len(dict_ori_graph['edges']) / 2 - target_edges)

        # sample from member and non-member to remove
        df_curr = pd.DataFrame(dict_ori_graph['edges'], columns=["n1", "n2", "mem", "EBC", "NS"])
        df_to_reduce = df_curr.groupby("mem").sample(n=edges_to_reduce)
        adj = dict_ori_graph["adj"].copy()
        adj[df_to_reduce['n1'].astype(int), df_to_reduce['n2'].astype(int)] = 0
        adj[df_to_reduce['n2'].astype(int), df_to_reduce['n1'].astype(int)] = 0

        ft = dict_ori_graph["ft"].copy()
        labels = dict_ori_graph['labels']
        edges = get_edge_properties(adj)
        dict_current_graph = {"adj": adj,
                              "ft": ft,
                              "labels": labels,
                              "edges": edges,
                              "density": target_density,
                              "density-label": not reduce}


    else:
        target_density = density_all * (1+change_ratio)
        target_edges = int(target_density * total_all)
        edges_to_add = int(target_edges - len(dict_ori_graph['edges']) / 2)

        # sample from member and non-member to remove
        df_curr = pd.DataFrame(dict_ori_graph['edges'], columns=["n1", "n2", "mem", "EBC", "NS"])
        adj = dict_ori_graph["adj"].copy()

        n1n, n2n = (1 - adj).nonzero()
        n1n, n2n = n1n[n1n < n2n], n2n[n1n < n2n]
        ind_sample = np.random.choice(np.arange(len(n1n)), edges_to_add, replace=False)
        n1n, n2n = n1n[ind_sample], n2n[ind_sample]
        adj[n1n, n2n] = 1
        adj[n1n, n2n] = 1

        ft = dict_ori_graph["ft"].copy()
        labels = dict_ori_graph['labels']
        edges = get_edge_properties(adj)
        dict_current_graph = {"adj": adj,
                              "ft": ft,
                              "labels": labels,
                              "edges": edges,
                              "density": target_density,
                              "density-label": not reduce}
    return dict_current_graph




def sample_graph(adj, ft, labels, ratio,
                 redo=True, ind=0, model="GAT", dataset="facebook"):
    if not redo:
        pkl_file = f"sample-graph/{dataset}_graph_{ind}/graph_detail.pkl"
        with open(pkl_file, 'rb') as f:
            if sys.version_info > (3, 0):
                dict1 = pkl.load(f, encoding='latin1')
            else:
                dict1 = pkl.load(f)

        pkl_file = f"sample-graph/{dataset}_graph_{ind + 1}/graph_detail.pkl"
        with open(pkl_file, 'rb') as f:
            if sys.version_info > (3, 0):
                dict2 = pkl.load(f, encoding='latin1')
            else:
                dict2 = pkl.load(f)
        return dict1, dict2
    num_node = len(ft)
    number_to_sample = int(ratio * num_node)
    node_samples = np.random.choice(np.arange(num_node), number_to_sample, replace=False)

    adj_sample = adj[node_samples][:, node_samples]
    ft_sample = ft[node_samples]
    labels_sample = labels[node_samples]
    edge_properties = get_edge_properties(adj_sample)

    total = number_to_sample * (number_to_sample - 1) / 2
    total_all = num_node * (num_node - 1) / 2
    density = len(adj_sample.nonzero()[0]) / total / 2
    density_all = len(adj.nonzero()[0]) / total_all / 2

    density_label = density > density_all
    # dict of this graph
    dict_current_graph = {"adj": adj_sample,
                          "ft": ft_sample,
                          "labels": labels_sample,
                          "edges": edge_properties,
                          "density": density,
                          "density-label": density_label}

    # get counterfactual graph detail
    dict_cf_graph = counterGraph(dict_current_graph, density_all)
    return dict_current_graph, dict_cf_graph


def DTPA_experiment(dict, ind, redo=True, dataset="facebook", model="GAT"):
    if model =="GCN":
        sub_graph_config = {
            "delta": 0.05,
            "nhid": 32,
            "lr": 0.001,
            "train": 0.6,
            "val": 0.2,
            "patience": 60,
            "dropout": 0.5}
    else:
        sub_graph_config = {
            "delta": 0.1,
            "nhid": 8,
            "nheads": 8,
        "lr": 0.004,
        "train": 0.5,
        "val": 0.3,
        "patience": 20,
        "dropout": 0.5}
    if redo:
        train_model([], dict['ft'], dict['adj'], dict['labels'], f"{dataset}_graph_{ind}", num_epoch=50, model_type="dense",
                saving_path="dense")

    run_target(model, sub_graph_config, [], dict['ft'], dict['adj'], dict['labels'],
               DP=False, Ptb=False, epochs=50, dataset=f"{dataset}_graph_{ind}", saving_path=f"sample-graph/{dataset}_graph_{ind}/{model}",
               null_model=False, ARR=False, mpre=False)
    # 3. partial generation: 20%
    get_partial(adj=dict['adj'], model_type=model, datapath=f"sample-graph/{dataset}_graph_{ind}/",
                pred_path=f"sample-graph/{dataset}_graph_{ind}/{model}",
                partial_path=f"sample-graph/{dataset}_graph_{ind}/{model}/",
                dataset=f"{dataset}_graph_{ind}", fair_sample=False, t=0)
    edge_prop = dict['edges'][dict['edges'][:, 2] == 1, -2:].mean(axis=0)
    res_list = []
    for at in [3, 6]:
        a, p, r, roc, acc_list = attack_main(datapath=f"sample-graph/{dataset}_graph_{ind}/{model}/partial/",
                                             dataset=f"{dataset}_graph_{ind}",
                                             saving_path=f"sample-graph/{dataset}_graph_{ind}/{model}/",
                                             ratio=0.2,
                                             attack_type=at,
                                             fair_sample=False,
                                             t=0)
        res_list.append([a, p, r, roc, 2 * p * r / (p + r), at, dict['density'], dict['density-label'], edge_prop[0],
             edge_prop[1]])

    return res_list






if __name__ == "__main__":
    # 1. sample group
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="GAT", help='Model Type, GAT or gcn')
    parser.add_argument('--dataset', type=str, default="facebook", help='Model Type, GAT or gcn')
    parser.add_argument('--reget_graph', type=int, default=1, help="regenerate graph")
    args = parser.parse_args()
    model = args.model
    dataset = args.dataset
    adj, ft, _, labels = get_ori_graph(dataset, model)
    count = 10
    ind = 0
    ## 1.1 sample nodes with ratio
    ## 1.2 extract edges from original graph
    ## 1.3 check density:
    ### 1.3.1 if density is larger than average density, remove edge to achieve 0.5*density
    ### 1.3.2 if density is smaller than average density, add edge to achieve 2 * density
    attack_results = []
    for c in range(count):
        for ratio in [0.05, 0.1, 0.15, 0.2]:
            if "tagged" in dataset and ratio == 0.05:
                continue
            dict_1, dict_2 = sample_graph(adj, ft, labels, ratio, redo=args.reget_graph, ind=ind, dataset=dataset)
            ## 1.4 save two graphs in sample-graph/graph_i.pkl and sample-graph/graph_i_ptb.pkl
            ### 1.4.1: each .pkl file include:
            #### - adj: array, [Nv, Nv]
            #### - ft: array, [ft, Nv]
            #### - labels: array, length=Nv
            #### - edge properties: array, [5, 2* Ne] (n1, n2, mem, EBC, NS)
            #### - density: float
            #### - density_label: bool, 1 means high density
            pkl_1 = f"sample-graph/{dataset}_graph_{ind}/graph_detail.pkl"
            if not os.path.exists(f"sample-graph/{dataset}_graph_{ind}"):
                os.makedirs(f"sample-graph/{dataset}_graph_{ind}")

            with open(pkl_1, 'wb') as f:
                pkl.dump(dict_1, f)

            # 2. target experiment on sample graph
            res_sub1 = DTPA_experiment(dict_1, ind, redo=args.reget_graph,
                                       dataset=dataset, model=model)
            attack_results += res_sub1

            ind += 1
            pkl_2 = f"sample-graph/{dataset}_graph_{ind}/graph_detail.pkl"

            if not os.path.exists(f"sample-graph/{dataset}_graph_{ind}"):
                os.makedirs(f"sample-graph/{dataset}_graph_{ind}")
            with open(pkl_2, 'wb') as f:
                pkl.dump(dict_2, f)

            res_sub2 = DTPA_experiment(dict_2, ind, redo=args.reget_graph,
                            dataset=dataset, model=model)
            attack_results += res_sub2

            ind += 1
        if (c + 1)%5 == 0:
            df_attack = pd.DataFrame(attack_results, columns=["Acc", "Prec", "Recall", "ROC", "F1",
                                                              "Attack", 'Density', "Density-label", "AEBC", "ANS"])
            df_attack.to_csv("sample-graph/Attack-sub.csv", index=False)
    df_attack = pd.DataFrame(attack_results, columns=["Acc", "Prec", "Recall", "ROC", "F1",
                                                      "Attack", 'Density', "Density-label", "AEBC", "ANS"])
    df_attack.to_csv(f"sample-graph/{dataset}_Attack-sub-{model}.csv", index=False)


    # 4. attack experiment


