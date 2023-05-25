import copy
import os.path

import numpy as np
from utils import load_data
import torch
from GAT import run_GAT
from GCN import run_GCN
import pickle as pkl
from stealing_link.partial_graph_generation import get_partial
import pandas as pd
from GCN_dense import train_model
from attack import attack_main

def cnr_link(node1, node2, adj):
    '''
    :param node1: id of node 1
    :param node2: id of node 2
    :param adj: adjacency matrix of graph
    :return: the common neighbor rate of this link
    '''
    if not torch.is_tensor(adj):
        adj = torch.LongTensor(adj)
    UN = torch.logical_or(adj[node1] > 0, adj[node2] > 0).sum().item()
    CN = torch.logical_and(adj[node1] > 0, adj[node2] > 0).sum().item()
    if UN == 0:
        return 0
    return CN/UN


def acnr_group(links, adj):
    cnr_list = []
    for link in links:
        node1, node2 = link
        cnr = cnr_link(node1, node2, adj)
        cnr_list.append(cnr)
    cnr_list = np.array(cnr_list)
    return cnr_list.mean()


def get_partial_links(links, p=0.1, pure_nodes=[]):
    num_links = round(len(links) * p)
    np.random.shuffle(links)
    i = 0
    count = 0
    node_set = set()
    link_to_perturb = []
    while i < len(links) and count < num_links:
        tmp_link = links[i]
        if tmp_link[0] in node_set or tmp_link[1] in node_set or tmp_link[1] == tmp_link[0] or tmp_link[0] in pure_nodes or tmp_link[1] in pure_nodes:
            pass
        else:
            link_to_perturb.append(tmp_link)
            node_set.add(tmp_link[0])
            node_set.add(tmp_link[1])
            count += 1
        i += 1
    if count < num_links:
        print("Warning! No isolated links ({}) compared with {}, please reduce p!".format(count, num_links))
    return np.array(link_to_perturb)


def get_cnr_for_links(links, adj):
    res = []
    for l in links:
        cnr = cnr_link(l[0], l[1], adj)
        res.append(cnr)
    return np.array(res)


def decrease_neighbor_by_x(link, x, adj):
    node1, node2 = link
    init_cnr = cnr_link(node1, node2, adj)
    # add edge to node2
    N1 = adj[node1].nonzero().numpy().reshape(-1)
    list_of_nodes = (adj[node2] == 0).nonzero().numpy().reshape(-1)
    count = 0
    i = 0
    np.random.shuffle(list_of_nodes)
    Nx = []
    while count < x and i < len(list_of_nodes):
        if list_of_nodes[i] != node1 and list_of_nodes[i] not in N1:
            Nx.append(list_of_nodes[i])
            adj[node2, list_of_nodes[i]] = 1
            adj[list_of_nodes[i], node2] = 1
            count += 1
        i += 1
    if count < x:
        print("Warning! there is only {} isolated non-neighbors for node {} instead of {}".format(count,
                                                                                                  node2,
                                                                                                  x))
    final_cnr = cnr_link(node1, node2, adj)
    print("The cnr between Node {} and Node {} decreased from {:.2f} to {:.2f}".format(node1, node2, init_cnr, final_cnr))
    return np.array(Nx), adj


def increase_neighbor_by_x(link, x, adj):
    node1, node2 = link
    init_cnr = cnr_link(node1, node2, adj)
    # add edge to node2
    N1 = adj[node1].nonzero().numpy().reshape(-1)
    N2 = adj[node2].nonzero().numpy().reshape(-1)
    list_of_nodes = np.array([x for x in N1 if x not in N2] + [x for x in N2 if x not in N1])
    count = 0
    i = 0
    np.random.shuffle(list_of_nodes)
    Nx = []
    while count < x and i < len(list_of_nodes):
        if list_of_nodes[i] != node1 and list_of_nodes[i] != node2:
            Nx.append(list_of_nodes[i])
            adj[node1, list_of_nodes[i]] = 1
            adj[list_of_nodes[i], node1] = 1
            adj[node2, list_of_nodes[i]] = 1
            adj[list_of_nodes[i], node2] = 1
            count += 1
        i += 1
    if count < x:
        print("Warning! there is only {} isolated non-neighbors for node {} instead of {}".format(count,
                                                                                                  node2,
                                                                                                  x))
    final_cnr = cnr_link(node1, node2, adj)
    print("The cnr between Node {} and Node {} increased from {:.2f} to {:.2f}".format(node1, node2, init_cnr, final_cnr))
    return np.array(Nx), adj


def increase_neighbor_by_delta(link, delta, adj, gender, g):
    node1, node2 = link
    init_cnr = cnr_link(node1, node2, adj)
    # add edge to node2
    N1 = adj[node1].nonzero().numpy().reshape(-1)
    N2 = adj[node2].nonzero().numpy().reshape(-1)
    # While increasing CNR of group, some uncommon neighbor will be removed. We should remove link not in the group
    if g == 0:
        list_of_nodes = np.array([x for x in N1 if (x not in N2 and gender[x] == gender[node1])] + [x for x in N2 if
                                                                                                    x not in N1
                                                                                                    and gender[x] == gender[node2]])
    else:
        list_of_nodes = np.array([x for x in N1 if (x not in N2 and gender[x] != gender[node1])] + [x for x in N2
                                                                                                    if x not in N1
                                                                                                    and gender[x] != gender[node1]])
    CN = len([x for x in N1 if x in N2])
    UN = len(N2) + len(N1) - CN
    x = get_x_from_delta(delta, UN, CN)
    count = 0
    i = 0
    np.random.shuffle(list_of_nodes)
    Nx = []
    while count < x and i < len(list_of_nodes):
        if list_of_nodes[i] != node1 and list_of_nodes[i] != node2:
            Nx.append(list_of_nodes[i])
            adj[node1, list_of_nodes[i]] = 0
            adj[list_of_nodes[i], node1] = 0
            adj[node2, list_of_nodes[i]] = 0
            adj[list_of_nodes[i], node2] = 0
            count += 1
        i += 1
    if count < x:
        print("Warning! there is only {} isolated non-neighbors for node {} instead of {}".format(count,
                                                                                                  node2,
                                                                                                  x))
    final_cnr = cnr_link(node1, node2, adj)
    print("The cnr between Node {} and Node {} increased from {:.2f} to {:.2f}".format(node1, node2, init_cnr, final_cnr))
    return np.array(Nx), adj


def increase_neighbor_by_delta2(link, delta, adj, gender, g, pure_list=[]):
    node1, node2 = link
    init_cnr = cnr_link(node1, node2, adj)
    # add edge to node2
    N1 = adj[node1].nonzero().numpy().reshape(-1)
    N2 = adj[node2].nonzero().numpy().reshape(-1)
    # While increasing CNR of group, some uncommon neighbor will be removed. We should remove link not in the group
    if g != 0:
        list_of_nodes = np.array([x for x in N1 if (x not in N2 and gender[x] == gender[node1])] + [x for x in N2 if
                                                                                                    x not in N1
                                                                                                    and gender[x] == gender[node2]])
    else:
        list_of_nodes = np.array([x for x in N1 if (x not in N2 and gender[x] != gender[node1])] + [x for x in N2
                                                                                                    if x not in N1
                                                                                                    and gender[x] != gender[node1]])
    CN = len([x for x in N1 if x in N2])
    UN = len(N2) + len(N1) - CN
    x = get_x_from_delta(delta, UN, CN)
    #x = delta * UN
    count = 0
    i = 0
    np.random.shuffle(list_of_nodes)
    Nx = []
    while count < x and i < len(list_of_nodes):
        if list_of_nodes[i] != node1 and list_of_nodes[i] != node2 and list_of_nodes[i] in pure_list:
            Nx.append(list_of_nodes[i])
            adj[node1, list_of_nodes[i]] = 0
            adj[list_of_nodes[i], node1] = 0
            adj[node2, list_of_nodes[i]] = 0
            adj[list_of_nodes[i], node2] = 0
            count += 1
        i += 1
    if count < x:
        print("Warning! there is only {} isolated non-neighbors for node {} instead of {}".format(count,
                                                                                                  node2,
                                                                                                  x))
    final_cnr = cnr_link(node1, node2, adj)
    print("The cnr between Node {} and Node {} increased from {:.2f} to {:.2f}".format(node1, node2, init_cnr, final_cnr))
    return np.array(Nx), adj


def reduce_neighbor_by_delta(link, delta, adj, gender, g):
    node1, node2 = link
    init_cnr = cnr_link(node1, node2, adj)
    # add edge to node2
    N1 = adj[node1].nonzero().numpy().reshape(-1)
    N2 = adj[node2].nonzero().numpy().reshape(-1)
    if len(N1) > len(N2):
        node1, node2 = node2, node1
        N1, N2 = N2, N1
    # To reduce the cnr, we need add some uncommon edge. For group 0, we should add inner gender edge, for group 2, we
    # should add intra gender edge
    if g == 0:
        list_of_nodes = [x for x in (adj[node2] == 0).nonzero().numpy().reshape(-1) if gender[x] == gender[node2]]
    else:
        list_of_nodes = [x for x in (adj[node2] == 0).nonzero().numpy().reshape(-1) if gender[x] != gender[node2]]
    CN = len([x for x in N1 if x in N2])
    UN = len(N2) + len(N1) - CN
    x = get_x_from_delta(-delta, UN, CN)
    count = 0
    i = 0
    np.random.shuffle(list_of_nodes)
    Nx = []
    while count < x and i < len(list_of_nodes):
        if list_of_nodes[i] != node1 and list_of_nodes[i] != node2 and list_of_nodes[i] not in N1:
            Nx.append(list_of_nodes[i])
            adj[node2, list_of_nodes[i]] = 1
            adj[list_of_nodes[i], node2] = 1
            count += 1
        i += 1
    if count < x:
        print("Warning! there is only {} isolated non-neighbors for node {} instead of {}".format(count,
                                                                                                  node2,
                                                                                                  x))
    final_cnr = cnr_link(node1, node2, adj)
    print("The cnr between Node {} and Node {} increased from {:.2f} to {:.2f}".format(node1, node2, init_cnr, final_cnr))
    return np.array(Nx), adj


def reduce_neighbor_by_delta2(link, delta, adj, gender, g, pure_list=[]):
    node1, node2 = link
    init_cnr = cnr_link(node1, node2, adj)
    # add edge to node2
    N1 = adj[node1].nonzero().numpy().reshape(-1)
    N2 = adj[node2].nonzero().numpy().reshape(-1)
    if len(N1) > len(N2):
        node1, node2 = node2, node1
        N1, N2 = N2, N1
    # To reduce the cnr, we need add some uncommon edge.
    if g == 2:
        list_of_nodes = [x for x in (adj[node2] == 0).nonzero().numpy().reshape(-1) if gender[x] == gender[node2]]
    else:
        list_of_nodes = [x for x in (adj[node2] == 0).nonzero().numpy().reshape(-1) if gender[x] != gender[node2]]
    CN = len([x for x in N1 if x in N2])
    UN = len(N2) + len(N1) - CN
    x = get_x_from_delta(-delta, UN, CN)
    count = 0
    i = 0
    np.random.shuffle(list_of_nodes)
    Nx = []
    while count < x and i < len(list_of_nodes):
        if list_of_nodes[i] != node1 and list_of_nodes[i] != node2 and list_of_nodes[i] not in N1 and list_of_nodes[i] in pure_list:
            if gender[list_of_nodes[i]]==1:
                pass
            Nx.append(list_of_nodes[i])
            adj[node2, list_of_nodes[i]] = 1
            adj[list_of_nodes[i], node2] = 1
            count += 1
        i += 1
    if count < x:
        print("Warning! there is only {} isolated non-neighbors for node {} instead of {}".format(count,
                                                                                                  node2,
                                                                                                  x))
    final_cnr = cnr_link(node1, node2, adj)
    print("The cnr between Node {} and Node {} reduced from {:.2f} to {:.2f}".format(node1, node2, init_cnr, final_cnr))
    return np.array(Nx), adj


def reduce_neighbor_by_delta3(link, delta, adj, gender, pure_list=[]):
    node1, node2 = link
    init_cnr = cnr_link(node1, node2, adj)
    # add edge to node2
    N1 = adj[node1].nonzero().numpy().reshape(-1)
    N2 = adj[node2].nonzero().numpy().reshape(-1)
    if len(N1) > len(N2):
        node1, node2 = node2, node1
        N1, N2 = N2, N1
    # To reduce the cnr, we need add some uncommon edge.
    CN = len([x for x in N1 if x in N2])
    UN = len(N2) + len(N1) - CN
    x = get_x_from_delta(-delta, UN, CN)
    count = 0
    i = 0
    Nx = []
    while count < x and i < len(pure_list):
        if pure_list[i] != node1 and pure_list[i] != node2 and pure_list[i] not in N1 and pure_list[i] not in N2:
            if gender[pure_list[i]] != gender[node1]:
                pass
            Nx.append(pure_list[i])
            adj[node2, pure_list[i]] = 1
            adj[pure_list[i], node2] = 1
            count += 1
        i += 1
    if count < x:
        print("Warning! there is only {} isolated non-neighbors for node {} instead of {}".format(count,
                                                                                                  node2,
                                                                                                  x))
    final_cnr = cnr_link(node1, node2, adj)
    print("The cnr between Node {} and Node {} reduced from {:.2f} to {:.2f}".format(node1, node2, init_cnr, final_cnr))
    return np.array(Nx), adj


def get_acc_for_links(df, links):
    df_sub = df[df[["Node1", "Node2"]].apply(tuple, axis=1).isin(list(map(tuple, links)))]
    df_other = df[~df[["Node1", "Node2"]].apply(tuple, axis=1).isin(list(map(tuple, links)))]
    return (df_sub['Label'] == df_sub['Pred']).mean(), (df_other['Label'] == df_other['Pred']).mean()


def get_x_from_delta(delta, UN, CN):
    # increase
    if delta > 0:
        x = round((UN**2) * delta/(CN + UN * delta))
        # x = round(delta * UN)
    else:
        try:
            x = round((UN**2) * (-delta)/(CN + UN * delta))
        except:
            x = round((UN**2) * (-delta * 0.9)/(CN + UN * delta*0.9))
    return max(0, x)


def experiment_full(adj, gender, ft, labels, dataset, model_type,
                    links_to_perturb, run_Target=False, run_partial=False, run_MIA=False, exp="Increase/", delta=0):
    res = [] # Attack type, delta, acc0, acc1, acc2, cnr0, cnr1, cnr2
    # Train target model using new adj
    epoch = 300
    c = 0
    while run_Target and c < 5:
        if model_type == "GAT":
            run_Target = run_GAT(gender, ft, adj, labels, epochs=epoch, dataset=dataset, saving_path="GAT/CNR/" + exp)
            c += 1
        else:
            #train_model(gender, ft, adj, labels, dataset, num_epoch=epoch, model_type="gcn", saving_path="gcn/CNR/" + exp)
            run_Target = run_GCN(gender, ft, adj, labels, epochs=epoch, dataset=dataset, saving_path="gcn/CNR/" + exp)
            break
    if run_partial:
        get_partial(model_type=model_type.lower(), datapath=model_type + "/CNR/" + exp,
                    dataset=dataset, fair_sample=True, t=0, ptb=True)

    shadow = "facebook" if dataset=="pokec" else "cora"

    for attack_type in [3, 6]:
        if run_MIA:
            a, p, r, roc, acc_list = attack_main(datapath=model_type + "/CNR/" + exp,
                                                 dataset=dataset.split("_ptb")[0],
                                                 shadow_data=shadow,
                                                 ratio=0.2,
                                                 attack_type=attack_type,
                                                 fair_sample=True,
                                                 t=0,
                                                 prepare_new=True)
            print("Disparity between group 1 and group 2 is {:.2%}".format(acc_list[6] - acc_list[7]))
        else:
            acc_list = [0]*9

        # pull original MIA result and aggregate acc:
        res_file = model_type + "/MIA_res/t=0/{}_0.2_fair_attack{}.csv".format(dataset.split("_ptb")[0], attack_type)
        df = pd.read_csv(res_file)
        original_acc, _ = get_acc_for_links(df, links_to_perturb)

        # pull perturbed MIA result and aggregate acc:
        res_file = model_type + "/CNR/" + exp + "MIA_res/t=0/{}_0.2_fair_attack{}.csv".format(dataset.split("_ptb")[0], attack_type)
        df_p = pd.read_csv(res_file)
        perturb_acc, other_acc = get_acc_for_links(df_p, links_to_perturb)

        print("The MIA testing accuracy for perturbed links changed from {:.2%} to {:.2%}".format(original_acc,
                                                                                                      perturb_acc))
        print("The disparity between perturbed and un-perturbed data is {:.2%} - {:.2%} = {:.2%}".format(perturb_acc,
                                                                                                         other_acc,
                                                                                                         perturb_acc-other_acc))
        res.append([attack_type, delta, acc_list[8], acc_list[6], acc_list[7]])
    return np.array(res) # 2*5: attack, delta, acc0,1,2


def increase_cnr_experiment(adj, dataset, run_new=False):
    if run_new:
        links = adj.nonzero().numpy()
        links_to_perturb = get_partial_links(links, p=0.005)
        pkl.dump(links_to_perturb, open("GAT/CNR/Increase/{}_linkstp.pkl".format(dataset), 'wb'))
        #unlinks_to_perturb = get_partial_links(unlinks, adj, p=0.005)
        cnrs = get_cnr_for_links(links_to_perturb, adj)
        # cnrs_un = get_cnr_for_links(unlinks_to_perturb, adj)
        for l in links_to_perturb:
            Nx, adj = increase_neighbor_by_x(l, 20, adj)
        cnrs_new = get_cnr_for_links(links_to_perturb, adj)
        print("\nAverage cnr is increased from {:.3f} to {:.3f}".format(cnrs.mean(), cnrs_new.mean()))
        rT, rP, rA = [True, True, True]
        experiment_full(adj, dataset, links_to_perturb, run_Target=rT, run_partial=rP, run_MIA=rA, exp="Increase/")
    else:
        new_adj = pkl.load(open("GAT/CNR/Increase/ind.{}.adj".format(dataset), "rb"))
        links_to_perturb = pkl.load(open("GAT/CNR/Increase/{}_linkstp.pkl".format(dataset), "rb"))
        rT, rP, rA = [False, False, False]
        experiment_full(new_adj, dataset, links_to_perturb, run_Target=rT, run_partial=rP, run_MIA=rA, exp="Increase/")


def get_groups_of_links(links, gender):
    g_list = []
    for l in links:
        node1, node2 = l
        if gender[node1] == gender[node2]:
            if gender[node1] == 1:
                g_list.append(1)
            else:
                g_list.append(2)
        else:
            g_list.append(0)
    return np.array(g_list)


def get_average_cnr(adj, gender, delta):
    new_links = adj.nonzero().numpy()
    new_g_list = get_groups_of_links(new_links, gender)
    new_cnrs = [0, delta]
    for gi in [0, 1, 2]:
        ind_gi = new_g_list == gi
        links_gi = new_links[ind_gi]
        cnrs_group = get_cnr_for_links(links_gi, adj)
        new_cnrs.append(cnrs_group.mean())
        #print("Average cnr for group {} is {:.3} after Increase Delta={}".format(gi, cnrs_group.mean(), delta))
    return np.array(new_cnrs)


def increase_cnr_group_experiment(adj, dataset, model_type, gender, delta=0.3, run_new=False):
    if run_new:
        links = adj.nonzero().numpy()
        g_list = get_groups_of_links(links, gender)
        #g_to_perturb = 1 if model_type == "GAT" else 2
        g_to_perturb = 2
        origin_cnr = get_average_cnr(adj, gender, delta)
        for g in [g_to_perturb]:
            ind_g = g_list == g
            links_g = links[ind_g]

            pr_list = find_pure_node(adj, gender)
            pure_list = np.where(pr_list >= 0.8)[0]
            links_to_perturb = get_partial_links(links_g, p=0.005, pure_nodes=pure_list)
            if not os.path.exists("{}/CNR/Group/Increase/Delta={}".format(model_type, delta)):
                os.makedirs("{}/CNR/Group/Increase/Delta={}".format(model_type, delta))
            pkl.dump(links_to_perturb, open("{}/CNR/Group/Increase/Delta={}/{}_linkstp.pkl".format(model_type, delta, dataset), 'wb'))
            #unlinks_to_perturb = get_partial_links(unlinks, adj, p=0.005)
            cnrs = get_cnr_for_links(links_to_perturb, adj)
            # cnrs_un = get_cnr_for_links(unlinks_to_perturb, adj)
            for l in links_to_perturb:
                Nx, adj = increase_neighbor_by_delta2(l,  delta * 2, adj, gender, g, pure_list)
                # check target
                '''tmp_cnr = get_average_cnr(adj, gender)
                if tmp_cnr[g+2] > origin_cnr[g+2] + delta:
                    print("Target delta has been reached.")
                    break'''
            cnrs_new = get_cnr_for_links(links_to_perturb, adj)
            print("\nAverage cnr is increased from {:.3f} to {:.3f}".format(cnrs.mean(), cnrs_new.mean()))
            with open("{}/CNR/Group/Increase/Delta={}/ind.{}.adj".format(model_type, delta, dataset), 'wb') as f:
                pkl.dump(adj, f)
            with open("{}/CNR/Group/Increase/Delta={}/{}_linkstp.pkl".format(model_type, delta, dataset), 'wb') as f:
                pkl.dump(links_to_perturb, f)

        cnr_array = get_average_cnr(adj, gender, delta)
        rT, rP, rA = [True, True, True]
        acc_array = experiment_full(adj, dataset, model_type, links_to_perturb, run_Target=rT, run_partial=rP, run_MIA=rA, exp="Group/Increase/Delta={}/".format(delta))
        acc_final = np.vstack([acc_array, cnr_array])
        return acc_final
    else:
        new_adj = pkl.load(open("{}/CNR/Group/Increase/Delta={}/ind.{}.adj".format(model_type, delta, dataset), "rb"))
        links_to_perturb = pkl.load(open("{}/CNR/Group/Increase/Delta={}/{}_linkstp.pkl".format(model_type, delta, dataset), "rb"))
        rT, rP, rA = [False, False, False]
        experiment_full(new_adj, dataset, model_type, links_to_perturb, run_Target=rT, run_partial=rP, run_MIA=rA, exp="Group/Increase/Delta={}/".format(delta))


def reduce_cnr_group_experiment(adj, dataset, model_type, gender, delta=0.3, run_new=False):

    if run_new:
        links = adj.nonzero().numpy()
        g_list = get_groups_of_links(links, gender)
        g_to_perturb = 2 if model_type == "GAT" else 2
        for g in [g_to_perturb]:
            ind_g = g_list == g
            links_g = links[ind_g]
            links_to_perturb = get_partial_links(links_g, p=0.005)
            if not os.path.exists("{}/CNR/Group/Reduce/Delta={}".format(model_type, delta)):
                os.makedirs("{}/CNR/Group/Reduce/Delta={}".format(model_type, delta))
            pkl.dump(links_to_perturb, open("{}/CNR/Group/Reduce/Delta={}/{}_linkstp.pkl".format(model_type, delta, dataset), 'wb'))
            #unlinks_to_perturb = get_partial_links(unlinks, adj, p=0.005)
            cnrs = get_cnr_for_links(links_to_perturb, adj)
            # cnrs_un = get_cnr_for_links(unlinks_to_perturb, adj)
            for l in links_to_perturb:
                Nx, adj = reduce_neighbor_by_delta(l, delta, adj, gender, g)
        cnr_array = get_average_cnr(adj, g_list)
        rT, rP, rA = [True, True, True]
        acc_array = experiment_full(adj, dataset, model_type, links_to_perturb, run_Target=rT, run_partial=rP, run_MIA=rA, exp="Group/Reduce/Delta={}/".format(delta))
        acc_final = np.vstack([acc_array, cnr_array])
        return acc_final
    else:
        new_adj = pkl.load(open("{}/CNR/Group/Reduce/Delta={}/ind.{}.adj".format(model_type, delta, dataset), "rb"))
        links_to_perturb = pkl.load(open("{}/CNR/Group/Reduce/Delta={}/{}_linkstp.pkl".format(model_type, delta, dataset), "rb"))
        rT, rP, rA = [False, False, False]
        experiment_full(new_adj, dataset, model_type, links_to_perturb, run_Target=rT, run_partial=rP, run_MIA=rA, exp="Group/Reduce/Delta={}/".format(delta))


def reduce_cnr_group_experiment2(ori_adj, dataset, model_type, gender, ft, labels, delta=0.3, run_new=False, group_to_reduce=2):
    adj = copy.deepcopy(ori_adj)
    if run_new:
        links = adj.nonzero().numpy()
        g_list = get_groups_of_links(links, gender)
        origin_cnr = get_average_cnr(adj, gender, delta)
        for g in [group_to_reduce]:
            pr_list = find_pure_node(adj, gender)
            pure_list = np.where(pr_list == 1)[0]
            ind_g = g_list == g
            links_g = links[ind_g]
            links_to_perturb = get_partial_links(links_g, p=0.005, pure_nodes=pure_list)
            if not os.path.exists("{}/CNR/Group/Reduce/Delta={}".format(model_type, delta)):
                os.makedirs("{}/CNR/Group/Reduce/Delta={}".format(model_type, delta))
            pkl.dump(links_to_perturb, open("{}/CNR/Group/Reduce/Delta={}/{}_linkstp.pkl".format(model_type, delta, dataset), 'wb'))
            #unlinks_to_perturb = get_partial_links(unlinks, adj, p=0.005)
            cnrs = get_cnr_for_links(links_to_perturb, adj)
            # cnrs_un = get_cnr_for_links(unlinks_to_perturb, adj)
            count = 0
            for l in links_to_perturb:
                Nx, adj = reduce_neighbor_by_delta2(l, delta, adj, gender, g, pure_list)
                count += 1
                if count % 10 == 0:
                    tmp_cnr = get_average_cnr(adj, gender, delta)
                    if tmp_cnr[g+2] < origin_cnr[g+2] - delta:
                        print("Target delta has been reached.")
                        break
            cnrs_new = get_cnr_for_links(links_to_perturb, adj)
            #print("\nAverage cnr is reduced from {:.3f} to {:.3f}".format(cnrs.mean(), cnrs_new.mean()))
        cnr_array = get_average_cnr(adj, gender, delta)
        print("\nAverage cnr of group {} is reduced from {:.3f} to {:.3f}".format(g, origin_cnr[g + 2], cnr_array[g + 2]))
        rT, rP, rA = [True, True, True]
        acc_array = experiment_full(adj, gender, ft, labels,
                                    dataset, model_type, links_to_perturb,
                                    run_Target=rT, run_partial=rP, run_MIA=rA,
                                    exp="Group/Reduce/Delta={}/".format(delta))
        acc_final = np.vstack([acc_array, cnr_array])
        return acc_final
    else:
        new_adj = pkl.load(open("{}/CNR/Group/Reduce/Delta={}/ind.{}.adj".format(model_type, delta, dataset), "rb"))
        links_to_perturb = pkl.load(open("{}/CNR/Group/Reduce/Delta={}/{}_linkstp.pkl".format(model_type, delta, dataset), "rb"))
        rT, rP, rA = [True, True, True]
        cnr_array = get_average_cnr(new_adj, gender, delta)
        print("Loaded cnr as {}".format(cnr_array))
        acc_array = experiment_full(new_adj, gender, ft, labels,
                                    dataset, model_type, links_to_perturb,
                                    run_Target=rT, run_partial=rP, run_MIA=rA,
                                    exp="Group/Reduce/Delta={}/".format(delta))
        acc_final = np.vstack([acc_array, cnr_array])
        return acc_final


def find_pure_node(adj, gender):
    gender = np.array(gender)
    pure_ratio = []
    for id in range(len(adj)):
        n_ids = adj[id].nonzero().numpy()
        neighbor_glist = gender[n_ids]
        pr = (neighbor_glist == gender[id]).mean()
        pure_ratio.append(pr)
    return np.array(pure_ratio)





if __name__ == "__main__":
    datapath = "dataset/"
    dataset = "facebook"
    ego_user = "107"
    model_type = "gcn"
    get_new_adj = False
    adj, ft, gender, labels = load_data(datapath, dataset, ego_user, dropout=0)
    if adj.is_sparse:
        adj = adj.to_dense()
    array_list = []
    for delta in [0.01, 0.05, 0.1, 0.2]:
        #array_increase = increase_cnr_group_experiment(adj, dataset, model_type, min, delta, True)
        array_reduce = reduce_cnr_group_experiment2(adj, dataset, model_type,
                                                    gender, ft, labels, delta,
                                                    get_new_adj, group_to_reduce=1)
        array_list.append(array_reduce)
        #array_list.append(array_increase)
    array_all = np.vstack(array_list)
    print(array_all)
    df_res = pd.DataFrame(array_all, columns=["", "Delta", "Group 0", "Group 1", "Group 2"])
    df_res.to_csv("{}/CNR/Group/{}_agg.csv".format(model_type, dataset))


    # increase_cnr_experiment(adj, dataset, run_new=True)
    # reduce_cnr_experiment(adj, dataset, run_new=True)
    pass




