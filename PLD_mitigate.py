import numpy as np
from utils import load_data
from common_Neighbor import *


if __name__ == "__main__":
    from_delta = 0.05
    datapath = "GAT/CNR/Group/Reduce/Delta={}/".format(from_delta)
    #datapath = "GAT/"
    dataset = "facebook_ptb"
    ego_user = "107"
    model_type = "GAT"
    adj, ft, gender, labels = load_data(datapath, dataset, ego_user)
    links = adj.nonzero().numpy()
    g_list = get_groups_of_links(links, gender)
    pr_list = find_pure_node(adj, gender)
    origin_cnr = get_average_cnr(adj, gender, 0)
    g_to_reduce = np.argmax(origin_cnr) - 2
    delta = origin_cnr[g_to_reduce + 2] - origin_cnr[3 - g_to_reduce + 2]
    # todo: how to set how many pure neighbors can we use? 1. 5% of all nodes 2. nodes with pr higher than threshold
    p_pure = 0.05
    node_ind_g = gender == g_to_reduce

    pure_list = np.argsort(pr_list)[::-1]
    gender_sort_by_pure = gender[pure_list]
    g_pure_list = pure_list[gender_sort_by_pure == g_to_reduce]
    g_pure_list = g_pure_list[: round(p_pure * len(g_pure_list))]

    ind_g = g_list == g_to_reduce
    links_g = links[ind_g]
    links_to_perturb = get_partial_links(links_g, p=0.01, pure_nodes=g_pure_list)

    cnrs = get_cnr_for_links(links_to_perturb, adj)
    # cnrs_un = get_cnr_for_links(unlinks_to_perturb, adj)
    count = 0
    for l in links_to_perturb:
        Nx, adj = reduce_neighbor_by_delta3(l, delta*2, adj, gender, g_pure_list)
        count += 1
        if count % 2 == 0:
            tmp_cnr = get_average_cnr(adj, gender, delta)
            if tmp_cnr[g_to_reduce + 2] < origin_cnr[g_to_reduce + 2] - delta:
                print("Target delta has been reached.")
                break
    cnrs_new = get_cnr_for_links(links_to_perturb, adj)
    print("\nAverage cnr is reduced from {:.3f} to {:.3f}".format(cnrs.mean(), cnrs_new.mean()))
    cnr_array = get_average_cnr(adj, gender, delta)
    rT, rP, rA = [True, True, True]
    acc_array = experiment_full(adj, gender, ft, labels, dataset, model_type, links_to_perturb,
                                run_Target=rT, run_partial=rP, run_MIA=rA,
                                exp="Group/mitigation_{}/".format(from_delta), delta=delta)
    acc_final = np.vstack([acc_array, cnr_array])
    print(acc_final)


    pass
