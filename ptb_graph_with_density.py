import os

import numpy as np
import pandas as pd
import torch

from utils import *
import random
import time


def get_link(adj, node_num, gender):
    unlink = []
    link = []
    g_link = []
    g_unlink = []
    existing_set = set([])
    if torch.is_tensor(adj):
        rows, cols = adj.nonzero().T
    else:
        rows, cols = adj.nonzero()
    print("There are %d edges in this dataset" % len(rows))
    try:
        min_gender, maj_gender = np.unique(gender, return_counts=True)[0][np.unique(gender, return_counts=True)[1].argsort()]
    except:
        min_gender, maj_gender = -2, -2
    for i in range(len(rows)):
        r_index = int(rows[i])
        c_index = int(cols[i])
        try:
            g0, g1 = gender[r_index], gender[c_index]
            if g0 == min_gender and g1 == min_gender:
                tmpg = 1
            elif g0 == maj_gender and g1 == maj_gender:
                tmpg = 2
            else:
                tmpg = 0
        except:
            tmpg = -1
        if r_index < c_index:
            link.append([r_index, c_index])
            g_link.append(tmpg)
            existing_set.add(",".join([str(r_index), str(c_index)]))

    random.seed(1)
    t_start = time.time()
    while len(unlink) < len(link)*2:
        '''if len(unlink) % 1000 == 0:
            print(len(unlink), time.time() - t_start)'''

        row = random.randint(0, node_num - 1)
        col = random.randint(0, node_num - 1)
        if row > col:
            row, col = col, row
        edge_str = ",".join([str(row), str(col)])
        try:
            g0, g1 = gender[row], gender[col]
            if g0 == min_gender and g1 == min_gender:
                tmpg = 1
            elif g0 == maj_gender and g1 == maj_gender:
                tmpg = 2
            else:
                tmpg = 0
        except:
            tmpg = -1
        if (row != col) and (edge_str not in existing_set):
            unlink.append([row, col])
            g_unlink.append(tmpg)
            existing_set.add(edge_str)
    return link, unlink, g_link, g_unlink


def same_label_links(labels, gender, links, g):
    links = np.array(links)
    gender = np.array(gender)
    links_gender = links[gender == g]
    labels_1 = labels[links_gender[:, 0]]
    labels_2 = labels[links_gender[:, 1]]
    same_label_links = links_gender[labels_1 == labels_2]
    return same_label_links


def densities(g_link, gender, node_num):
    density_all = len(g_link) / node_num/ (node_num - 1) *2
    gender = np.array(gender)
    g_link = np.array(g_link)
    num_1 = (gender == 1).sum()
    num_2 = (gender == 2).sum()
    #0
    link_0 = (g_link == 0).sum()
    density_0 = link_0/num_1/num_2
    link_1 = (g_link==1).sum()
    density_1 = link_1/(num_1*(num_1-1)/2)

    link_2 = (g_link==2).sum()
    density_2 = link_2/(num_2*(num_2-1)/2)
    return density_all, density_0, density_1, density_2



if __name__ == "__main__":
    datapath = "dataset/"
    dataset = "pokec"
    ego_user = "107"
    g_to_ptb = 2
    change_ratio = 4.0
    adj, ft, gender, labels = load_data(datapath, dataset, ego_user, dropout=0)
    try:
        adj = adj.to_dense()
    except:
        print("Adj is not sparse, no more process")


    node_2_ind = np.array(gender == 2)
    node_num = adj.shape[1]
    link, unlink, g_link, g_unlink = get_link(adj, node_num, gender)
    same_label_link = same_label_links(labels, g_link, link, g=g_to_ptb)
    same_label_unlinks = same_label_links(labels, g_unlink, unlink, g=g_to_ptb)
    link_total = (np.array(g_link) == g_to_ptb).sum()
    sample_ind = np.random.choice(len(same_label_unlinks), size=int(change_ratio * link_total))
    g_unlinks_sample = same_label_unlinks[sample_ind]
    for ul in g_unlinks_sample:
        id1, id2 = ul[0], ul[1]
        adj[id1, id2] = 1
        adj[id2, id1] = 1

    link_new, unlink_new, g_link_new, g_unlink_new = get_link(adj, node_num, gender)
    num_node_to_ptb = (gender == g_to_ptb).sum()
    density_old = (np.array(g_link) == g_to_ptb).sum() / (num_node_to_ptb * (num_node_to_ptb - 1) / 2)
    density_new = (np.array(g_link_new)==g_to_ptb).sum()/(num_node_to_ptb * (num_node_to_ptb - 1) / 2)
    da, d0, d1, d2 = densities(g_link_new, gender, node_num)
    print("density for all:{:.4f}\n"
          "density for group 1:{:.4f}\n"
          "density for group 2:{:.4f}\n"
          "density for group 0:{:.4f}".format(da, d1, d2, d0))


    file_name = "ind.{}.adj".format(dataset)
    file_loc = "ptb_with_density/r={}".format(change_ratio)
    if not os.path.exists(file_loc):
        os.makedirs(file_loc)
    adj = torch.FloatTensor(adj)
    with open('/'.join([file_loc, file_name]), 'wb') as f:
        pkl.dump(adj, f)



    pass