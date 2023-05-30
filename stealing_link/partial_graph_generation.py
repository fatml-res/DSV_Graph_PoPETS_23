import os

import numpy as np
import pandas as pd
import torch
from stealing_link.utils import *
import pickle as pkl
import json
import random
import time
from attack import operator_func, get_metrics
from scipy.spatial.distance import cosine, euclidean, correlation, chebyshev,\
    braycurtis, canberra, cityblock, sqeuclidean
from sklearn.model_selection import train_test_split


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
    while len(unlink) < len(link):
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


def top_k_post(post, k):
    p = np.array(post) + min(post)
    k = min(len(post), k)
    low_inds = np.argsort(post)[:-k]
    p[low_inds] = 0
    return list(p)


def generate_train_test(link, unlink, dense_pred, gcn_pred, train_ratio,
                        feature_arr, dataset, saving_path="GAT/partial/",
                        g_link=[], g_unlink=[]):
    train_len = max(len(link) * train_ratio, 1)
    groups, link_groups_lens = np.unique(g_link, return_counts=True)
    if min(link_groups_lens) < train_len//3:
        print("Training rate is too large for fair sampling")
        print("The minimum group size is only {} compared with required {}".format(min(link_groups_lens),
                                                                                       train_len//3))
        print("Apply inplace sampling")

    diff_list = []
    df_columns = []
    for member in range(2):
        link_or_unlink = link if not member else unlink
        g_link_or_unlink = g_link if not member else g_unlink
        ind_link = np.array(range(len(link_or_unlink)))
        for g in range(3):
            print("working on group {}-{}".format(g, "member" if not member else "non-member"))
            ind_current_group = ind_link[np.array(g_link_or_unlink) == g]
            if len(ind_current_group) < train_len // 3:
                ind_sample_g_test, ind_sample_g_train = train_test_split(ind_current_group, test_size=0.5)
                ind_sample_g_test = np.random.choice(ind_sample_g_test, int(train_len // 3))
                ind_sample_g_train = np.random.choice(ind_sample_g_train, int(train_len // 3))
            else:
                ind_sample_g_test, ind_sample_g_train = train_test_split(ind_current_group,
                                                                         test_size=int(train_len // 3))
            ind_sample_g = np.concatenate([ind_sample_g_train, ind_sample_g_test])
            labels = np.ones([len(ind_sample_g), 1]) * (1 - member)
            gcn0 = np.array(gcn_pred)[np.array(link_or_unlink)[ind_sample_g, 0]]
            gcn1 = np.array(gcn_pred)[np.array(link_or_unlink)[ind_sample_g, 1]]
            dense0 = np.array(dense_pred)[np.array(link_or_unlink)[ind_sample_g, 0]]
            dense1 = np.array(dense_pred)[np.array(link_or_unlink)[ind_sample_g, 1]]
            feat0 = np.array(feature_arr)[np.array(link_or_unlink)[ind_sample_g, 0]]
            feat1 = np.array(feature_arr)[np.array(link_or_unlink)[ind_sample_g, 1]]
            nodes = np.array(link_or_unlink)[ind_sample_g]
            group = np.ones([len(ind_sample_g), 1]) * g
            train_test = np.vstack([np.ones([len(ind_sample_g_train), 1]), np.zeros([len(ind_sample_g_test), 1])])

            link_content_list = [labels, gcn0, gcn1, dense0, dense1, feat0, feat1, nodes, group, train_test]
            content_names = ['label',
                             'gcn_pred0',
                             'gcn_pred1',
                             "dense_pred0",
                             "dense_pred1",
                             "feature_arr0",
                             "feature_arr1",
                             "id_node",
                             "gender_group",
                             "train_test"]
            if len(df_columns) == 0:
                for i in range(len(content_names)):
                    len_content = link_content_list[i].shape[1]
                    if len_content > 1:
                        df_columns += ['{}_{}'.format(content_names[i], x) for x in range(len_content)]
                    else:
                        df_columns.append(content_names[i])

            diff = get_diff(link_content_list)
            print("generated {} training candidates and {} testing candidates".format(
                np.unique(diff[:, -1], return_counts=True)[1][1],
                np.unique(diff[:, -1], return_counts=True)[1][0]))

            diff_list.append(diff)

    diff_df = np.vstack(diff_list)
    diff_df = pd.DataFrame(diff_df)
    diff_file = saving_path + "diff_%s_train_ratio_%0.1f_train_fair.csv" % (dataset, train_ratio)
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    diff_df.to_csv(diff_file, index=False, header=False)


def get_diff(link_contents):
    similarity_list = [cosine, euclidean, correlation, chebyshev,
                       braycurtis, canberra, cityblock, sqeuclidean]
    # Posterior aggregation, used in attack 3 and 6
    member = link_contents[0]
    t0 = np.array(link_contents[1])
    r0 = np.array(link_contents[3])
    f0 = np.array(link_contents[5])
    t1 = np.array(link_contents[2])
    r1 = np.array(link_contents[4])
    f1 = np.array(link_contents[6])
    gender = link_contents[-2]
    train_test = link_contents[-1]
    ids = link_contents[7]
    post_agg_vec = np.hstack([operator_func("average", np.array(t0), np.array(t1)),
                              operator_func("hadamard", np.array(t0), np.array(t1)),
                              operator_func("weighted_l1", np.array(t0), np.array(t1)),
                              operator_func("weighted_l2", np.array(t0), np.array(t1))]) # 16/8

    # Posterior similarity, posterior entropy similarity, used in all attacks
    target_similarity = np.array([[sim(t0[i], t1[i]) for sim in similarity_list] for i in range(len(t0))]) #8
    target_metric_vec = np.array([get_metrics(t0[i] - t0[i].min(),
                                     t1[i] - t1[i].min(),
                                    'entropy', operator_func) for i in range(len(t0))]) #4

    # Feature related, reference post similarity, used in attack 5, 6, 7
    reference_similarity = np.array([[sim(r0[i], r1[i]) for sim in similarity_list] for i in range(len(r0))])  # 8
    feature_similarity = np.array([[sim(f0[i], f1[i]) for sim in similarity_list] for i in range(len(f0))])  # 8
    reference_metric_vec = np.array([get_metrics(r0[i] - r0[i].min(),
                                     r1[i] - r1[i].min(),
                                    'entropy', operator_func) for i in range(len(r0))]) # 4
    return np.hstack([post_agg_vec, target_similarity,
                      target_metric_vec,reference_similarity,
                      feature_similarity, reference_metric_vec,
                      ids, member, gender, train_test])


def get_partial(adj, model_type, datapath, partial_path, pred_path, dataset, t=0):
    _, features, _, _, _, _, _, _ = load_data(datapath, dataset)
    if isinstance(features, np.ndarray):
        feature_arr = features
    else:
        feature_arr = features.numpy()
    feature_arr = feature_arr.tolist()

    dense_pred = pkl.loads(open("dense/" + "%s_dense_pred.pkl" % dataset, "rb").read())
    try:
        gat_pred = pkl.loads(open(pred_path + "/{}_{}_pred.pkl".format(dataset, model_type), "rb").read())
    except:
        if "ep" in partial_path:
            pred_file = "{}/DP_con/{}_{}_pred_{}.pkl".format(model_type, dataset, model_type, pred_path.split('/')[-1])
            gat_pred = pkl.loads(open(pred_file, "rb").read())
        else:
            print("Target model output not found")
            return

    try:
        gender = pkl.loads(open(datapath + '/ind.{}.gender'.format(dataset), "rb").read())
    except:
        gender = - np.ones(len(gat_pred))

    dense_pred = dense_pred.tolist()
    gat_pred = gat_pred.tolist()

    node_num = len(dense_pred)
    link, unlink, g_link, g_unlink = get_link(adj, node_num, gender)
    label = []
    for row in link:
        label.append(1)
    for row in unlink:
        label.append(0)

    # generate 10% to 100% of known edges
    t_start = time.time()
    saving_path = partial_path + "partial/t={}/".format(t)
    print("generating: 20 percent", time.time() - t_start)
    generate_train_test(link, unlink, dense_pred, gat_pred, 0.2,
                               feature_arr, dataset, saving_path=saving_path,
                               g_link=g_link, g_unlink=g_unlink)


if __name__ == "__main__":
    get_partial()

