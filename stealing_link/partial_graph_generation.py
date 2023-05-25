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
                        g_link=[], g_unlink=[], fair_sample=False, topk=-1):
    train = []
    test = []
    train_len = len(link) * train_ratio
    ind_link = np.arange(len(link))
    ind_unlink = np.arange(len(unlink))
    np.random.shuffle(ind_link)
    np.random.shuffle(ind_unlink)
    if fair_sample:
        groups, link_groups_lens = np.unique(g_link, return_counts=True)
        if min(link_groups_lens) < train_len//3:
            print("Training rate is too large for fair sampling")
            print("Training set is reduced from {} to {}".format(train_len, min(link_groups_lens)//2 * 3))
            train_len = min(link_groups_lens)//2 * 3

    count_g = [0, 0, 0]
    diff_list = []
    # links
    for i in ind_link:

        link_id0 = link[i][0]
        link_id1 = link[i][1]
        gi = g_link[i]
        line_link = {
            'label': 1,
            'gcn_pred0': gcn_pred[link_id0] if not topk else top_k_post(gcn_pred[link_id0], topk),
            'gcn_pred1': gcn_pred[link_id1] if not topk else top_k_post(gcn_pred[link_id1], topk),
            "dense_pred0": dense_pred[link_id0],
            "dense_pred1": dense_pred[link_id1],
            "feature_arr0": feature_arr[link_id0],
            "feature_arr1": feature_arr[link_id1],
            "id_pair": [int(link_id0), int(link_id1)],
            "gender_group": gi
            }
        diff = get_diff(line_link, 1, gender=gi)
        diff_list.append(diff)



        if fair_sample:
            if count_g[gi] < train_len//3:
                train.append(line_link)
                count_g[gi] += 1
            else:
                test.append(line_link)
        else:
            if len(train) < train_len:
                train.append(line_link)
            else:
                test.append(line_link)

    # unlinks
    for i in ind_unlink:
        unlink_id0 = unlink[i][0]
        unlink_id1 = unlink[i][1]
        gi = g_unlink[i]

        line_unlink = {
            'label': 0,
            'gcn_pred0': gcn_pred[unlink_id0] if not topk else top_k_post(gcn_pred[unlink_id0], topk),
            'gcn_pred1': gcn_pred[unlink_id1] if not topk else top_k_post(gcn_pred[unlink_id1], topk),
            "dense_pred0": dense_pred[unlink_id0],
            "dense_pred1": dense_pred[unlink_id1],
            "feature_arr0": feature_arr[unlink_id0],
            "feature_arr1": feature_arr[unlink_id1],
            "id_pair": [int(unlink_id0), int(unlink_id1)],
            "gender_group": gi
        }
        diff = get_diff(line_unlink, 0,  gender=gi)
        diff_list.append(diff)
        if fair_sample:
            if count_g[gi] < 2 * train_len // 3:
                train.append(line_unlink)
                count_g[gi] += 1
            else:
                test.append(line_unlink)
        else:
            if len(train) < 2 * train_len:
                train.append(line_unlink)
            else:
                test.append(line_unlink)
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    diff_list = pd.DataFrame(np.array(diff_list))

    if fair_sample:
        diff_file = saving_path + "diff_%s_train_ratio_%0.1f_train_fair.csv" % (dataset, train_ratio)
        partial_train_file = saving_path + "%s_train_ratio_%0.1f_train_fair.json" % (dataset, train_ratio)
        partial_test_file = saving_path + "%s_train_ratio_%0.1f_test_fair.json" % (dataset, train_ratio)
    else:
        diff_file = saving_path + "diff_%s_train_ratio_%0.1f_train.csv" % (dataset, train_ratio)
        partial_train_file = saving_path + "%s_train_ratio_%0.1f_train.json" % (dataset, train_ratio)
        partial_test_file = saving_path + "%s_train_ratio_%0.1f_test.json" % (dataset, train_ratio)
    diff_list.to_csv(diff_file, index=False, header=False)
    with open(partial_train_file, "w") as wf1, open(partial_test_file, "w") as wf2:
        for row in train:
            wf1.write("%s\n" % json.dumps(row))
        for row in test:
            wf2.write("%s\n" % json.dumps(row))


def generate_train_test_v2(link, unlink, dense_pred, gcn_pred, train_ratio,
                        feature_arr, dataset, saving_path="GAT/partial/",
                        g_link=[], g_unlink=[], fair_sample=False, topk=-1):
    train = []
    test = []
    train_len = len(link) * train_ratio
    ind_link = np.arange(len(link))
    ind_unlink = np.arange(len(unlink))
    np.random.shuffle(ind_link)
    np.random.shuffle(ind_unlink)
    if fair_sample:
        groups, link_groups_lens = np.unique(g_link, return_counts=True)
        if min(link_groups_lens) < train_len//3:
            print("Training rate is too large for fair sampling")
            print("Training set is reduced from {} to {}".format(train_len, min(link_groups_lens)//2 * 3))
            train_len = min(link_groups_lens)//2 * 3

    count_g = [0, 0, 0]
    diff_list = []
    # links
    df_columns = []
    for i in ind_link:

        link_id0 = link[i][0]
        link_id1 = link[i][1]
        gi = g_link[i]
        link_contents = [1,
                         gcn_pred[link_id0] if not topk else top_k_post(gcn_pred[link_id0], topk),
                         gcn_pred[link_id1] if not topk else top_k_post(gcn_pred[link_id1], topk),
                         dense_pred[link_id0],
                         dense_pred[link_id1],
                         feature_arr[link_id0],
                         feature_arr[link_id1],
                         int(link_id0),
                         int(link_id1),
                         gi]
        line_row = np.concatenate(link_contents, axis=None)
        content_names = ['label',
                         'gcn_pred0',
                         'gcn_pred1',
                         "dense_pred0",
                         "dense_pred1",
                         "feature_arr0",
                         "feature_arr1",
                         "id_node0",
                         "id_node1",
                         "gender_group"]
        if len(df_columns) == 0:
            for i in range(len(content_names)):
                try:
                    len_content = len(link_contents[i])
                    df_columns += ['{}_{}'.format(content_names[i], x) for x in range(len_content)]
                except:
                    df_columns.append(content_names[i])

        diff = get_diff_v2(link_contents, 1, gender=gi)
        diff_list.append(diff)

        if fair_sample:
            if count_g[gi] < train_len//3:
                train.append(line_row)
                count_g[gi] += 1
            else:
                test.append(line_row)
        else:
            if len(train) < train_len:
                train.append(line_row)
            else:
                test.append(line_row)

    # unlinks
    for i in ind_unlink:
        unlink_id0 = unlink[i][0]
        unlink_id1 = unlink[i][1]
        gi = g_unlink[i]

        link_contents = [0,
                         gcn_pred[unlink_id0] if not topk else top_k_post(gcn_pred[unlink_id0], topk),
                         gcn_pred[unlink_id1] if not topk else top_k_post(gcn_pred[unlink_id1], topk),
                         dense_pred[unlink_id0],
                         dense_pred[unlink_id1],
                         feature_arr[unlink_id0],
                         feature_arr[unlink_id1],
                         int(unlink_id0),
                         int(unlink_id1),
                         gi]
        line_row = np.concatenate(link_contents, axis=None)
        diff = get_diff_v2(link_contents, 0,  gender=gi)
        diff_list.append(diff)
        if fair_sample:
            if count_g[gi] < 2 * train_len // 3:
                train.append(line_row)
                count_g[gi] += 1
            else:
                test.append(line_row)
        else:
            if len(train) < 2 * train_len:
                train.append(line_row)
            else:
                test.append(line_row)
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    diff_list = pd.DataFrame(np.array(diff_list))
    train_df = pd.DataFrame(np.array(train), columns=df_columns)
    test_df = pd.DataFrame(np.array(test), columns=df_columns)

    if fair_sample:
        diff_file = saving_path + "diff_%s_train_ratio_%0.1f_train_fair.csv" % (dataset, train_ratio)
        partial_train_file = saving_path + "%s_train_ratio_%0.1f_train_fair.csv" % (dataset, train_ratio)
        partial_test_file = saving_path + "%s_train_ratio_%0.1f_test_fair.csv" % (dataset, train_ratio)
    else:
        diff_file = saving_path + "diff_%s_train_ratio_%0.1f_train.csv" % (dataset, train_ratio)
        partial_train_file = saving_path + "%s_train_ratio_%0.1f_train.csv" % (dataset, train_ratio)
        partial_test_file = saving_path + "%s_train_ratio_%0.1f_test.csv" % (dataset, train_ratio)
    diff_list.to_csv(diff_file, index=False, header=False)
    train_df.to_csv(partial_train_file, index=False)
    test_df.to_csv(partial_test_file, index=False)


def generate_train_test_v3(link, unlink, dense_pred, gcn_pred, train_ratio,
                        feature_arr, dataset, saving_path="GAT/partial/",
                        g_link=[], g_unlink=[], fair_sample=False, topk=-1):
    train_len = max(len(link) * train_ratio, 1)
    if fair_sample:
        groups, link_groups_lens = np.unique(g_link, return_counts=True)
        if min(link_groups_lens) < train_len//3:
            print("Training rate is too large for fair sampling")
            print("The minimum group size is only {} compared with required {}".format(min(link_groups_lens),
                                                                                       train_len//3))
            print("Apply inplace sampling")

    diff_list = []
    df_columns = []
    if fair_sample:
        # fair sample option:
        for member in range(2):
            link_or_unlink = link if not member else unlink
            g_link_or_unlink = g_link if not member else g_unlink
            ind_link = np.array(range(len(link_or_unlink)))
            for g in range(3):
                print("working on group {}-{}".format(g, "member" if not member else "non-member"))
                ind_current_group = ind_link[np.array(g_link_or_unlink) == g]
                if len(ind_current_group) < train_len//3:
                    ind_sample_g_test, ind_sample_g_train = train_test_split(ind_current_group, test_size=0.5)
                    ind_sample_g_test = np.random.choice(ind_sample_g_test, int(train_len//3))
                    ind_sample_g_train = np.random.choice(ind_sample_g_train, int(train_len//3))
                else:
                    ind_sample_g_test, ind_sample_g_train = train_test_split(ind_current_group, test_size=int(train_len//3))
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

                diff = get_diff_v3(link_content_list)
                print("generated {} training candidates and {} testing candidates".format(np.unique(diff[:, -1], return_counts=True)[1][1],
                                                                                          np.unique(diff[:, -1], return_counts=True)[1][0]))

                diff_list.append(diff)
    else:
        # random sample option:
        for member in range(2):
            link_or_unlink = link if not member else unlink
            g_link_or_unlink = g_link if not member else g_unlink
            if isinstance(g_link_or_unlink, list):
                g_link_or_unlink = np.array(g_link_or_unlink).reshape(-1, 1)
            ind_link = np.array(range(len(link_or_unlink)))
            # random select 20% member as training set, else testing set
            print("working on group {}".format("member" if not member else "non-member"))
            ind_sample_test, ind_sample_train = train_test_split(ind_link, test_size=int(train_len))
            ind_sample = np.concatenate([ind_sample_train, ind_sample_test])
            labels = np.ones([len(ind_sample), 1]) * (1 - member)
            gcn0 = np.array(gcn_pred)[np.array(link_or_unlink)[ind_sample, 0]]
            gcn1 = np.array(gcn_pred)[np.array(link_or_unlink)[ind_sample, 1]]
            dense0 = np.array(dense_pred)[np.array(link_or_unlink)[ind_sample, 0]]
            dense1 = np.array(dense_pred)[np.array(link_or_unlink)[ind_sample, 1]]
            feat0 = np.array(feature_arr)[np.array(link_or_unlink)[ind_sample, 0]]
            feat1 = np.array(feature_arr)[np.array(link_or_unlink)[ind_sample, 1]]
            nodes = np.array(link_or_unlink)[ind_sample]
            group = g_link_or_unlink[ind_sample]
            train_test = np.vstack([np.ones([len(ind_sample_train), 1]), np.zeros([len(ind_sample_test), 1])])
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

            diff = get_diff_v3(link_content_list)
            print("generated {} training candidates and {} testing candidates".format(
                np.unique(diff[:, -1], return_counts=True)[1][1],
                np.unique(diff[:, -1], return_counts=True)[1][0]))
            diff_list.append(diff)
    diff_df = np.vstack(diff_list)
    diff_df = pd.DataFrame(diff_df)
    if fair_sample:
        diff_file = saving_path + "diff_%s_train_ratio_%0.1f_train_fair.csv" % (dataset, train_ratio)
    else:
        diff_file = saving_path + "diff_%s_train_ratio_%0.1f_train.csv" % (dataset, train_ratio)

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    diff_df.to_csv(diff_file, index=False, header=False)


def get_diff(line_link, member=0, gender=0):
    similarity_list = [cosine, euclidean, correlation, chebyshev,
                       braycurtis, canberra, cityblock, sqeuclidean]
    # Posterior aggregation, used in attack 3 and 6
    t0 = np.array(line_link["gcn_pred0"])
    r0 = np.array(line_link["dense_pred0"])
    f0 = np.array(line_link["feature_arr0"])
    t1 = np.array(line_link["gcn_pred1"])
    r1 = np.array(line_link["dense_pred1"])
    f1 = np.array(line_link["feature_arr1"])
    post_agg_vec = operator_func("concate_all", np.array(t0), np.array(t1))#16

    # Posterior similarity, posterior entropy similarity, used in all attacks
    target_similarity = np.array([row(t0, t1) for row in similarity_list]) #8
    target_metric_vec = get_metrics(t0 - min(t0),
                                    t1 - min(t1),
                                    'entropy', operator_func) #4

    # Feature related, reference post similarity, used in attack 5, 6, 7
    reference_similarity = np.array([row(r0, r1) for row in similarity_list]) # 8
    feature_similarity = np.array([row(f0, f1) for row in similarity_list]) # 8
    reference_metric_vec = get_metrics(np.array(r0) - min(r0),
                                       np.array(r1) - min(r1),
                                       'entropy', operator_func) # 4
    return np.concatenate([post_agg_vec,
                           target_similarity, target_metric_vec,
                           reference_similarity, feature_similarity, reference_metric_vec, [member, gender]])


def get_diff_v2(link_contents, member=0, gender=0):
    similarity_list = [cosine, euclidean, correlation, chebyshev,
                       braycurtis, canberra, cityblock, sqeuclidean]
    # Posterior aggregation, used in attack 3 and 6
    t0 = np.array(link_contents[1])
    r0 = np.array(link_contents[3])
    f0 = np.array(link_contents[5])
    t1 = np.array(link_contents[2])
    r1 = np.array(link_contents[4])
    f1 = np.array(link_contents[6])
    post_agg_vec = operator_func("concate_all", np.array(t0), np.array(t1))#16

    # Posterior similarity, posterior entropy similarity, used in all attacks
    target_similarity = np.array([row(t0, t1) for row in similarity_list]) #8
    target_metric_vec = get_metrics(t0 - min(t0),
                                    t1 - min(t1),
                                    'entropy', operator_func) #4

    # Feature related, reference post similarity, used in attack 5, 6, 7
    reference_similarity = np.array([row(r0, r1) for row in similarity_list]) # 8
    feature_similarity = np.array([row(f0, f1) for row in similarity_list]) # 8
    reference_metric_vec = get_metrics(np.array(r0) - min(r0),
                                       np.array(r1) - min(r1),
                                       'entropy', operator_func) # 4
    return np.concatenate([post_agg_vec,
                           target_similarity, target_metric_vec,
                           reference_similarity, feature_similarity, reference_metric_vec, [member, gender]])


def get_diff_v3(link_contents):
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


def get_partial(adj, model_type, datapath, partial_path, pred_path, dataset, fair_sample=False, t=0, ptb=False, top_k=-1):
    _, features, _, _, _, _, _, _ = load_data(datapath, dataset, ptb)
    if isinstance(features, np.ndarray):
        feature_arr = features
    else:
        feature_arr = features.numpy()
    feature_arr = feature_arr.tolist()
    dataset = dataset.split("_ptb")[0]

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
    for i in [2]:
        if top_k > 0:
            saving_path = partial_path + "partial/t={}/K={}/".format(t, top_k)
        else:
            saving_path = partial_path + "partial/t={}/".format(t)
        print("generating: %d percent" % (i * 10), time.time() - t_start)
        generate_train_test_v3(link, unlink, dense_pred, gat_pred, i / 10.0,
                               feature_arr, dataset, saving_path=saving_path,
                               g_link=g_link, g_unlink=g_unlink, fair_sample=fair_sample, topk=top_k)


if __name__ == "__main__":
    get_partial()

