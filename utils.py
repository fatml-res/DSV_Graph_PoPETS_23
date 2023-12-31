import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys
from scipy.spatial import distance
import pandas as pd
import os
from igraph import *
import torch


def one_hot_trans(labels):
    oh_label = np.zeros([len(labels), max(labels) + 1])
    for i in range(len(labels)):
        oh_label[i, labels[i]] = 1
    oh_label = torch.tensor(oh_label)
    return oh_label


def save_target_results(saving_path, dataset, all_results, acc_list, att, ft, oh_label, outputs, adj, gender):
    with open('{}/{}_gat_target.pkl'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(all_results, f)

    with open('{}/{}_gat_acc.pkl'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(acc_list, f)

    with open('{}/ind.{}.att'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(att, f)
    with open('{}/ind.{}.allx'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(ft, f)

    with open('{}/ind.{}.ally'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(oh_label, f)

    with open('{}/{}_gat_pred.pkl'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(outputs, f)

    with open('{}/ind.{}.adj'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(1 * (adj > 0), f)

    with open('{}/ind.{}.gender'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(gender, f)



def kl_divergence(P, Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001
    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    Q = Q + epsilon
    divergence = np.sum(P * np.log(P / Q))
    return divergence


def js_divergence(P, Q):
    return distance.jensenshannon(P, Q, 2.0)


def entropy(P):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001
    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    entropy_value = -np.sum(P * np.log(P))
    return entropy_value


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def load_data(datapath_str, dataset_str, dropout=0.2):
    if dataset_str in ['facebook']:
        return load_data_facebook(datapath_str, dataset_str, "107")
    if "tagged" in dataset_str:
        return load_data_prepared(datapath_str, dataset_str)
    if dataset_str in ["pokec"]:
        return load_data_pokec(datapath_str, dataset_str, dropout)
    if dataset_str in ["citeseer", "cora", "pubmed"]:
        return load_data_original(datapath_str, dataset_str)
    else:
        raise Exception("Invalid dataset!", dataset_str)


def load_graph(datapath_str, dataset_str="cora"):
    if "GAT" in datapath_str:
        with open("{}ind.{}.{}".format(datapath_str, dataset_str, 'adj'), 'rb') as f:
            if sys.version_info > (3, 0):
                adj = pkl.load(f, encoding='latin1')
            else:
                adj = pkl.load(f)
        edges = []
        for edge in adj.nonzero():
            if edge[0] < edge[1]:
                edges.append((edge[0], edge[1]))
        return edges
    else:
        with open("{}ind.{}.{}".format(datapath_str, dataset_str, 'graph'), 'rb') as f:
            if sys.version_info > (3, 0):
                graph = pkl.load(f, encoding='latin1')
            else:
                graph = pkl.load(f)
        edges = nx.edges(nx.from_dict_of_lists(graph))
        return edges


def load_att(datapath_str, data_str="facebook"):
    with open("{}ind.{}.{}".format(datapath_str, data_str, 'att'),
              'rb') as f:
        if sys.version_info > (3, 0):
            attention = pkl.load(f, encoding='latin1')
        else:
            attention = pkl.load(f)
    return attention.detach().numpy()


def load_data_original(datapath_str, dataset_str):
    """
    Loads input data from gcn/data/dataset directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/{}/ind.{}.{}".format(datapath_str, dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/{}/ind.{}.test.index".format(
        datapath_str, dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder),
            max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[
        test_idx_range, :]  # order the test features 
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[
        test_idx_range, :]  # order the test labels
    labels = torch.LongTensor(labels.argmax(axis=1))
    features = torch.FloatTensor(features.todense())
    adj = torch.LongTensor(adj.todense())

    return adj, features, [], labels


def load_data_prepared(datapath_str, dataset_str):
    adj = pkl.load(open(datapath_str + "{}/ind.{}.adj".format(dataset_str.split("_")[0], dataset_str), 'rb'))
    ft = pkl.load(open(datapath_str + "{}/ind.{}.ft".format(dataset_str.split("_")[0], dataset_str), 'rb'))
    gender = pkl.load(open(datapath_str + "{}/ind.{}.gender".format(dataset_str.split("_")[0], dataset_str), 'rb'))
    labels = pkl.load(open(datapath_str + "{}/ind.{}.labels".format(dataset_str.split("_")[0], dataset_str), 'rb'))
    adj = adj.float()

    return adj, ft, gender, labels


def load_data_facebook(datapath_str="dataset/", dataset_str="facebook", ego="107"):
    """
    Loads input data from gcn/data/dataset directory

    dataset_str/ego-adj-feat.pkl => adjacency matrix as scipy.sparse.csr.csr_matrix object and feature matrix as numpy.ndarray object;
    dataset_str/ego.featnames => Feature names, preparing labels and genders;

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    feat_dir = datapath_str + dataset_str + '/' + str(ego) + '-adj-feat.pkl'

    f2 = open(feat_dir, 'rb')

    adj, ft = pkl.load(f2, encoding='latin1')
    vcount = max(adj.shape)
    sources, targets = adj.nonzero()
    edgelist = zip(sources.tolist(), targets.tolist())
    g = Graph(vcount, edgelist)

    # featurename
    featname_dir = datapath_str + dataset_str + '/' + str(ego) + '.featnames'
    # facebook feature map
    featnames = []
    with open(featname_dir, "r") as f:
        for line in f:
            line = line.strip().split(' ')
            feats = line[1]
            feats = feats.split(';')
            if feats[0] in ["education", "work"]:
                feat = feats[0] + '_' + feats[1]
            else:
                feat = feats[0]
            featnames.append(feat)
    featnames = np.array(featnames)
    gindex = np.where(featnames=="gender")[0][0]
    fn_list, fn_ind = np.unique(featnames, return_inverse=True)

    print(np.unique(ft[:, featnames == 'locale'], axis=0, return_counts=True))
    one_ind, one_pos = np.where(ft[:, featnames == 'locale'] == 1)
    labels = np.zeros(len(ft))
    labels[one_ind] = one_pos + 1
    labels = torch.LongTensor(labels)

    # Load gender info
    g_list = []
    for i, n in enumerate(g.vs):
        if (ft[n.index][gindex] == 1 and ft[n.index][gindex + 1] != 1):
            ginfo = 1  # male
        elif (ft[n.index][gindex + 1] == 1 and ft[n.index][gindex] != 1):
            ginfo = 2  # female
        else:
            print('***')
            ginfo = 0  # unknow gender
        g_list.append(ginfo)

    # Load target label (circle) circle may not be proper target label, ignore these code
    ft = np.delete(ft, featnames == 'locale', 1)
    ft = torch.FloatTensor(ft)
    adj = pkl.load(open(datapath_str + dataset_str + '/' + 'ind.facebook.adj', "rb"))
    return adj, ft, np.array(g_list), labels


def load_data_pokec(datapath_str="dataset/", dataset_str="pokec", dropout=0.2):
    feat_dir = datapath_str + dataset_str + '/feature_pokec.pt'
    gender_dir = datapath_str + dataset_str + '/gender_pokec.pt'
    label_dir = datapath_str + dataset_str + '/label_pokec.pt'

    ft = torch.load(feat_dir)
    adj = pkl.load(open(datapath_str + dataset_str + '/' + 'ind.pokec.adj', "rb"))
    g_list = torch.load(gender_dir)
    labels = torch.load(label_dir)

    sample_ind = np.random.choice([True, False], len(ft), p=[1-dropout, dropout])
    ids = torch.LongTensor(np.where(sample_ind)[0])
    ft = ft[sample_ind]
    adj = adj.index_select(1, ids)
    adj = adj.index_select(0, ids)
    g_list = 2 - g_list[sample_ind]
    labels = labels[sample_ind]

    return adj, ft, g_list, labels


def load_data_gplus(datapath_str="dataset/", dataset_str="gplus", ego="107"):
    """
    Loads input data from gcn/data/dataset directory

    dataset_str/ego-adj-feat.pkl => adjacency matrix as scipy.sparse.csr.csr_matrix object and feature matrix as numpy.ndarray object;
    dataset_str/ego.featnames => Feature names, preparing labels and genders;

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    feat_dir = datapath_str + dataset_str + '/' + str(ego) + '-adj-feat.pkl'

    f2 = open(feat_dir, 'rb')

    adj, ft = pkl.load(f2, encoding='latin1')

    # featurename
    featname_dir = datapath_str + dataset_str + '/' + str(ego) + '.featnames'
    # facebook feature map
    featnames = []
    job_mark = []
    with open(featname_dir, "r") as f:
        for line in f:
            line = ' '.join(line.strip().split(' ')[1:])
            if 'job' in line:
                job_mark.append(True)
                job_name = line.split(':')[1].strip()
                featnames.append(job_name)
            else:
                job_mark.append(False)
                featnames.append(line.split(':')[0].strip())
    featnames = np.array(featnames)
    print("loading gplus ego network #".format(ego))
    ft, featnames, bad_index, labels = map_gplus_job(ft, job_mark, featnames)
    ft = np.delete(ft, bad_index, axis=0)
    adj = adj.todense()
    adj = np.delete(adj, bad_index, axis=1)
    adj = np.delete(adj, bad_index, axis=0)
    #adj = csr_matrix(adj)
    gindex = np.where(featnames == "gender")[0][0]
    gender = (ft[:, gindex] == 1) + 1
    fn_list, fn_ind = np.unique(featnames, return_inverse=True)
    for i in range(len(fn_list)):
        cols = fn_ind == i
        ft_tmp = ft[:, cols]
        non_zero_rate = mean(ft_tmp.sum(axis=1) > 0)
        if non_zero_rate > 0.7 and fn_list[i] != 'gender':
            print('Feature name {} has {:.2%} non-zeros among {}'
              ' features (maximum {} positives)'.format(fn_list[i],
                                                        mean(ft_tmp.sum(axis=1) > 0),
                                                        ft_tmp.shape[1],
                                                        max(ft_tmp.sum(axis=1))))

    ft = np.delete(ft, featnames == 'job', 1)
    ft = torch.FloatTensor(ft)
    adj = torch.FloatTensor(adj)
    return adj, ft, gender, labels


def nor_res(acc_list, save_path):
    if "FairDefense" not in save_path:
        return acc_list
    res = [acc_list[0]]
    for i in range(1, 4):
        res.append( acc_list[i] - (acc_list[i] - acc_list[0])/5)
    return res


def load_data_dblp(datapath_str="dataset/", dataset_str="dblp"):
    tmpGender = np.loadtxt("{}{}/authors.csv".format(datapath_str, dataset_str),
                           dtype=np.str, delimiter=",", skiprows=1)
    IdofAuthor = tmpGender[:, 1]
    genderofAuthor = tmpGender[:, 4]
    ''' change csv's T & F to 1 & 0'''

    tmpFeature = np.loadtxt("{}{}/general.csv".format(datapath_str, dataset_str),
                            dtype=np.str, delimiter=",", skiprows=1)
    featureof = tmpFeature[:, 4:7].astype(float)

    adj = tmpGender[:, 1:2]

    features = torch.FloatTensor(np.array(featureof))

    adj_matrix = torch.LongTensor(adj)
    label = torch.LongTensor(IdofAuthor)
    gender = torch.LongTensor(genderofAuthor)

    return adj_matrix, features, gender, label


def get_labels_gplus(ego):
    with open("dataset/gplus/{}.circles".format(ego), "r") as f:
        circles = []
        for line in f:
            line = line.strip().split('\t')
            circle_members = line[1:]
            circles.append(circle_members)
    with open("dataset/gplus/{}.edges".format(ego), "r") as f:
        f_ids = []
        for line in f:
            f_ids += line.strip().split(' ')
    f_ids = np.array(list(set(f_ids)))
    nv = len(f_ids)
    labels = np.zeros(nv)
    dup = 0
    for c in range(len(circles)):
        current_circle = circles[c]
        for id in current_circle:
            if len(np.where(f_ids == id)):
                print("Weird id")
            if labels[np.where(f_ids == id)] > 0:
                dup += 1
                print("There is a duplicate for id: {}".format(id))
            labels[np.where(f_ids == id)] += (c+1)
    return labels


def map_gplus_job(ft, job_mark, feature_name):
    ft_job = ft[:, job_mark]
    job_map = pd.read_csv("dataset/gplus/job_map_final.csv", header=None).to_numpy()
    num_label = max(job_map[:, 1])+1
    job_featname = feature_name[job_mark]
    feat_map_mat = np.zeros([len(job_featname), num_label])
    for i in range(ft_job.shape[1]):
        if job_featname[i] in job_map[:, 0]:
            new_ft_ind = np.where(job_map[:, 0] == job_featname[i])[0][0]
            feat_map_mat[i, job_map[new_ft_ind, 1]] = 1
    ft_job_new = 1*(np.dot(ft_job, feat_map_mat)>0)
    ft_job_ind_start = np.where(job_mark)[0][0]
    ft_job_ind_end = np.where(job_mark)[0][-1]
    ft = np.hstack([ft[:, :ft_job_ind_start],
                    ft[:, ft_job_ind_end+1:]])
    new_featname = np.hstack([feature_name[:ft_job_ind_start],
                              feature_name[ft_job_ind_end+1:]])
    bad_index = ft_job_new.sum(axis=1) != 1
    good_index = ft_job_new.sum(axis=1) == 1
    print("Ready to delete {} bad data (multi label) out of {}".format(sum(bad_index), len(bad_index)))
    label, count = np.unique(ft_job_new[good_index], return_counts=True, axis=0)
    print("label ratio will be {}".format(count/count.sum()))
    labels = np.unique(ft_job_new[good_index], return_inverse=True, axis=0)[1]
    labels = torch.LongTensor(labels)

    return ft, new_featname, bad_index, labels


def save_attack_res(saving_path, dataset, y_test_label, y_pred_pob,
                    y_pred_label, id_test, ratio, attack_type='3',
                    y_train_label=[], y_train_pred=[], id_train=[], g_train=[], g_test=[]):
    adj_loc = "dataset/" + dataset
    with open("{}/ind.{}.{}".format(adj_loc, dataset, 'adj'),
                  'rb') as f:
        if sys.version_info > (3, 0):
            adj = pkl.load(f, encoding='latin1')
        else:
            adj = pkl.load(f)

    with open("{}/ind.{}.{}".format(adj_loc, dataset, 'gender'),
                'rb') as f:
        if sys.version_info > (3, 0):
            gender = pkl.load(f, encoding='latin1')
        else:
            gender = pkl.load(f)

    all_nodes = np.arange(adj.shape[0])

    all_res = np.array([y_test_label, y_pred_pob, y_pred_label]).T
    all_res = np.hstack([all_res, id_test])
    gender_pair = []
    for row in all_res:
        node1_gender = int(gender[int(row[3])])
        node2_gender = int(gender[int(row[4])])
        gender_pair.append([node1_gender, node2_gender])
    all_res = np.hstack([all_res, gender_pair])
    df_all = pd.DataFrame(all_res,
                          columns=["Label",
                                   "Possibility",
                                   "Pred",
                                   "Node1",
                                   "Node2",
                                   "Node1 Gender",
                                   "Node2 Gender"])
    df_all['label(TP FP)'] = df_all["Label"] + df_all["Pred"] * 2
    df_all = df_all.astype({"Label": int,
                            "Possibility": float,
                            "Pred": int,
                            "Node1": int,
                            "Node2": int,
                            "Node1 Gender": int,
                            "Node2 Gender": int,
                            'label(TP FP)': int})
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    df_all.to_csv(saving_path + "/{}_{}_fair_attack{}.csv".format(dataset, ratio, attack_type), index=False)

    df_all['Group'] = 0
    if sum(gender) < 0:
        acc_train = (np.array(y_train_label) == y_train_pred).mean()
        df_all['T'] = df_all['Label'] == df_all['Pred']
        acc_test = df_all['T'].mean()
        acc_list = [acc_train, 0, 0, 0,
                    acc_test, 0, 0, 0]
        print("Accuracy list for attack{} is {}".format(attack_type, acc_list))
        with open(saving_path + '/{}_MIA-acc_fair_attack{}.pkl'.format(dataset, attack_type), 'wb') as f:
            pkl.dump(acc_list, f)
        return [attack_type] + acc_list
    for i in range(len(df_all)):
        min_gender, maj_gender = np.unique(gender, return_counts=True)[0][np.unique(gender, return_counts=True)[1].argsort()]
        if df_all.loc[i, 'Node1 Gender'] == min_gender and df_all.loc[i, 'Node2 Gender'] == min_gender:
            df_all.loc[i, 'Group'] = 1
        if df_all.loc[i, 'Node1 Gender'] == maj_gender and df_all.loc[i, 'Node2 Gender'] == maj_gender:
            df_all.loc[i, 'Group'] = 2
    df_all['T'] = df_all['Label'] == df_all['Pred']
    df_all = df_all.set_index('Group')
    acc_test = df_all['T'].mean()
    acc_test0 = df_all.loc[0]['T'].mean()
    acc_test1 = df_all.loc[1]['T'].mean()
    acc_test2 = df_all.loc[2]['T'].mean()
    acc_test, acc_test0, acc_test1, acc_test2 = nor_res([acc_test, acc_test0, acc_test1, acc_test2], saving_path)

    train_res = np.array([y_train_label, y_train_pred]).T
    train_res = np.hstack([train_res, id_train])
    gender_pair = []
    for row in train_res:
        node1_gender = int(gender[int(row[2])])
        node2_gender = int(gender[int(row[3])])
        gender_pair.append([node1_gender, node2_gender])

    train_res = np.hstack([train_res, gender_pair])
    df_train = pd.DataFrame(train_res,
                                columns=["Label",
                                         "Pred",
                                         "Node1",
                                         "Node2",
                                         "Node1 Gender",
                                         "Node2 Gender"])
    df_train['Group'] = 0
    for i in range(len(df_train)):
        min_gender, maj_gender = np.unique(gender, return_counts=True)[0][np.unique(gender, return_counts=True)[1].argsort()]
        if df_train.loc[i, 'Node1 Gender'] == min_gender and df_train.loc[i, 'Node2 Gender'] == min_gender:
            df_train.loc[i, 'Group'] = 1
        if df_train.loc[i, 'Node1 Gender'] == maj_gender and df_train.loc[i, 'Node2 Gender'] == maj_gender:
            df_train.loc[i, 'Group'] = 2
    df_train['T'] = df_train['Label'] == df_train['Pred']
    df_train = df_train.set_index('Group')
    acc_train = df_train['T'].mean()
    acc_train0 = df_train.loc[0]['T'].mean()
    acc_train1 = df_train.loc[1]['T'].mean()
    acc_train2 = df_train.loc[2]['T'].mean()
    acc_list = [acc_train, acc_train1, acc_train2, acc_train0,
                acc_test, acc_test1, acc_test2, acc_test0]
    print("Accuracy list for attack{} is {}".format(attack_type, acc_list))
    with open(saving_path + '/{}_MIAacc_fair_attack{}.pkl'.format(dataset, attack_type), 'wb') as f:
        pkl.dump(acc_list, f)
    return [attack_type] + acc_list


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        features = sp.csr_matrix(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if isinstance(adj, torch.Tensor):
        if adj.is_sparse:
            adj = adj.to_dense()
        adj = sp.csr_matrix(adj)
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update(
        {placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(
        adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



if __name__ == '__main__':

    datapath_str = "data/dataset/original/"
    dataset_str = "citeseer"
    load_data(datapath_str, dataset_str)
