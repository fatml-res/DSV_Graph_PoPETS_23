from __future__ import print_function

import os.path

import pandas as pd
import tensorflow.keras
from keras.models import Sequential
from stealing_link.keras_utils import *
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, f1_score
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine, euclidean, correlation, chebyshev,\
    braycurtis, canberra, cityblock, sqeuclidean
from utils import kl_divergence, js_divergence, entropy, save_attack_res
import argparse
import pickle as pkl
import sys
from tqdm import tqdm

batch_size = 128
num_classes = 2
epochs = 50
tf.compat.v1.enable_eager_execution()



'''parser = argparse.ArgumentParser()


parser.add_argument(
    '--operator',
    type=str,
    default='concate_all',
    help='average,hadamard,weighted_l1,weighted_l2,concate_all')
parser.add_argument(
    '--metric_type',
    type=str,
    default='entropy',
    help='kl_divergence, js_divergence, entropy')


args = parser.parse_args()'''
operator = 'concate_all'
metric_type = 'entropy'


def average(a, b):
    return (a + b) / 2


def hadamard(a, b):
    return a * b


def weighted_l1(a, b):
    return abs(a - b)


def weighted_l2(a, b):
    return abs((a - b) * (a - b))


def concate_all(a, b):
    return np.concatenate(
        (average(a, b), hadamard(a, b), weighted_l1(a, b), weighted_l2(a, b)))


def operator_func(operator, a, b):
    if operator == "average":
        return average(a, b)
    elif operator == "hadamard":
        return hadamard(a, b)
    elif operator == "weighted_l1":
        return weighted_l1(a, b)
    elif operator == "weighted_l2":
        return weighted_l2(a, b)
    elif operator == "concate_all":
        return concate_all(a, b)


def get_metrics(a, b, metric_type, operator_func):
    if metric_type == "kl_divergence":
        s1 = np.array([kl_divergence(a, b)])
        s2 = np.array(kl_divergence(b, a))

    elif metric_type == "js_divergence":
        s1 = np.array([js_divergence(a, b)])
        s2 = np.array(js_divergence(b, a))

    elif metric_type == "entropy":
        s1 = np.array([entropy(a)])
        s2 = np.array([entropy(b)])
    return operator_func(operator, s1, s2)


def prepare_attack_line(row, attack_type, similarity_list):
    all_features = []
    if attack_type in [3, 6]:
        a = np.array(row["gcn_pred0"])
        b = np.array(row["gcn_pred1"])
        feature_vec1 = operator_func(operator, a, b)  # posterior poerator
        all_features.append(feature_vec1)

    t0 = np.array(row["gcn_pred0"])
    r0 = np.array(row["dense_pred0"])
    f0 = np.array(row["feature_arr0"])
    t1 = np.array(row["gcn_pred1"])
    r1 = np.array(row["dense_pred1"])
    f1 = np.array(row["feature_arr1"])

    target_similarity = np.array([row(t0, t1) for row in similarity_list])
    target_metric_vec = get_metrics(t0-t0.min(), t1 - t1.min(), metric_type, operator_func)
    all_features.append(target_similarity)
    all_features.append(target_metric_vec)

    if attack_type in [5, 6, 7]:
        reference_similarity = np.array(
            [row(r0, r1) for row in similarity_list])
        feature_similarity = np.array([row(f0, f1) for row in similarity_list])
        reference_metric_vec = get_metrics(r0 - r0.min(), r1 - r1.min(), metric_type, operator_func)
        all_features += [reference_similarity, feature_similarity, reference_metric_vec]

    line = np.concatenate(all_features)
    line = np.nan_to_num(line)
    return line


def prepare_attack_line_v2(row, attack_type, similarity_list):
    columns = list(row.index)
    all_features = []
    if attack_type in [3, 6]:
        a = np.array(row[search_in_columns("gcn_pred0", columns)])
        b = np.array(row[search_in_columns("gcn_pred1", columns)])
        feature_vec1 = operator_func(operator, a, b)  # posterior poerator
        all_features.append(feature_vec1)

    t0 = np.array(row[search_in_columns("gcn_pred0", columns)])
    r0 = np.array(row[search_in_columns("dense_pred0", columns)])
    f0 = np.array(row[search_in_columns("feature_arr0", columns)])
    t1 = np.array(row[search_in_columns("gcn_pred1", columns)])
    r1 = np.array(row[search_in_columns("dense_pred1", columns)])
    f1 = np.array(row[search_in_columns("feature_arr1", columns)])

    target_similarity = np.array([row(t0, t1) for row in similarity_list])
    target_metric_vec = get_metrics(t0-t0.min(), t1 - t1.min(), metric_type, operator_func)
    all_features.append(target_similarity)
    all_features.append(target_metric_vec)

    if attack_type in [5, 6, 7]:
        reference_similarity = np.array(
            [row(r0, r1) for row in similarity_list])
        feature_similarity = np.array([row(f0, f1) for row in similarity_list])
        reference_metric_vec = get_metrics(r0 - r0.min(), r1 - r1.min(), metric_type, operator_func)
        all_features += [reference_similarity, feature_similarity, reference_metric_vec]

    line = np.concatenate(all_features)
    line = np.nan_to_num(line)
    return line


def load_data(train_path, test_path, shadow_train_path="", shadow_test_path="", attack_type=3):
    similarity_list = [cosine, euclidean, correlation, chebyshev,
                       braycurtis, canberra, cityblock, sqeuclidean]
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    id_train = []
    id_test = []
    g_train = []
    g_test = []
    train_list = []
    if attack_type in [1, 5]:
        train_data = open(shadow_train_path).readlines() + open(shadow_test_path).readlines()
    elif attack_type in [3, 6]:
        train_data = open(train_path).readlines()
    else:
        train_data = open(train_path).readlines() + open(shadow_train_path).readlines() + \
                     open(shadow_test_path).readlines()
    test_data = open(test_path).readlines()
    for row in train_data:
        row = json.loads(row)
        line = prepare_attack_line(row, attack_type, similarity_list)
        x_train.append(line)  # concatenate target and reference output, []
        y_train.append(row["label"])
        id_train.append(row['id_pair'])
        try:
            g_train.append(row['gender_group'])
        except:
            g_train.append(-1)
    for row in test_data:
        row = json.loads(row)
        line = prepare_attack_line(row, attack_type, similarity_list)
        x_test.append(line)
        y_test.append(row["label"])
        id_test.append(row['id_pair'])
        try:
            g_test.append(row['gender_group'])
        except:
            g_test.append(-1)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(
        y_test), np.array(id_train), np.array(id_test), np.array(g_train), np.array(g_test)


def search_in_columns(name, columns):
    res = []
    for c in columns:
        if name in c:
            res.append(c)
    return res


def load_data_v2(train_path, test_path, shadow_train_path="", shadow_test_path="", attack_type=3):
    similarity_list = [cosine, euclidean, correlation, chebyshev,
                       braycurtis, canberra, cityblock, sqeuclidean]
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    id_train = []
    id_test = []
    g_train = []
    g_test = []
    train_list = []
    if attack_type in [1, 5]:
        train_list.append(shadow_train_path)
        train_list.append(shadow_test_path)
    elif attack_type in [3, 6]:
        train_list.append(train_path)
    else:
        train_list.append(shadow_train_path)
        train_list.append(shadow_test_path)
        train_list.append(train_path)
    if len(train_list) > 1:
        train_data = pd.concat([pd.read_csv(source) for source in train_list])
    else:
        train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)


    for index, row in tqdm(train_data.iterrows(), total=train_data.shape[0]):
        line = prepare_attack_line_v2(row, attack_type, similarity_list)
        x_train.append(line)  # concatenate target and reference output, []
        y_train.append(row["label"])
        id_train.append([row['id_node0'], row['id_node1']])
        try:
            g_train.append(row['gender_group'])
        except:
            g_train.append(-1)
    for index, row in tqdm(test_data.iterrows(), total=test_data.shape[0]):
        line = prepare_attack_line_v2(row, attack_type, similarity_list)
        x_test.append(line)
        y_test.append(row["label"])
        id_test.append([row['id_node0'], row['id_node1']])
        try:
            g_test.append(row['gender_group'])
        except:
            g_test.append(-1)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(
        y_test), np.array(id_train), np.array(id_test), np.array(g_train), np.array(g_test)


def load_data_v3(mia_input_path, mia_input_length=-5):
    train_test_data = pd.read_csv(mia_input_path, header=None)
    # **** id0, id1, member, gender, train_test
    train_ind = train_test_data.iloc[:, -1] == 1
    test_ind = train_test_data.iloc[:, -1] == 0
    x_train = train_test_data[train_ind].iloc[:, :mia_input_length]
    x_test = train_test_data[test_ind].iloc[:, :mia_input_length]
    g_train = train_test_data[train_ind].iloc[:, -2]
    g_test = train_test_data[test_ind].iloc[:, -2]
    y_train = train_test_data[train_ind].iloc[:, -3]
    y_test = train_test_data[test_ind].iloc[:, -3]
    id_train = train_test_data[train_ind].iloc[:, -5:-3]
    id_test = train_test_data[test_ind].iloc[:, -5:-3]

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(
        y_test), np.array(id_train), np.array(id_test), np.array(g_train), np.array(g_test)


def prepare_MIA_inputs(datapath, fair_sample, dataset, ratio, shadow_data, t, attack_type, prepare_new=True, top_k=-1):
    if not prepare_new:
        MIA_input_file = datapath + "MIA_input/t={}/{}_{}attack{}.pkl".format(t, dataset, "fair_" if fair_sample else "", attack_type)
        x_train, x_test, y_train, y_test, id_train, id_test, g_train, g_test = pkl.load(MIA_input_file, encoding="latin1")
        return x_train, x_test, y_train, y_test, id_train, id_test, g_train, g_test
    partial_graph_path = datapath + "partial/t={}/".format(t)
    if top_k > 0:
        partial_graph_path += "K={}/".format(top_k)
    if fair_sample:
        train_path = partial_graph_path + "%s_train_ratio_%s_train_fair.json" % (dataset, ratio)
        test_path = partial_graph_path + "%s_train_ratio_%s_test_fair.json" % (dataset, ratio)
    else:
        train_path = partial_graph_path + "%s_train_ratio_%s_train.json" % (dataset, ratio)
        test_path = partial_graph_path + "%s_train_ratio_%s_test.json" % (dataset, ratio)
    strain_path = datapath.split('/')[0] + "/partial/t=0/%s_train_ratio_%s_train.json" % (shadow_data, ratio)
    stest_path = datapath.split('/')[0] + "/partial/t=0/%s_train_ratio_%s_test.json" % (shadow_data, ratio)

    x_train, x_test, y_train, y_test, id_train, id_test, g_train, g_test = load_data(train_path, test_path,
                                                                                     strain_path,
                                                                                     stest_path,
                                                                                     attack_type=attack_type)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.fit_transform(x_test)
    x_train_shape = x_train.shape[-1]
    x_test_shape = x_train_shape

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    all_inputs = [x_train, x_test, y_train, y_test, id_train, id_test, g_train, g_test]
    if not os.path.exists(datapath + "MIA_input/t={}".format(t)):
        os.makedirs(datapath + "MIA_input/t={}".format(t))
    with open(datapath + "MIA_input/t={}/{}_{}attack{}.pkl".format(t, dataset, "fair_" if fair_sample else "", attack_type), 'wb') as f:
        pkl.dump(all_inputs, f)
    return x_train, x_test, y_train, y_test, id_train, id_test, g_train, g_test


def prepare_MIA_inputs_v2(datapath, fair_sample, dataset, ratio, shadow_data, t, attack_type, prepare_new=True, top_k=-1):
    if not prepare_new:
        MIA_input_file = datapath + "MIA_input/t={}/{}_{}attack{}.pkl".format(t, dataset, "fair_" if fair_sample else "", attack_type)
        x_train, x_test, y_train, y_test, id_train, id_test, g_train, g_test = pkl.load(MIA_input_file, encoding="latin1")
        return x_train, x_test, y_train, y_test, id_train, id_test, g_train, g_test
    partial_graph_path = datapath + "partial/t={}/".format(t)
    if top_k > 0:
        partial_graph_path += "K={}/".format(top_k)
    if fair_sample:
        train_path = partial_graph_path + "%s_train_ratio_%s_train_fair.csv" % (dataset, ratio)
        test_path = partial_graph_path + "%s_train_ratio_%s_test_fair.csv" % (dataset, ratio)
    else:
        train_path = partial_graph_path + "%s_train_ratio_%s_train.csv" % (dataset, ratio)
        test_path = partial_graph_path + "%s_train_ratio_%s_test.csv" % (dataset, ratio)
    strain_path = datapath.split('/')[0] + "/partial/t=0/%s_train_ratio_%s_train.csv" % (shadow_data, ratio)
    stest_path = datapath.split('/')[0] + "/partial/t=0/%s_train_ratio_%s_test.csv" % (shadow_data, ratio)

    x_train, x_test, y_train, y_test, id_train, id_test, g_train, g_test = load_data_v2(train_path, test_path,
                                                                                     strain_path,
                                                                                     stest_path,
                                                                                     attack_type=attack_type)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.fit_transform(x_test)
    x_train_shape = x_train.shape[-1]
    x_test_shape = x_train_shape

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    all_inputs = [x_train, x_test, y_train, y_test, id_train, id_test, g_train, g_test]
    if not os.path.exists(datapath + "MIA_input/t={}".format(t)):
        os.makedirs(datapath + "MIA_input/t={}".format(t))
    with open(datapath + "MIA_input/t={}/{}_{}attack{}.pkl".format(t, dataset, "fair_" if fair_sample else "", attack_type), 'wb') as f:
        pkl.dump(all_inputs, f)
    return x_train, x_test, y_train, y_test, id_train, id_test, g_train, g_test


def prepare_MIA_inputs_v3(datapath, partial_graph_path, fair_sample, dataset, ratio, t, attack_type, top_k=-1):

    if fair_sample:
        diff_path = partial_graph_path + "diff_%s_train_ratio_%0.1f_train_fair.csv" % (dataset, ratio)
    else:
        diff_path = partial_graph_path + "diff_%s_train_ratio_%0.1f_train.csv" % (dataset, ratio)
    if str(attack_type) == "6":
        mia_input_lenth = -5
    elif dataset == "facebook":
        mia_input_lenth = 28
    elif dataset == "pokec":
        mia_input_lenth = 20
    else:
        mia_input_lenth = 20

    x_train, x_test, y_train, y_test, id_train, id_test, g_train, g_test = load_data_v3(diff_path,
                                                                                        mia_input_length=mia_input_lenth)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.fit_transform(x_test)
    x_train_shape = x_train.shape[-1]

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    all_inputs = [x_train, x_test, y_train, y_test, id_train, id_test, g_train, g_test]
    x_train = np.nan_to_num(x_train, 0)
    x_test = np.nan_to_num(x_test, 0)
    return x_train, x_test, y_train, y_test, id_train, id_test, g_train, g_test


def attack_main(datapath="GAT/", dataset="facebook", saving_path="GAT/",
                ratio=0.2, attack_type=3, fair_sample=False, t=0, prepare_new=True, top_k=-1):
    partial_path = saving_path + "partial/t={}/".format(t)
    x_train, x_test, y_train, y_test, id_train, id_test, g_train, g_test = prepare_MIA_inputs_v3(datapath,
                                                                                                 partial_path,
                                                                                                 fair_sample,
                                                                                                 dataset,
                                                                                                 ratio,
                                                                                                 t,
                                                                                                 attack_type)
    x_train_shape = x_train.shape[-1]
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(x_train_shape,)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    #callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
    model.compile(
        loss='categorical_crossentropy', optimizer=tf.compat.v1.train.AdamOptimizer())
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
       # callbacks=[callback],
        validation_data=(x_test, y_test))

    '''# 1st layer weights
    weights = model.layers[0].weights[0].numpy().mean(axis=1)
    w_names = []
    if attack_type not in [5, 6, 7]:
        len_post = (len(weights) - 8 - 4)//4
    else:
        len_post = (len(weights) - 8 - 4 - 8 - 8 - 4)//4
    sim_name_list = ['cosine', 'euclidean', 'correlation', 'chebyshev',
                     'braycurtis', 'canberra', 'cityblock', 'sqeuclidean']
    if attack_type in [3, 6]:
        w_names = w_names + ['avg_post[{}]'.format(num) for num in range(len_post)]
        w_names = w_names + ['mul_post[{}]'.format(num) for num in range(len_post)]
        w_names = w_names + ['l1_post[{}]'.format(num) for num in range(len_post)]
        w_names = w_names + ['l2_post[{}]'.format(num) for num in range(len_post)]
    w_names = w_names + ["post_{}".format(name) for name in sim_name_list]
    w_names = w_names + ["avg_E(post)", "mul_E(post)", "l1_E(post)", "l2_E(post)"]
    if attack_type in [5, 6, 7]:
        w_names = w_names + ["ref_{}".format(name) for name in sim_name_list]
        w_names = w_names + ["feat_{}".format(name) for name in sim_name_list]
        w_names = w_names + ["avg_E(ref)", "mul_E(ref)", "l1_E(ref)", "l2_E(ref)"]
    array_weight = np.array([weights, abs(weights), w_names]).T
    df_weight = pd.DataFrame(array_weight, columns=['Weights', "Abs Weights", "Weight Names"])'''

    y_pred_train = model.predict(x_train)
    y_train_label = [row[1] for row in y_train]
    y_train_pred = [round(row[1]) for row in y_pred_train]


    y_pred = model.predict(x_test)

    # add precision recall score
    y_test_label = [row[1] for row in y_test]
    y_pred_label = [round(row[1]) for row in y_pred]
    y_pred_pob = [row[1] for row in y_pred]

    test_acc = accuracy_score(y_test_label, y_pred_label)
    test_precision = precision_score(y_test_label, y_pred_label)
    test_recall = recall_score(y_test_label, y_pred_label)
    test_auc = roc_auc_score(y_test, y_pred)

    print('Test accuracy:', test_acc)
    print("Test Precision", test_precision)
    print("Test Recall", test_recall)
    print('Test auc:', test_auc)
    saving_path = saving_path + 'MIA_res/t={}'.format(t)
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    #df_weight.to_csv(saving_path+'/{}attack{}_weights_{}.csv'.format("fair_" if fair_sample else "",
    #                                                              attack_type, dataset), index=False)

    acc_list = save_attack_res(saving_path, dataset, y_test_label,
                    y_pred_pob, y_pred_label, id_test, ratio,
                    attack_type=attack_type, fair_sample=fair_sample,
                    y_train_label=y_train_label, y_train_pred=y_train_pred,
                    id_train=id_train, g_train=g_train, g_test=g_test)
    del model
    return test_acc, test_precision, test_recall, test_auc, acc_list



