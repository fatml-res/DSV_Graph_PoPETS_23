import os.path
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from stealing_link.keras_utils import *
from keras.models import Sequential
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
from utils import save_attack_res
import os
from Stat_check_with_gender import plot_hist, plot_density
import matplotlib.pyplot as plt


def perturb_experiment(k, noise_std, attack_type, t):
    input_file = "GAT/MIA_input/t={}/facebook_fair_attack{}.pkl".format(t, attack_type)
    x_train, x_test, y_train, y_test, id_train, id_test, g_train, g_test = pkl.loads(open(input_file, "rb").read())
    # get top-k index
    if k == 0:
        k = x_train.shape[1]
    weight_file = "GAT/MIA_res/t={}/attack{}_weights.csv".format(t, attack_type)
    weights = pd.read_csv(weight_file)
    important_index = weights.sort_values("Abs Weights").index[:k]
    index_1 = g_train == 1
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.fit_transform(x_test)
    for m_ind in important_index:
        minority_topk_features = x_train[index_1, m_ind]
        for g in [0, 2]:
            noise = np.random.normal(0, noise_std/(g+1), minority_topk_features.shape)
            new_topk = noise + minority_topk_features
            index_g = g_train == g
            x_train[index_g, m_ind] = new_topk
    for m_ind in weights.sort_values("Abs Weights").index[k:]:
        minority_bottom_features = x_train[index_1, m_ind]
        for g in [0, 2]:
            index_g = g_train == g
            x_train[index_g, m_ind] = minority_bottom_features
    x_train = ss.fit_transform(x_train)
    for g in [0, 2]:
        index_g = g_train == g
        y_train[index_g] = y_train[g_train == 1]

    # distribution histgram check
    distance_analysis(x_train, g_train, y_train[:, 1], noise_std, t,
                      attack_type=attack_type, model_type="GAT", index=important_index)
    '''# considering the gender group sizes are not identical in x_test, group 0 and 2 will randomly select value from group 1

    index_1 = g_test == 1
    for m_ind in metric_index:
        minority_topk_features = x_test[index_1, m_ind]
        for g in [0, 2]:
            index_g = g_test == g
            feat_base = np.random.choice(minority_topk_features, sum(index_g))
            noise = np.random.normal(0, noise_std, sum(index_g))
            x_test[index_g, m_ind] = feat_base + noise
    x_test = ss.fit_transform(x_test)'''

    # generate noise
    # replace top-k index of group 0 and 2 with group 1's top-k feature
    # re-run MIA
    x_train_shape = x_train.shape[-1]
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(x_train_shape,)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.summary()

    model.compile(
        loss='categorical_crossentropy', optimizer=Adam())
    model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=50,
        verbose=2,
        validation_data=(x_test, y_test))

    # measure MIA performance
    y_pred_train = model.predict(x_train)
    y_train_label = [row[1] for row in y_train]
    y_train_pred = [round(row[1]) for row in y_pred_train]

    y_pred = model.predict(x_test)
    y_test_label = [row[1] for row in y_test]
    y_pred_label = [round(row[1]) for row in y_pred]
    y_pred_prob = [row[1] for row in y_pred]

    test_acc = accuracy_score(y_test_label, y_pred_label)
    test_precision = precision_score(y_test_label, y_pred_label)
    test_recall = recall_score(y_test_label, y_pred_label)
    test_auc = roc_auc_score(y_test, y_pred)

    print('Test accuracy:', test_acc)
    print("Test Precision", test_precision)
    print("Test Recall", test_recall)
    print('Test auc:', test_auc)

    # save perturb
    saving_address = "GAT/perturbation/std={}/t={}".format(noise_std, t)
    if not os.path.exists(saving_address):
        os.makedirs(saving_address)
    save_attack_res(saving_address, "facebook", y_test_label, y_pred_prob, y_pred_label, id_test, ratio=0.2,
                    attack_type=attack_type, fair_sample=True, y_train_label=y_train_label,
                    y_train_pred=y_train_pred, id_train=id_train)
    w_names = weights['Weight Names']
    new_weights = model.layers[0].weights[0].numpy().mean(axis=1)
    array_weight = np.array([new_weights, abs(new_weights), w_names]).T
    df_weight = pd.DataFrame(array_weight, columns=['Weights', "Abs Weights", "Weight Names"])
    df_weight.to_csv(saving_address + '/perturbation_attack{}_weights.csv'.format(attack_type), index=None)


def distance_analysis(x_train, gender, y, std, t, attack_type=3, len_post=4, model_type="GAT", index=[]):
    w_names = []
    sim_name_list = ['cosine', 'euclidean', 'correlation', 'chebyshev',
                     'braycurtis', 'canberra', 'cityblock', 'sqeuclidean']
    w_names = w_names + ['avg_post[{}]'.format(num) for num in range(len_post)]
    w_names = w_names + ['mul_post[{}]'.format(num) for num in range(len_post)]
    w_names = w_names + ['l1_post[{}]'.format(num) for num in range(len_post)]
    w_names = w_names + ['l2_post[{}]'.format(num) for num in range(len_post)]
    w_names = w_names + ["post_{}".format(name) for name in sim_name_list]
    w_names = w_names + ["avg_E(post)", "mul_E(post)", "l1_E(post)", "l2_E(post)"]
    w_names = w_names + ["ref_{}".format(name) for name in sim_name_list]
    w_names = w_names + ["feat_{}".format(name) for name in sim_name_list]
    w_names = w_names + ["avg_E(ref)", "mul_E(ref)", "l1_E(ref)", "l2_E(ref)"]
    post_columns = list(range(x_train.shape[1]))
    res = x_train
    if len(w_names) == 0:
        w_names = ['metrics {}'.format(w) for w in post_columns]
    for c in post_columns:
        if c not in index:
            continue
        x0 = list(res[(gender == 0) * (y == 1), c])
        x1 = list(res[(gender == 0) * (y == 0), c])
        x2 = list(res[(gender == 1) * (y == 1), c])
        x3 = list(res[(gender== 1) * (y == 0), c])
        x4 = list(res[(gender == 2) * (y == 1), c])
        x5 = list(res[(gender == 2) * (y == 0), c])
        names = ['Intra Gender Member',
                 'Intra Gender Non-member',
                 'Inner 1 Member',
                 'Inner 1 Non-member',
                 'Inner 2 Member',
                 'Inner 2 Non-member'
                 ]
        weights = [np.ones(len(x0)) / len(x0),
                            np.ones(len(x1)) / len(x1),
                            np.ones(len(x2)) / len(x2),
                            np.ones(len(x3)) / len(x3),
                            np.ones(len(x4)) / len(x4),
                            np.ones(len(x5)) / len(x5)]
        plt.hist([x0, x1, x2, x3, x4, x5], bins=int(180 / 15), density=False, label=names, weights=weights)

        if not os.path.exists("{}/perturbation/std={}/t={}/attack{}/figure".format(model_type, std, t, attack_type)):
            os.makedirs("{}/perturbation/std={}/t={}/attack{}/figure".format(model_type, std, t, attack_type))
        plt.legend()
        plt.xlabel("Metrics Value")
        plt.ylabel("Count of data")
        plt.title("Hist Plot Distribution of {}".format(w_names[c]))
        # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.savefig("{}/perturbation/std={}/t={}/attack{}/figure/hist_metric{}.png".format(model_type, std, t, attack_type, c))
        plt.close()


def aggregate_disparity():
    acc_list = []
    for std in [0, 0.5, 1.0, 3.0, 5.0]:
        for at in [3, 6]:
            for t in range(1):
                acc_file = "GAT/perturbation/std={}/t={}/facebook_MIAacc_fair_attack{}.pkl".format(std, t, at)
                list_acc = pkl.loads(open(acc_file, 'rb').read())
                acc_list.append([at, std] + list_acc)
    acc_list = np.array(acc_list)
    df_acc = pd.DataFrame(acc_list, columns=['Attack Type', 'std',
                                             'train acc', 'train acc 1', 'train acc 2', 'train acc 0',
                                             'test acc', 'test acc 1', 'test acc 2', 'test acc 0' ])
    df_acc.astype({'train acc': float, 'train acc 1':float, 'train acc 2':float, 'train acc 0':float,
                                             'test acc':float, 'test acc 1':float, 'test acc 2':float, 'test acc 0':float })
    df_avg = df_acc.groupby(['Attack Type', 'std']).mean()
    df_avg.to_csv("GAT/perturbation/Disparity_acc.csv")



if __name__ == "__main__":
    # aggregate_disparity()
    pass
    k = 0
    fair_sample = True
    for std in [0, 0.5, 1.0, 3.0, 5.0]:
    # for std in [0.1, 0.3]:
        for at in [3, 6]:
            for t in range(1):
                perturb_experiment(k, std, at, t)
    aggregate_disparity()




