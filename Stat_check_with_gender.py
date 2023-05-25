import os.path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from igraph import *
from utils import load_graph, load_att, load_data
import pickle as pkl
import seaborn as sns
from matplotlib.ticker import PercentFormatter


def has_node(graph, name):
    try:
        graph.vs.find(name=name)
    except:
        return False
    return True


def get_metric(avg_pop):
    acc_pop = (avg_pop.loc[2, 'TN'] + avg_pop.loc[2, 'TP']) / avg_pop.loc[2].sum()
    prec_pop = avg_pop.loc[2, 'TP'] / (avg_pop.loc[2, 'TP'] + avg_pop.loc[2, 'FP'])
    recall_pop = avg_pop.loc[2, 'TP'] / (avg_pop.loc[2, 'TP'] + avg_pop.loc[2, 'FN'])
    count_pop = avg_pop.loc[2].sum()

    acc_pop1 = (avg_pop.loc[1, 'TN'] + avg_pop.loc[1, 'TP']) / avg_pop.loc[1].sum()
    prec_pop1 = avg_pop.loc[1, 'TP'] / (avg_pop.loc[1, 'TP'] + avg_pop.loc[1, 'FP'])
    recall_pop1 = avg_pop.loc[1, 'TP'] / (avg_pop.loc[1, 'TP'] + avg_pop.loc[1, 'FN'])
    count_pop1 = avg_pop.loc[1].sum()

    acc_unpop = (avg_pop.loc[0, 'TN'] + avg_pop.loc[0, 'TP']) / avg_pop.loc[0].sum()
    prec_unpop = avg_pop.loc[0, 'TP'] / (avg_pop.loc[0, 'TP'] + avg_pop.loc[0, 'FP'])
    recall_unpop = avg_pop.loc[0, 'TP'] / (avg_pop.loc[0, 'TP'] + avg_pop.loc[0, 'FN'])
    count_unpop = avg_pop.loc[0].sum()
    return np.array([['Inner 2', float(acc_pop), float(prec_pop), float(recall_pop), int(count_pop)],
                     ['Inner 1', float(acc_pop1), float(prec_pop1), float(recall_pop1), int(count_pop1)],
                     ['Inter gender', float(acc_unpop), float(prec_unpop), float(recall_unpop), int(count_unpop)]])


def agg_results(dataset, ratio, res_path, graph_file, attack_type=3, fair_sample=True):
    if fair_sample:
        res_file = res_path + "/{}_{}_fair_attack{}.csv".format(dataset, ratio, attack_type)
    else:
        res_file = res_path + "/{}_{}_attack{}.csv".format(dataset, ratio, attack_type)
    res = pd.read_csv(res_file)
    res['degree_product'] = res['Node1 Degree'] * res['Node2 Degree']
    names = np.array(['TN', 'FN', 'FP', 'TP'])
    res['group_name'] = names[res['label(TP FP)'].to_numpy()]
    res['Node Degree min'] = res[['Node1 Degree', 'Node2 Degree']].min(axis=1)
    res['Node Degree max'] = res[['Node1 Degree', 'Node2 Degree']].max(axis=1)
    res['Member'] = res['group_name'].isin(['TP', 'FN'])

    all_nodes = np.unique(
        np.vstack([res[['Node1', 'Node1 Degree']].to_numpy(), res[['Node2', 'Node2 Degree']].to_numpy()]), axis=0)

    # init graph
    edges = load_graph(graph_file, dataset)
    attentions = load_att("GAT/", dataset)


    max_vid = max(res['Node1'].max(), res['Node2'].max())
    g = Graph()
    g.add_vertices(max_vid + 1)
    g.add_edges(edges)

    # Vertics
    vb = g.betweenness()
    evc = g.evcent()
    res['Vertex Betweenness min'] = 0
    res['Vertex Eigenvector Centrality max'] = 0
    res['Vertex Betweenness min'] = 0
    res['Vertex Eigenvector Centrality max'] = 0
    res['Group on Gender'] = 0
    res['attention min'] = 0
    res['attention max'] = 0
    res['attention avg'] = 0
    for i in range(len(res)):
        id1 = int(res.loc[i, 'Node1'])
        id2 = int(res.loc[i, 'Node2'])
        res.loc[i, 'Vertex Betweenness min'] = min(vb[id1], vb[id2])
        res.loc[i, 'Vertex Eigenvector Centrality min'] = min(evc[id1], evc[id2])
        res.loc[i, 'Vertex Betweenness max'] = max(vb[id1], vb[id2])
        res.loc[i, 'Vertex Eigenvector Centrality max'] = max(evc[id1], evc[id2])
        att = [attentions[id1, id2], attentions[id2, id1]]
        res.loc[i, 'attention min'] = min(att)
        res.loc[i, 'attention max'] = max(att)
        res.loc[i, 'attention avg'] = mean(att)

        if res.loc[i, 'Node1 Gender'] == 1 and res.loc[i, 'Node2 Gender'] == 1:
            res.loc[i, 'Group on Gender'] = 1
        if res.loc[i, 'Node1 Gender'] == 2 and res.loc[i, 'Node2 Gender'] == 2:
            res.loc[i, 'Group on Gender'] = 2

    attention_avg = res.groupby("group_name")[["attention max", "attention min", "attention avg"]].mean()
    attention_avg2 = res.groupby("Group on Gender")[["attention max", "attention min", "attention avg"]].mean()
    attention_avg = pd.concat([attention_avg, attention_avg2])
    if fair_sample:
        attention_avg.to_csv(res_path + "{}_{}_fair_attack{}_attention.csv".format(dataset, ratio, attack_type))
    else:
        attention_avg.to_csv(res_path + "{}_{}_attack{}_attention.csv".format(dataset, ratio, attack_type))



    avg_group = res.groupby(['Group on Gender', 'group_name'])[['Label']].count()
    metric = get_metric(avg_group)
    group_true_count = res.groupby(['Group on Gender'])[['Label']].sum()
    df_metric_with = pd.DataFrame(metric,
                                  columns=['Group on Gender', 'Accuracy', 'Precision', 'Recall', 'Count'])
    df_metric_with['True Count'] = group_true_count
    if fair_sample:
        df_metric_with.to_csv(res_path + "{}_{}_fair_attack{}_Disparity.csv".format(dataset, ratio, attack_type),
                              index=False)
    else:
        df_metric_with.to_csv(res_path + "{}_{}_attack{}_Disparity.csv".format(dataset, ratio, attack_type), index=False)


def attack_agg(dataset, res_path="GAT/MIA_res/", fair_sample=True):
    res_file = res_path+"{}_attack{}acc.csv".format(dataset, "_fair_" if fair_sample else "")
    df = pd.read_csv(res_file)
    df_avg = df.groupby(['Attack']).mean()
    df_avg.to_csv(res_path + "{}_attack_{}agg.csv".format(dataset, "fair_" if fair_sample else ""))
    pass


def distance_analysis(len_post=4, model_type="GAT", res_root="", dataset="facebook"):
    res_path = "{}/{}partial/t=0/diff_{}_train_ratio_0.2_train_fair.csv".format(model_type, res_root, dataset)
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
    post_columns = list(range(4 * len_post + 32))
    res = pd.read_csv(res_path, header=None)
    # plot_density(res, post_columns, w_names, model_type, res_root)
    plot_hist(res, post_columns, w_names, model_type, res_root, dataset=dataset)


def plot_density(res, post_columns, w_names=[], model_type="GAT", res_root="", member=False, dataset=""):
    if len(w_names) == 0:
        w_names = ['metrics {}'.format(w) for w in post_columns]
    for c in post_columns:
        for group in [2, 1, 0]:
            if member:
                for mem in [0, 1]:
                    sub_res = res[(res[res.columns[-1]] == group) * (res[res.columns[-2]] == mem)]
                    sns.distplot(sub_res[c], hist=False, kde=True,
                                 kde_kws={'linewidth': 1},
                                 label="Group {}, {}".format(group, "Member" if mem else "Non-Member"))
            else:
                sub_res = res[(res[res.columns[-1]] == group)]
                sns.distplot(sub_res[c], hist=False, kde=True,
                             kde_kws={'linewidth': 1},
                             label="Group {}".format(group))
        plt.xlabel("Metrics Value")
        plt.ylabel("Probability Density Distribution")
        plt.title("Pdf curve of {}".format(w_names[c]))
        if not os.path.exists("{}/{}partial/t=0/figure/{}".format(model_type, res_root, dataset)):
            os.makedirs("{}/{}partial/t=0/figure/{}".format(model_type, res_root, dataset))
        plt.savefig("{}/{}partial/t=0/figure/{}/density_metric{}".format(model_type, res_root, dataset, c))
        plt.close()


def plot_hist(res, post_columns, w_names=[], model_type="GAT", res_root="", member=False, dataset="pokec"):
    if len(w_names) == 0:
        w_names = ['metrics {}'.format(w) for w in post_columns]
    ss = StandardScaler()
    res.iloc[:, :-2] = ss.fit_transform(res.iloc[:, :-2])
    for c in post_columns:
        if member:
            x0 = list(res[(res[res.columns[-1]] == 0) * (res[res.columns[-2]] == 1)][c])
            x1 = list(res[(res[res.columns[-1]] == 0) * (res[res.columns[-2]] == 0)][c])
            x2 = list(res[(res[res.columns[-1]] == 1) * (res[res.columns[-2]] == 1)][c])
            x3 = list(res[(res[res.columns[-1]] == 1) * (res[res.columns[-2]] == 0)][c])
            x4 = list(res[(res[res.columns[-1]] == 2) * (res[res.columns[-2]] == 1)][c])
            x5 = list(res[(res[res.columns[-1]] == 2) * (res[res.columns[-2]] == 0)][c])
            x_list = [x0, x1, x2, x3, x4, x5]
            names = ['Intra Gender Member',
                     'Intra Gender Non-member',
                     'Inner 1 Member',
                     'Inner 1 Non-member',
                     'Inner 2 Member',
                     'Inner 2 Non-member']
        else:
            x0 = list(res[(res[res.columns[-1]] == 0)][c])
            x1 = list(res[(res[res.columns[-1]] == 1)][c])
            x2 = list(res[(res[res.columns[-1]] == 2)][c])
            x_list = [x0, x1, x2]
            names = ['Intra Gender',
                     'Inner 1',
                     'Inner 2']
        weights = [np.ones(len(xi)) / len(xi) for xi in x_list]
        plt.hist(x_list, bins=int(180 / 15), density=False, label=names, weights=weights)

        if not os.path.exists("{}/{}partial/t=0/figure/{}".format(model_type, res_root, dataset)):
            os.makedirs("{}/{}partial/t=0/figure/{}".format(model_type, res_root, dataset))
        plt.legend()
        plt.xlabel("Metrics Value")
        plt.ylabel("Percent of data")
        plt.title("Hist Plot Distribution of {}".format(w_names[c]))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.savefig("{}/{}partial/t=0/figure/{}/hist_metric{}".format(model_type, res_root, dataset, c))
        plt.close()


if __name__ == "__main__":
    dataset = 'tagged_20'
    len_post = 2
    #distance_analysis(model_type='gcn', res_root="", len_post=len_post, dataset=dataset)
    for delta in [0.1, 0.2]:
        pass

        #distance_analysis(model_type='gcn', res_root="CNR/Group/Reduce/Delta={}/".format(delta), len_post=len_post, dataset=dataset)
    #attack_agg(dataset, res_path="gcn/MIA_res/", fair_sample=True)
    for attack_type in [3]:
        pass
        agg_results(dataset, ratio=0.2, res_path="gcn/DP/beta=1.0/t=0/MIA_res/t=0/K=-1/", graph_file="gcn/",
                    attack_type=attack_type, fair_sample=False)