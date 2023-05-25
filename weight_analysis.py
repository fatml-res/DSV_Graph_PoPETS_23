import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from common_Neighbor import cnr_link
from stealing_link.partial_graph_generation import get_link
import pickle as pkl
from attack import operator_func, get_metrics
from scipy.spatial.distance import cosine, euclidean, correlation, chebyshev,\
    braycurtis, canberra, cityblock, sqeuclidean
from sklearn.preprocessing import StandardScaler
import igraph
from sklearn.metrics import roc_curve, auc, f1_score
from tqdm import tqdm
import argparse


def get_link_genders(g1, g2):
    genders = []
    for i in range(len(g1)):
        if g1[i] == g2[i]:
            if g1[i] == 1:
                genders.append(1)
            else:
                genders.append(2)
        else:
            genders.append(0)
    return np.array(genders)


def two_files_check(wdf, adf):
    res = True
    for i in range(len(wdf)):
        wid1 = wdf['node1'][i]
        wid2 = wdf['node2'][i]
        aid1 = adf['Node1'][i]
        aid2 = adf['Node2'][i]
        if wid1 == aid1 and wid2 == aid2:
            pass
        else:
            print("Warning!")
            res = False
    return res


def two_files_join(wdf, adf):
    res_df = pd.merge(left=wdf, right=adf,
                      left_on=["node1", "node2"], right_on=['Node1', 'Node2'],
                      how="inner")
    return res_df.drop("Unnamed: 0", axis=1)


def get_interval_by_pct(weight, num_bin=10):
    sort_ind = np.argsort(weight)
    bins = np.zeros(len(weight))
    for i in range(num_bin):
        ids = sort_ind[round(i * 0.1 * len(weight)): round((i+1) * 0.1 * len(weight))]
        bins[ids] = i
    return bins


def get_cnrs(adj, all_df, dataset, model_type):
    # if there is cnr file in current location, read cnr from file
    file = "{}/weight/{}_cnr.npy".format(model_type, dataset)
    if os.path.exists(file):
        cnrs = np.load(file)
        all_df["cnr"] = cnrs
    else:
        # if not generate cnr and save this column to weight: {}/weight/{dataset}_cnr.npy
        all_df["cnr"] = 0
        for i in range(len(all_df)):
            all_df.loc[i, "cnr"] = cnr_link(all_df.loc[i, "Node1"],
                                            all_df.loc[i, "Node2"],
                                            adj)
        np_to_save = np.array(all_df["cnr"])
        np.save(file, np_to_save)
    return all_df


def get_edges_from_adj(adj, gender):
    edges = []
    node1, node2 = adj.nonzero()
    for i in range(len(node1)):
        if node1[i] > node2[i]:
            gi = (gender[node1[i]] == gender[node2[i]]) * gender[node1[i]]
            edges.append([node1[i], node2[i], gi])
    return np.array(edges)


def get_density(adj, gender):
    if torch.is_tensor(adj):
        adj = adj.numpy()
    N1 = (gender == 1).sum()
    N2 = (gender == 2).sum()
    N = len(gender)

    total = N * (N-1) / 2
    total0 = N1 * N2
    total1 = N1 * (N1-1) / 2
    total2 = N2 * (N2-1) /2

    density = len(adj.nonzero()[0]) / total / 2

    edge_genders = get_edges_from_adj(adj, gender)
    nums = np.unique(edge_genders[:, -1], return_counts=True)[1]
    density0 = nums[0] / total0
    density1 = nums[1] / total1
    density2 = nums[2] / total2

    return density, density0, density1, density2


def prepare_MIA_inputs(pred_array, dense_pred, ft, node1, node2):
    similarity_list = [cosine, euclidean, correlation, chebyshev,
                       braycurtis, canberra, cityblock, sqeuclidean]
    t0, t1 = pred_array[node1], pred_array[node2]
    r0, r1 = dense_pred[node1], dense_pred[node2]
    f0, f1 = ft[node1], ft[node2]
    post_agg_vec = operator_func("concate_all", np.array(t0), np.array(t1))
    target_similarity = np.array([row(t0, t1) for row in similarity_list])  # 8
    target_metric_vec = get_metrics(t0 - min(t0),
                                    t1 - min(t1),
                                    'entropy', operator_func)  # 4

    # Feature related, reference post similarity, used in attack 5, 6, 7
    reference_similarity = np.array([row(r0, r1) for row in similarity_list])  # 8
    feature_similarity = np.array([row(f0, f1) for row in similarity_list])  # 8
    reference_metric_vec = get_metrics(np.array(r0) - min(r0),
                                       np.array(r1) - min(r1),
                                       'entropy', operator_func)  # 4

    return np.concatenate([post_agg_vec,
                           target_similarity, target_metric_vec,
                           reference_similarity, feature_similarity, reference_metric_vec])


def get_all_MIA(g1, g2, pred_array, dense_pred, ft):
    res = []
    for i in range(len(g1)):
        res.append(prepare_MIA_inputs(pred_array, dense_pred, ft, g1[i], g2[i]))
    return np.vstack(res)


def weight_analysis_v2(Graph_name="Facebook-0.2_fair_attack3",
                       model_type="GAT",
                       delta=0.1,
                       run_plot=False):
    weight_names = ["bc_edge"]
    Graph_name = Graph_name.replace('-', '_')
    if "tagged" in Graph_name:
        Graph_name = Graph_name.replace("_fair", "")
    if "tagged" in Graph_name:
        dataset = "_".join(Graph_name.split("_")[:-2]).lower()
    else:
        dataset = Graph_name.split("_")[0].lower()
    if delta > 0:
        datasource = "{}/CNR/Group/Reduce/Delta={}/".format(model_type, delta)
    else:
        datasource = "{}/".format(model_type)
    weight_file = "{}/weight/{}_all_weights.csv".format(model_type,
                                                        dataset)
    attack_res = datasource + "MIA_res/t=0/{}.csv".format(Graph_name)
    adj_file = datasource + "ind.{}.adj".format(dataset)
    adj = pkl.load(open(adj_file, 'rb')).detach().numpy()
    target_res_file = datasource + "{}_{}_target.pkl".format(dataset,
                                                             model_type.lower())
    if model_type == "GAT":
        att_file = datasource + "ind.{}.att".format(dataset)
    else:
        att_file = "gcn/weight/{}_edge_influence.dict".format(dataset)

    att = load_att_infl(model_type, att_file=att_file, num_node=len(adj))
    weight_df = pd.read_csv(weight_file)
    attack_df = pd.read_csv(attack_res)
    all_items = pkl.load(open(target_res_file, 'rb'))
    pred_array = all_items[2].detach().numpy()
    ft = all_items[0].detach().numpy()

    all_df = two_files_join(weight_df, attack_df)
    infl_name = 'attention' if model_type == "GAT" else "influence"
    if model_type == "GAT":
        all_df[infl_name] = att[all_df['Node1'], all_df['Node2']] * att[all_df['Node2'], all_df['Node1']]
    else:
        all_df[infl_name] = att[all_df['Node1'], all_df['Node2']]

    all_df[infl_name] = (all_df[infl_name] - all_df[infl_name].min())/(all_df[infl_name].max() - all_df[infl_name].min())

    g1 = np.array(all_df['Node1 Gender'])
    g2 = np.array(all_df['Node2 Gender'])
    link_groups = get_link_genders(g1, g2)
    print(np.unique(link_groups, return_counts=True))


    all_df['Group'] = link_groups
    all_df["correct"] = all_df['Pred'] == all_df['Label']

    all_df = get_cnrs(adj, all_df, dataset=dataset, model_type=model_type)
    #all_df['cnr'] = all_df['jac_sim']
    agg_group = all_df.groupby(["Group", "Label"]).agg({'bc_edge': "mean",
                                                        'cnr': "mean",
                                                        'correct': "mean",
                                                        infl_name: "mean",
                                                        'Node1': "count"})
    group_diff = agg_group.query("Label==1").droplevel(1) - agg_group.query("Label==0").droplevel(1)
    group_acc_count = all_df.groupby("Group").agg(Accuracy=("correct", "mean"),
                                                  Count=("Node1", "count"))
    group_weight_diff = pd.merge(group_diff[["bc_edge", "cnr", infl_name]], group_acc_count,
                                 left_on="Group", right_on="Group")
    group_weight_diff.to_csv("{}/weight/{}_attack{}_Gap.csv".format(model_type,
                                                                    dataset,
                                                                    Graph_name[-1]))

    agg_member = all_df.groupby(["member"])[weight_names + [infl_name, 'cnr', 'correct']].mean()
    agg_group_member = all_df.groupby(["Group", "Label"])[weight_names + [infl_name, 'cnr', 'correct']].mean()
    agg_member_label = all_df.groupby(["Label", "correct"])[weight_names + [infl_name, 'cnr', 'correct']].mean()
    agg_group = all_df.groupby(["Group"])[weight_names + [infl_name, 'cnr', 'correct']].mean()
    agg_group['density'] = 0


    # density
    gender_file = datasource + "ind.{}.gender".format(dataset)
    genders = np.array(pkl.loads(open(gender_file, 'rb').read()))
    min_gender, maj_gender = np.unique(genders, return_counts=True)[0][
        np.unique(genders, return_counts=True)[1].argsort()]
    n_gender = [(genders == min_gender).sum(), (genders == maj_gender).sum()]
    for group in [0, 1, 2]:
        if group > 0:
            n_total = n_gender[group - 1] * (n_gender[group - 1]-1)//2
        else:
            n_total = n_gender[0] * n_gender[1]
        n_group = (all_df['Group'] == group).sum()
        agg_group.loc[group, 'density'] = n_group/n_total

    # overall
    row_overall = all_df[weight_names + [infl_name, 'cnr', 'correct']].mean()
    density_all = len(all_df) / len(genders) / (len(genders) - 1) * 2
    row_overall['density'] = density_all
    row_overall = pd.DataFrame(row_overall).T

    agg_group = row_overall.append(agg_group)
    agg_group = agg_group[["bc_edge", "cnr", infl_name, "density", 'correct']]


    with pd.ExcelWriter("{}/weight/{}_by_member.xlsx".format(model_type, Graph_name)) as writer:
        agg_group.to_excel(writer, sheet_name='By Group')
        agg_member.to_excel(writer, sheet_name='By Member')
        agg_group_member.to_excel(writer, sheet_name='By Member Group')
        agg_member_label.to_excel(writer, sheet_name='By Member TF')


    if run_plot:
        dense_pred = pkl.loads(open("dense/" + "%s_dense_pred.pkl" % dataset, "rb").read())
        w_names = get_names(Graph_name.split("-")[0].lower())

        inputs = get_all_MIA(np.array(all_df["Node1"]), np.array(all_df["Node2"]), pred_array, dense_pred, ft)
        ss = StandardScaler()
        inputs = ss.fit_transform(inputs)
        all_df[w_names] = inputs

        rank_accuracy = []
        rank_cnr = []
        MIA_input_list = []
        mem_nm_lists = []
        group_mem_lists = []
        group_nm_lists = []
        mem_ind = all_df["member"] == 1
        nm_ind = all_df["member"] == 0
        groups = all_df["Group"]

        weight_to_plot = weight_names + [infl_name, "cnr"]
        for w in weight_to_plot:
            weight = all_df[w]
            correct = all_df["correct"]
            cnrs = all_df["cnr"]
            bins = get_interval_by_pct(weight, num_bin=10)
            mem_acc_list = []
            cnr_list = []
            mem_list = []
            group_mem_list = []
            group_nm_list = []
            for i in range(10):
                ind_bin = bins == i
                mem_list.append((ind_bin * mem_ind).sum() / ind_bin.sum())
                mem_acc_list.append(correct[ind_bin * mem_ind].mean())
                cnr_list.append(cnrs[ind_bin].mean())
                MIA_input_list.append(all_df[w_names][ind_bin].mean().to_numpy())
                for g in [0, 1, 2]:
                    g_ind = groups == g
                    group_mem_list.append((ind_bin * mem_ind * g_ind).sum() / (g_ind * mem_ind).sum())
                    group_nm_list.append((ind_bin * nm_ind * g_ind).sum() / (g_ind * nm_ind).sum())
            rank_accuracy.append(mem_acc_list)
            rank_cnr.append(cnr_list)
            mem_nm_lists.append(mem_list)
            group_mem_lists.append(group_mem_list)
            group_nm_lists.append(group_nm_list)
        rank_accuracy = np.array(rank_accuracy)
        rank_cnr = np.array(rank_cnr)
        mem_nm_lists = np.array(mem_nm_lists)
        MIA_input_list = np.vstack(MIA_input_list).T
        group_mem_lists = np.array(group_mem_lists)
        group_nm_lists = np.array(group_nm_lists)

        for i in range(len(weight_to_plot)):
            x = range(10)
            plt.plot(x, rank_accuracy[i], label=weight_to_plot[i])
        plt.legend()
        plt.xticks(x, ["{}%".format(i * 10) for i in x])
        plt.xlabel("Weight Range")
        plt.ylabel("Average Accuracy")
        plt.savefig("GAT/weight/{}_Acc.png".format(Graph_name))
        plt.close()

        for i in range(len(weight_to_plot)):
            x = range(10)
            plt.plot(x, rank_cnr[i], label=weight_to_plot[i])
        plt.legend()
        plt.xticks(x, ["{}%".format(i * 10) for i in x])
        plt.xlabel("Weight Range")
        plt.ylabel("Average $cnr$")
        plt.savefig("GAT/weight/{}_cnr.png".format(Graph_name))
        plt.close()

        for i in range(len(weight_to_plot)):
            x = range(10)
            plt.plot(x, mem_nm_lists[i], label=weight_to_plot[i])
        plt.legend()
        plt.xticks(x, ["{}%".format(i * 10) for i in x])
        plt.xlabel("Weight Range")
        plt.ylabel("Ratio of Members")
        plt.savefig("GAT/weight/{}_ROM.png".format(Graph_name))
        plt.close()

        for i in range(len(weight_to_plot)):
            colors = [["royalblue", "purple", "orange"],
                      ["lightsteelblue", "violet", "moccasin"]]
            x = np.array(range(10))
            for g in [0, 1, 2]:
                mem_g = np.array([group_mem_lists[i][ind * 3 + g] for ind in range(10)])
                non_mem_g = np.array([group_nm_lists[i][ind * 3 + g] for ind in range(10)])
                plt.bar(x + 0.25 * g, non_mem_g, bottom=mem_g, color=colors[1][g],
                        label="{}_Non-Mem{}".format(weight_to_plot[i], g), width=0.25)
                plt.bar(x + 0.25 * g, mem_g, color=colors[0][g],
                        label="{}_Mem{}".format(weight_to_plot[i], g), width=0.25)
            plt.legend()
            plt.xticks(x, ["{}%".format(i * 10) for i in x])
            plt.xlabel("Weight Range")
            plt.ylabel("Distribution of Mem/Non-mem")
            plt.savefig("GAT/weight/{}_{}_group_mem.png".format(Graph_name.split("-")[0], weight_to_plot[i]))
            plt.close()

        for w in range(len(w_names)):
            for i in range(len(weight_to_plot)):
                x = range(10)
                plt.plot(x, MIA_input_list[w][i * 10: (i + 1) * 10], label=weight_to_plot[i])
            plt.legend()
            plt.xticks(x, ["{}%".format(i * 10) for i in x])
            plt.xlabel("Weight Range")
            plt.ylabel("Average {}".format(w_names[w]))
            plt.savefig("GAT/weight/m_input/{}/{}.png".format(Graph_name.split('-')[0].lower(), w_names[w]))
            plt.close()


def get_group_mean(df, p=0.05, num_node=0):
    # sample node
    number_to_sample = int(p * num_node)
    node_samples = np.random.choice(np.arange(num_node), number_to_sample, replace=False)
    # get all the edges (members)
    # get non-members
    df_sub = df[df['node1'].isin(node_samples) & df['node2'].isin(node_samples)]
    # mean
    bc_edge = df_sub[df_sub['Label']==1]['bc_edge'].mean()
    cnr = df_sub[df_sub['Label']==1]['cnr'].mean()
    density = len(df_sub) / (number_to_sample * (number_to_sample-1) / 2)
    acc = df_sub['correct'].sum() / len(df_sub)
    group_size = len(df_sub)
    f1 = f1_score(df_sub['Label'], df_sub['Pred'])
    non_member_l2, mem_l2 = df_sub.groupby('Label')['L2-dist'].mean().to_list()
    return [bc_edge, cnr, density, group_size, acc, f1, mem_l2, non_member_l2]


def tprfpr(df, at=0.001):
    # get output
    # Find proper threshold
    fpr, tpr, thresholds = roc_curve(df['Label'], df['Possibility'])
    threshold = thresholds[fpr <= at][-1]
    tprfpr = tpr[fpr <= at][-1]
    tpr_gs = []
    tpr_gs1 = []
    '''for g in [0, 1, 2]:
        sub_df = df.loc[df["Group"] == g]
        f, t, th = roc_curve(sub_df['Label'], sub_df['Possibility'])
        tpr_g = t[th >= threshold][-1]
        tpr_gs.append(tpr_g)
        tpr_g1 = t[f <= at][-1]
        tpr_gs1.append(tpr_g1)
    pld = (max(tpr_gs) - min(tpr_gs)) * 2/3
    pld1 = (max(tpr_gs1) - min(tpr_gs1)) * 2/3'''
    pld=0
    pld1=0

    return tprfpr, pld, pld1

def getF1(df):
    f1_all = f1_score(df['Label'], df['Pred'])
    f1_groups = []
    for g in [0, 1, 2]:
        sub_df = df.loc[df["Group"] == g]
        f1_sub = f1_score(sub_df['Label'], sub_df['Pred'])
        f1_groups.append(f1_sub)
    pld = (max(f1_groups) - min(f1_groups)) * 2/3

    return f1_all, f1_groups, pld


def join_posterior(df, model, dataset):
    pred_file = f"{model}/{dataset}_{model}_pred.pkl"
    pred_vector = pkl.load(open(pred_file, 'rb')).detach().numpy()
    pred_node1 = pred_vector[df['node1'].astype(int).to_numpy()]
    pred_node2 = pred_vector[df['node2'].astype(int).to_numpy()]
    dist = [euclidean(pred_node1[i], pred_node2[i]) for i in range(len(pred_node1))]
    df['L2-dist'] = dist
    df['L2-dist'] = (df['L2-dist'] - df['L2-dist'].min()) / (df['L2-dist'].max() - df['L2-dist'].min())
    return df



def weight_analysis_v3(Graph_name="Facebook-0.2_fair_attack3",
                       model_type="GAT",
                       delta=0.1,
                       run_plot=False):
    weight_names = ["bc_edge"]
    Graph_name = Graph_name.replace('-', '_')
    if "tagged" in Graph_name:
        Graph_name = Graph_name.replace("t=0/", "t=0/K=-1/")
    if "tagged" in Graph_name:
        dataset = "_".join(Graph_name.split("_")[:-3]).lower()
    else:
        dataset = Graph_name.split("_")[0].lower()
    datasource = "{}/".format(model_type)
    weight_file = "{}/weight/{}_all_weights.csv".format(model_type,
                                                        dataset)
    attack_res = datasource + "MIA_res/t=0/{}.csv".format(Graph_name)
    adj_file = datasource + "ind.{}.adj".format(dataset)
    adj = pkl.load(open(adj_file, 'rb')).detach().numpy()
    target_res_file = datasource + "{}_{}_pred.pkl".format(dataset,
                                                             model_type)
    if model_type == "GAT":
        att_file = datasource + "ind.{}.att".format(dataset)
    else:
        att_file = "gcn/weight/{}_edge_influence.dict".format(dataset)
    gender_file = datasource + "ind.{}.gender".format(dataset)
    genders = np.array(pkl.loads(open(gender_file, 'rb').read()))
    min_gender, maj_gender = np.unique(genders, return_counts=True)[0][
        np.unique(genders, return_counts=True)[1].argsort()]

    genders = 1 + (genders == maj_gender)

    weight_df = pd.read_csv(weight_file)
    attack_df = pd.read_csv(attack_res)
    pred_array = pkl.load(open(target_res_file, 'rb')).detach().numpy()
    ft = pkl.load(open(datasource + "ind.{}.allx".format(dataset), 'rb')).detach().numpy()
    densities = get_density(adj, genders)

    all_df = two_files_join(weight_df, attack_df)

    g1 = np.array(all_df['Node1 Gender'])
    g2 = np.array(all_df['Node2 Gender'])
    link_groups = get_link_genders(g1, g2)
    print(np.unique(link_groups, return_counts=True))
    all_df["label(TP FP)"] = np.array(["TN", "FN", "FP", "TP"])[all_df["label(TP FP)"]]

    all_df['Group'] = link_groups
    all_df["correct"] = all_df['Pred'] == all_df['Label']

    all_df = get_cnrs(adj, all_df, dataset=dataset, model_type=model_type)
    random_group_res = []

    #print("TPR@{}FPR : {}".format(0.1, tprfpr(all_df, 0.1)))
    #print("TPR@{}FPR : {}".format(0.05, tprfpr(all_df, 0.05)))
    #print("TPR@{}FPR : {}".format(0.01, tprfpr(all_df, 0.01)))
    print("TPR@{}FPR : {}".format(0.001, tprfpr(all_df, 0.001)))
    print("TPR@{}FPR : {}".format(0.00001, tprfpr(all_df, 0.00001)))
    f1_all, f1_groups, f1_pld = getF1(all_df)
    print("F1 = {}\nF1 Groups are {}\nF1 PLD is {}".format(f1_all, f1_groups, f1_pld))

    all_df = join_posterior(all_df, model_type, dataset)
    for i in tqdm(range(250)):
        for p in [0.05, 0.1, 0.15, 0.2]:
            random_group_res.append(get_group_mean(all_df, p, adj.shape[1]))

    df_random_group = pd.DataFrame(random_group_res, columns=["EBC", "NS", "Density", "Group_size", "MIA Accuracy", 'F1-score', 'mem-l2', 'nme-l2'])
    print(df_random_group.corr())
    df_random_group['m-nm'] = df_random_group['nme-l2'] - df_random_group['mem-l2']
    df_random_group['EBC-Rank'] = df_random_group['EBC'].rank(pct=True)
    df_random_group['EBC-Rank-label'] = df_random_group['EBC-Rank'] > 0.5
    df_group_ebc = df_random_group.groupby('EBC-Rank-label')['m-nm'].mean()
    print(df_group_ebc)
    df_random_group['NS-Rank'] = df_random_group['NS'].rank(pct=True)
    df_random_group['NS-Rank-label'] = df_random_group['NS-Rank'] > 0.5
    df_group_ns = df_random_group.groupby('NS-Rank-label')['m-nm'].mean()
    print(df_group_ns)
    df_random_group.to_csv(f"{model_type}/{dataset}_random_group.csv")




    return 0
    #all_df['cnr'] = all_df['jac_sim']
    agg_group = all_df.groupby(["Group", "Label"]).agg({'bc_edge': "mean",
                                                        'cnr': "mean",
                                                        'correct': "mean",
                                                        'Node1': "count"})
    group_diff = agg_group.query("Label==1").droplevel(1) - agg_group.query("Label==0").droplevel(1)
    group_acc_count = all_df.groupby("Group").agg(Accuracy=("correct", "mean"),
                                                  Count=("Node1", "count"))
    group_weight_diff = pd.merge(group_diff[["bc_edge", "cnr"]], group_acc_count,
                                 left_on="Group", right_on="Group")
    group_weight_diff.to_csv("{}/weight/{}_attack{}_Gap.csv".format(model_type,
                                                                    dataset,
                                                                    Graph_name[-1]))

    agg_member = all_df.groupby(["member"])[weight_names + ['cnr', 'correct']].mean()
    agg_group_member = all_df.groupby(["Group", "Label"])[weight_names + ['cnr', 'correct']].mean()
    agg_member_label = all_df.groupby(["Label", "correct"])[weight_names + [ 'cnr', 'correct']].mean()
    agg_group = all_df.groupby(["Group"])[weight_names + ['cnr', 'correct']].mean()
    agg_group['density'] = 0


    # density
    for group in [0, 1, 2]:
        agg_group.loc[group, 'density'] = densities[group + 1]

    # overall
    row_overall = all_df[weight_names + ['cnr', 'correct']].mean()
    density_all = len(all_df) / len(genders) / (len(genders) - 1) * 2
    row_overall['density'] = density_all
    row_overall = pd.DataFrame(row_overall).T

    agg_group = row_overall.append(agg_group)
    agg_group = agg_group[["bc_edge", "cnr", "density", 'correct']]

    agg_group_tp = all_df.groupby(["Group", "label(TP FP)"]).size()
    agg_group_tp_2g = [agg_group_tp[1] + agg_group_tp[2], agg_group_tp[0]]

    agg_T_rate = pd.DataFrame([[i,
                                agg_group_tp[i, "TP"] / (agg_group_tp[i, 'FN'] + agg_group_tp[i, 'TP']),
                                agg_group_tp[i, "TN"] / (agg_group_tp[i, 'TN'] + agg_group_tp[i, 'FP'])] for i in range(3)],
                              columns=['Group', "TP Rate", "TN Rate"]).set_index('Group')
    agg_T_rate_2group = pd.DataFrame([[0,
                                agg_group_tp_2g[0]["TP"] / (agg_group_tp_2g[0]['FN'] + agg_group_tp_2g[0]['TP']),
                                agg_group_tp_2g[0]["TN"] / (agg_group_tp_2g[0]['TN'] + agg_group_tp_2g[0]['FP'])],
                                      [1,
                                       agg_group_tp_2g[1]["TP"] / (agg_group_tp_2g[1]['FN'] + agg_group_tp_2g[1]['TP']),
                                       agg_group_tp_2g[1]["TN"] / (agg_group_tp_2g[1]['TN'] + agg_group_tp_2g[1]['FP'])] ],
                              columns=['Group', "TP Rate", "TN Rate"]).set_index('Group')


    with pd.ExcelWriter("{}/weight/{}_by_member.xlsx".format(model_type, Graph_name)) as writer:
        agg_group.to_excel(writer, sheet_name='By Group')
        agg_T_rate_2group.to_excel(writer, sheet_name='By 2 Group TF')
        agg_T_rate.to_excel(writer, sheet_name='By Group TF')
        agg_member.to_excel(writer, sheet_name='By Member')
        agg_group_member.to_excel(writer, sheet_name='By Member Group')
        #agg_member_label.to_excel(writer, sheet_name='By Member TF')


    if run_plot:
        dense_pred = pkl.loads(open("dense/" + "%s_dense_pred.pkl" % dataset, "rb").read())
        w_names = get_names(Graph_name.split("-")[0].lower())

        inputs = get_all_MIA(np.array(all_df["Node1"]), np.array(all_df["Node2"]), pred_array, dense_pred, ft)
        ss = StandardScaler()
        inputs = ss.fit_transform(inputs)
        all_df[w_names] = inputs

        rank_accuracy = []
        rank_cnr = []
        MIA_input_list = []
        mem_nm_lists = []
        group_mem_lists = []
        group_nm_lists = []
        mem_ind = all_df["member"] == 1
        nm_ind = all_df["member"] == 0
        groups = all_df["Group"]

        weight_to_plot = weight_names + ["cnr"]
        for w in weight_to_plot:
            weight = all_df[w]
            correct = all_df["correct"]
            cnrs = all_df["cnr"]
            bins = get_interval_by_pct(weight, num_bin=10)
            mem_acc_list = []
            cnr_list = []
            mem_list = []
            group_mem_list = []
            group_nm_list = []
            for i in range(10):
                ind_bin = bins == i
                mem_list.append((ind_bin * mem_ind).sum() / ind_bin.sum())
                mem_acc_list.append(correct[ind_bin * mem_ind].mean())
                cnr_list.append(cnrs[ind_bin].mean())
                MIA_input_list.append(all_df[w_names][ind_bin].mean().to_numpy())
                for g in [0, 1, 2]:
                    g_ind = groups == g
                    group_mem_list.append((ind_bin * mem_ind * g_ind).sum() / (g_ind * mem_ind).sum())
                    group_nm_list.append((ind_bin * nm_ind * g_ind).sum() / (g_ind * nm_ind).sum())
            rank_accuracy.append(mem_acc_list)
            rank_cnr.append(cnr_list)
            mem_nm_lists.append(mem_list)
            group_mem_lists.append(group_mem_list)
            group_nm_lists.append(group_nm_list)
        rank_accuracy = np.array(rank_accuracy)
        rank_cnr = np.array(rank_cnr)
        mem_nm_lists = np.array(mem_nm_lists)
        MIA_input_list = np.vstack(MIA_input_list).T
        group_mem_lists = np.array(group_mem_lists)
        group_nm_lists = np.array(group_nm_lists)

        for i in range(len(weight_to_plot)):
            x = range(10)
            plt.plot(x, rank_accuracy[i], label=weight_to_plot[i])
        plt.legend()
        plt.xticks(x, ["{}%".format(i * 10) for i in x])
        plt.xlabel("Weight Range")
        plt.ylabel("Average Accuracy")
        plt.savefig("GAT/weight/{}_Acc.png".format(Graph_name))
        plt.close()

        for i in range(len(weight_to_plot)):
            x = range(10)
            plt.plot(x, rank_cnr[i], label=weight_to_plot[i])
        plt.legend()
        plt.xticks(x, ["{}%".format(i * 10) for i in x])
        plt.xlabel("Weight Range")
        plt.ylabel("Average $cnr$")
        plt.savefig("GAT/weight/{}_cnr.png".format(Graph_name))
        plt.close()

        for i in range(len(weight_to_plot)):
            x = range(10)
            plt.plot(x, mem_nm_lists[i], label=weight_to_plot[i])
        plt.legend()
        plt.xticks(x, ["{}%".format(i * 10) for i in x])
        plt.xlabel("Weight Range")
        plt.ylabel("Ratio of Members")
        plt.savefig("GAT/weight/{}_ROM.png".format(Graph_name))
        plt.close()

        for i in range(len(weight_to_plot)):
            colors = [["royalblue", "purple", "orange"],
                      ["lightsteelblue", "violet", "moccasin"]]
            x = np.array(range(10))
            for g in [0, 1, 2]:
                mem_g = np.array([group_mem_lists[i][ind * 3 + g] for ind in range(10)])
                non_mem_g = np.array([group_nm_lists[i][ind * 3 + g] for ind in range(10)])
                plt.bar(x + 0.25 * g, non_mem_g, bottom=mem_g, color=colors[1][g],
                        label="{}_Non-Mem{}".format(weight_to_plot[i], g), width=0.25)
                plt.bar(x + 0.25 * g, mem_g, color=colors[0][g],
                        label="{}_Mem{}".format(weight_to_plot[i], g), width=0.25)
            plt.legend()
            plt.xticks(x, ["{}%".format(i * 10) for i in x])
            plt.xlabel("Weight Range")
            plt.ylabel("Distribution of Mem/Non-mem")
            plt.savefig("GAT/weight/{}_{}_group_mem.png".format(Graph_name.split("-")[0], weight_to_plot[i]))
            plt.close()

        for w in range(len(w_names)):
            for i in range(len(weight_to_plot)):
                x = range(10)
                plt.plot(x, MIA_input_list[w][i * 10: (i + 1) * 10], label=weight_to_plot[i])
            plt.legend()
            plt.xticks(x, ["{}%".format(i * 10) for i in x])
            plt.xlabel("Weight Range")
            plt.ylabel("Average {}".format(w_names[w]))
            plt.savefig("GAT/weight/m_input/{}/{}.png".format(Graph_name.split('-')[0].lower(), w_names[w]))
            plt.close()


def load_att_infl(model_type="GAT", att_file="", num_node=0):
    if model_type == "GAT":
        atts = pkl.loads(open(att_file, 'rb').read()).detach().numpy()
    else:
        att_dict = pkl.loads(open(att_file, 'rb').read())
        atts = -np.ones([num_node, num_node])

        for id1, id2 in att_dict.keys():
            atts[id1, id2] = abs(att_dict[(id1, id2)])
            atts[id2, id1] = abs(att_dict[(id1, id2)])
    return atts


def get_names(dataset):
    len_post = 4 if dataset=="facebook" else 2
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
    return w_names


if __name__ == "__main__":

    '''datasets = ["facebook", "facebook", "pokec", "pokec", "tagged_40", "tagged_40"]
    model_types = ["GAT", "gcn", "GAT", "gcn", "GAT", "gcn"]
    deltas = [0.1, 0.05, 0.1, 0.1, 0, 0]
    for i in [2]:
        for Gn in ["{}-0.2_fair_attack3".format(datasets[i]),
                   "{}-0.2_fair_attack6".format(datasets[i])]:
            #weight_analysis(Graph_name=Gn)
            weight_analysis_v2(Graph_name=Gn,
                               model_type=model_types[i],
                               delta=deltas[i],
                               run_plot=False)'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="GAT", help='Model Type, GAT or gcn')
    parser.add_argument('--dataset', type=str, default="facebook", help='dataset, facebook or cora')
    args = parser.parse_args()
    for Gn in ["{}-0.2_fair_attack3".format(args.dataset),
               "{}-0.2_fair_attack6".format(args.dataset)]:
        #weight_analysis(Graph_name=Gn)
        weight_analysis_v3(Graph_name=Gn,
                           model_type=args.model_type)



    pass
