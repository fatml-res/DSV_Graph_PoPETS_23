from attack import operator_func, get_metrics
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine, euclidean, correlation, chebyshev,\
    braycurtis, canberra, cityblock, sqeuclidean
import numpy as np
import pickle as pkl
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import os
import matplotlib.pyplot as plt
from weight_analysis import get_cnrs


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


def two_files_join(wdf, adf):
    ori_columns = adf.columns
    new_columns = list(ori_columns) + ['bc_edge']
    res_df = pd.merge(left=wdf, right=adf,
                      left_on=["node1", "node2"], right_on=['Node1', 'Node2'],
                      how="inner")
    return res_df.drop("Unnamed: 0", axis=1)[new_columns]


def customized_plot(df_sample, color_option, saving_file3):
    df_sample_label = df_sample[df_sample['Label'] == 1]
    df_sample_label0 = df_sample[df_sample['Label'] == 0]

    df_sample["Label"] = np.where(df_sample["Label"] == 1, "Member", "Non-member")
    df_sample["label(TP FP)"] = np.array(["TN", "FN", "FP", "TP"])[df_sample["label(TP FP)"]]
    #df_sample["Group"] = "$G_" + df_sample["Group"].astype("string") +"$"
    if color_option == 'cnr':
        g = sns.FacetGrid(df_sample_label)
        g.map(sns.scatterplot, "comp-1-3", "comp-2-3", hue=df_sample_label[color_option],
              palette=sns.color_palette("flare", as_cmap=True), style=df_sample_label['Pred'])
        g.add_legend()
        g.savefig(saving_file3)
        plt.close(g.fig)

        g = sns.FacetGrid(df_sample_label)
        g.map(sns.scatterplot, "comp-1-6", "comp-2-6", hue=df_sample_label[color_option],
              palette=sns.color_palette("flare", as_cmap=True), style=df_sample_label['Pred'])
        g.add_legend()
        g.savefig(saving_file3.replace('attack3', 'attack6'))
        plt.close(g.fig)
    elif color_option == 'bc_edge':

        df_sample_label.loc[:, "HL"] = np.where(df_sample_label[color_option] > df_sample_label[color_option].median(), "High", "Low")
        g = sns.FacetGrid(df_sample_label)
        g.map(sns.scatterplot, "comp-1-3", "comp-2-3", hue=df_sample_label["HL"],
              style=df_sample_label['Pred'])
        g.add_legend()
        g.savefig(saving_file3)
        plt.close(g.fig)

        g = sns.FacetGrid(df_sample_label)
        g.map(sns.scatterplot, "comp-1-6", "comp-2-6", hue=np.log(df_sample_label[color_option]),
              palette=sns.color_palette("flare", as_cmap=True), style=df_sample_label['Pred'])
        g.add_legend()
        g.savefig(saving_file3.replace('attack3', 'attack6'))
        plt.close(g.fig)
    elif color_option == "Pred":
        g = sns.FacetGrid(df_sample, col="Group", hue="label(TP FP)")
        g.map(sns.scatterplot, "comp-1-3", "comp-2-3", alpha=0.5)
        g.add_legend()
        #plt.setp(g.legend.get_texts(), fontsize=15)
        for lh in g._legend.legendHandles:
            lh._sizes = [225]
        if "GAT_facebook_attack3" in saving_file3:
            for ax in g.axes.flat:
                #ax.plot([0, 0], [-60, 60], '--', lw=2, color='red')
                ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
        if "gcn_pokec_attack3" in saving_file3:
            for ax in g.axes.flat:
                ax.plot([-40, 0, -20], [40, 0, -40], '--', lw=2, color='red')

        if "tagged_40_attack3" in saving_file3:
            for ax in g.axes.flat:
                #ax.plot([-25, 50], [40, -40], '--', lw=2, color='red')
                ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
        g.set_titles(row_template='{row_name}', col_template='$G_{col_name}$')
        g.tight_layout()
        g.savefig(saving_file3, dpi=250)
        plt.close(g.fig)
    else:
        for i in range(3):
            df_sample_i = df_sample[df_sample["Group"] == i]
            g = sns.scatterplot(data=df_sample_i, x="comp-1-3", y="comp-2-3", hue="label(TP FP)", alpha=1.0)
            if i == 1:
                g.legend(loc=2, bbox_to_anchor=(-0.015, 1.15), ncol=4, fontsize=15)
                g.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
            g.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
            plt.subplots_adjust(left=0.01, right=0.99)
            g.figure.savefig(saving_file3.replace(".png", "_G{}.png".format(i)))
            plt.close(g.figure)



'''
        g = sns.FacetGrid(df_sample_label, col="Group", hue=color_option)
        g.map(sns.scatterplot, "comp-1-6", "comp-2-6", alpha=0.5)
        g.add_legend()
        g.savefig(saving_file3.replace('attack3', 'attack6'))
        plt.close(g.fig)

        g = sns.FacetGrid(df_sample_label0, col="Group", hue=color_option)
        g.map(sns.scatterplot, "comp-1-3", "comp-2-3", alpha=0.5)
        g.add_legend()
        if "GAT_facebook_attack3" in saving_file3:
            for ax in g.axes.flat:
                ax.plot([0, 0], [-60, 60], '--', lw=2, color='red')
        if "gcn_pokec_attack3" in saving_file3:
            for ax in g.axes.flat:
                ax.plot([-40, 0, -20], [40, 0, -40], '--', lw=2, color='red')
        if "gcn_tagged_40_attack3" in saving_file3:
            for ax in g.axes.flat:
                ax.plot([-25, 50], [40, -40], '--', lw=2, color='red')
                ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
        g.savefig(saving_file3.replace('.png', 'label0.png'))
        plt.close(g.fig)

        g = sns.FacetGrid(df_sample_label0, col="Group", hue=color_option)
        g.map(sns.scatterplot, "comp-1-6", "comp-2-6", alpha=0.5)
        g.add_legend()
        g.savefig(saving_file3.replace('attack3', 'attack6').replace('.png', 'label0.png'))
        plt.close(g.fig)'''


def plot_MIA_input(dataset, model_type, delta, add_on="", color_option="Pred", remap=False):
    saving_path = "Distance Analysis Plot ({})".format(color_option)
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    para = ""
    if len(add_on) > 0:
        saving_path += '/' + add_on.split("/")[0]
        para = '_' + add_on.split("/")[1].split('=')[1]

    if remap:
        if delta > 0 and len(add_on) == 0:
            MIA_loc = "{}/CNR/Group/Reduce/Delta={}".format(model_type, delta) + "/" + add_on
            MIA_file = "{}/MIA_res/t=0/{}_0.2_fair_attack3.csv".format(MIA_loc, dataset)
        else:
            MIA_loc = model_type + "/" + add_on
            MIA_file = "{}/MIA_res/t=0/K=-1/{}_0.2_fair_attack3.csv".format(MIA_loc, dataset)

        try:
            pred_array = pkl.loads(open("{}/{}_{}_pred.pkl".format(MIA_loc,
                                                                   dataset, model_type.lower()),
                                        "rb").read()).detach().numpy()

            ft = pkl.loads(open("{}/ind.{}.allx".format(MIA_loc,
                                                        dataset), "rb").read()).detach().numpy()
            all_df = pd.read_csv(MIA_file)
        except:
            print("Skip setting: {} for dataset: {}, model_type: {}".format(add_on.split('/')[1], dataset, model_type))
            return
        all_df["Group"] = all_df["Node1 Gender"] * (all_df["Node1 Gender"] == all_df["Node2 Gender"])

        w_names = get_names(dataset)
        dense_pred = pkl.loads(open("dense/" + "%s_dense_pred.pkl" % dataset, "rb").read())

        inputs = get_all_MIA(np.array(all_df["Node1"]), np.array(all_df["Node2"]), pred_array, dense_pred, ft)
        ss = StandardScaler()
        inputs = ss.fit_transform(inputs)
        all_df[w_names] = inputs
        df_average_by_group = all_df.groupby(["Label", "Group"])[w_names].mean()
        df_average_by_group_pred = all_df.groupby(["Label", "Group", "Pred"])[w_names].mean()
        # dis_columns = [name for name in w_names if
        #                   ('[' not in name or 'l1' in name or 'l2' in name) and 'mul' not in name and 'avg' not in name]
        # dis_columns3 = [name for name in dis_columns if 'ref' not in name and 'feat' not in name]
        dis_columns = w_names
        dis_columns3 = [name for name in w_names if 'ref' not in name and 'feat' not in name]
        # dis_attack3 = df_average_by_group[dis_columns3]
        # dis_lgp = df_average_by_group_pred[dis_columns3]
        adj_file = MIA_loc + "/ind.{}.adj".format(dataset)
        adj = pkl.load(open(adj_file, 'rb')).detach().numpy()
        all_df = get_cnrs(adj, all_df, dataset=dataset, model_type=model_type)

        weight_file = "{}/weight/{}_all_weights.csv".format(model_type,
                                                            dataset)
        weight_df = pd.read_csv(weight_file)

        all_df = two_files_join(weight_df, all_df)

        # sample 400 from each group
        index_list = []
        edge_dict = pkl.load(open("Kun/gat_edge_influence_{}_edge900.dict".format(dataset), 'rb'))
        values = []
        for i, j in edge_dict.keys():
            try:
                index = int(np.where((all_df['Node1']==i) & (all_df['Node2']==j))[0])
                index_list.append(index)
                values.append(edge_dict[(i, j)])
            except:
                print("Skip edge [{}]".format((i, j)))
                values.append(0)

        df_sample500 = all_df.iloc[index_list].copy()
        df_sample500['memory'] = values
        df_sample500['memory'] = (df_sample500['memory'] - df_sample500['memory'].min())/(df_sample500['memory'].max() - df_sample500['memory'].min())
        acc_500 = len(np.where(df_sample500['Pred']==df_sample500['Label'])[0])/len(df_sample500)
        df_true = df_sample500.query("Pred==Label")
        accs = df_true.groupby("Group").size()/300
        scores = df_sample500.groupby("Group")['memory'].mean()

        if color_option == "Pred":
            pass
        elif color_option == "cnr":
            pass
        elif color_option == "bc_edge":
            weight_file = "{}/weight/{}_all_weights.csv".format(model_type,
                                                                dataset)
            weight_df = pd.read_csv(weight_file)
        else:
            print("Error! Unexpected color option as {}!".format(color_option))
        print("Generating new mapping...")
        if color_option == "Label":
            df_sample = all_df.groupby(["Group"]).sample(n=800, replace=True).fillna(0).reset_index()
        else:
            df_sample = all_df.groupby(["Label", "Group"]).sample(n=400, replace=True).fillna(0).reset_index()
        X_embedded = TSNE(n_components=2, verbose=1, init='random').fit_transform(df_sample[dis_columns3].to_numpy())
        # df["y"] = visualization_labels
        df_sample["comp-1-3"] = X_embedded[:, 0]
        df_sample["comp-2-3"] = X_embedded[:, 1]

        # visualization using tsne: attack 6
        X_embedded = TSNE(n_components=2, verbose=1, init='random').fit_transform(df_sample[dis_columns].to_numpy())
        # df["y"] = visualization_labels
        df_sample["comp-1-6"] = X_embedded[:, 0]
        df_sample["comp-2-6"] = X_embedded[:, 1]
        df_sample.to_csv(saving_path + "/{}_{}_attack3{}.csv".format(model_type, dataset, para))
    else:
        print("Reloading previous mapping...")
        df_sample = pd.read_csv(saving_path + "/{}_{}_attack3{}.csv".format(model_type, dataset, para))

    savingfile3 = saving_path + "/{}_{}_attack3{}.png".format(model_type, dataset, para)
    customized_plot(df_sample, color_option, savingfile3)


if __name__ == "__main__":
    datasets = ["facebook", "facebook", "pokec", "pokec", "tagged_40", "tagged_40"]
    model_types = ["GAT", "gcn", "GAT", "gcn", "GAT", "gcn"]
    deltas = [0.1, 0.05, 0.1, 0.1, 0, 0]
    color_option = "Label"
    for i in [1]:
        pass
        plot_MIA_input(datasets[i], model_types[i], deltas[i], color_option=color_option, remap=True)

    for i in [4, 5]:
        for beta in [5.0, 3.0, 1.0, 0.5, 0.1, 0.01, 0.001]:
            add_on = "DP/beta={}/t=0".format(beta)
            #plot_MIA_input(datasets[i], model_types[i], 0, add_on=add_on)

    for i in [2, 5]:
        for gamma in [10.0, 5.0, 3.0, 1.0, 0.5, 0.1]:
            add_on = "ptb_adj/gamma={}/t=0".format(gamma)
            #plot_MIA_input(datasets[i], model_types[i], 0, add_on=add_on)

