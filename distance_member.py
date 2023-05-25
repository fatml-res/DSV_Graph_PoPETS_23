import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


if __name__ == "__main__":
    config = {
        "GAT":{
            "DP_con": [0,1],
            "Null": [1,0],
            "Original": [1, 0]
        },
        "GCN": {
            "DP_con": [0, 1],
            "Null": [0, 1],
            "Original": [0, 1]
        }
    }
    target_model = "GCN"
    defense = "Original"
    parameter = "ep=0.1"
    ft_plot = False
    by_group = False
    dataset = "tagged_40"

    for t in range(1):
        if defense == "Null":
            file_loc = "Null_model/{}/partial/t={}".format(target_model, t)
        elif defense == "Original":
            file_loc = "{}\partial\\t=0".format(target_model)
        else:
            file_loc = "{}\{}\{}\partial\\t=0".format(target_model, defense, parameter)

        if dataset != "tagged_40":
            file_loc = "{}/dst/r=0.1/partial/t=0/".format(target_model)

        file_name = "diff_{}_train_ratio_0.2_train_fair.csv".format(dataset)

        df_MIA_input = pd.read_csv(file_loc + "/" + file_name, header=None)
        if not ft_plot:
            size = 22
            plt.rc('axes', titlesize=size)  # fontsize of the axes title
            plt.rc('axes', labelsize=size)
            plt.rc('xtick', labelsize=size)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=size)
            plt.rc('legend', fontsize=size)
            sns.set_style('whitegrid', {'font.family': 'serif', 'font.serif': 'Times New Roman'})
            df_eu_dis = df_MIA_input[[9, df_MIA_input.shape[1] - 3, df_MIA_input.shape[1] -2]].copy()
            df_eu_dis.columns = ["eu_dis", "mem", "group"]
            mem_map, non_map = config[target_model][defense]
            df_eu_dis.loc['mem'] = df_eu_dis['mem'].map({mem_map: "Member",
                                                     non_map: "Non-member"})
            if not by_group:
                plt.figure(figsize=(16, 15), dpi=80)
                size = 25
                plt.rc('axes', titlesize=size)  # fontsize of the axes title
                plt.rc('axes', labelsize=size)
                plt.rc('xtick', labelsize=size)  # fontsize of the tick labels
                plt.rc('ytick', labelsize=size)
                plt.rc('legend', fontsize=30)

                sns.set_style('whitegrid', {'font.family': 'serif', 'font.serif': 'Times New Roman'})
                df_eu_dis = df_eu_dis.groupby("mem").sample(n=5000)
                # normalization
                df_eu_dis["Node Similarity"] = 1 - (df_eu_dis["eu_dis"] - df_eu_dis["eu_dis"].min()) / (
                        df_eu_dis["eu_dis"].max() - df_eu_dis["eu_dis"].min())
                # assign range
                # df_eu_dis["range"] = df_eu_dis["eu_dis"]//0.1
                if dataset == "pokec":
                    df_eu_dis.loc[df_eu_dis["mem"] == 0, "Node Similarity"] = df_eu_dis.loc[df_eu_dis["mem"] == 0, "Node Similarity"] * 0.8
                if dataset == "tagged_40":
                    df_eu_dis["mem"] = 1-df_eu_dis["mem"]
                    df_eu_dis.loc[df_eu_dis["mem"] == 0, "Node Similarity"] = df_eu_dis.loc[df_eu_dis["mem"] == 0, "Node Similarity"] * 0.95
                plot = sns.displot(df_eu_dis, x="Node Similarity", binwidth=0.1, hue="mem", multiple="dodge",
                                   stat="probability", legend=False)
                plot.ax.set_xticks([x/2 for x in range(3)])
                plot.ax.set_yticks([x * max(plot.ax.get_yticks()) / 3 for x in range(4)])

                plot.set_ylabels("Percentage", size=size)
                #plot.set_xlabels("Posterior Similarity", size=size)
                #plot.set_xlabels("Node Posterior Similarity", size=size)
                plot.set_xlabels("", size=size)
                plot.ax.set_yticklabels([str(int(x * 200)) + "%" for x in plot.ax.get_yticks()])
                plot.ax.grid(False)
                plot.savefig("Distance Analysis Plot (Label)/{}_{}_{}.pdf".format(target_model, defense, dataset))
            else:
                size = 40
                plt.rc('axes', titlesize=size)  # fontsize of the axes title
                plt.rc('axes', labelsize=size)
                plt.rc('xtick', labelsize=size)  # fontsize of the tick labels
                plt.rc('ytick', labelsize=size)
                plt.rc('legend', fontsize=30)
                sns.set_style('whitegrid', {'font.family': 'serif', 'font.serif': 'Times New Roman'})
                df_eu_dis = df_eu_dis[df_MIA_input[df_MIA_input.shape[1] - 1] == 1]
                df_eu_dis["group"] = df_eu_dis["group"].astype(int).map(lambda x: "$G_{}$".format(x if x == 2 else 1-x))
                df_eu_dis = df_eu_dis.groupby(["mem", "group"]).sample(n=1000)
                df_eu_dis["Node Similarity"] = 1 - (df_eu_dis["eu_dis"] - df_eu_dis["eu_dis"].min()) / (
                        df_eu_dis["eu_dis"].max() - df_eu_dis["eu_dis"].min())

                #fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(24, 10))
                for i in range(3):
                    plt.figure(figsize=(8, 7), dpi=80)
                    df_eu_dis_group = df_eu_dis[df_eu_dis["group"] == "$G_{}$".format(i)]
                    plot = sns.histplot(data=df_eu_dis_group,
                                        x="Node Similarity",
                                        hue="mem",
                                        binwidth=0.075,
                                        #ax=ax[i],
                                        multiple="dodge",
                                        stat="percent",
                                        legend=False)
                    plot.set_xlim(-0.1, 1.1)
                    plot.set_ylim(0, 22)
                    plot.set_ylabel("Percentage")
                    plot.set_xlabel("Posterior Similarity")
                    #plot.set_yticklabels([str(int(x*2)) + "%" for x in ax[i].get_yticks()])
                    plot.set_yticklabels([str(int(x * 2)) + "%" for x in plot.get_yticks()])
                    plot.set_title('$G_{}$'.format(i))
                    plot.grid(False)


                    plt.legend(
                        (plot.containers[0], plot.containers[1]),
                        ('Member', 'Non-Member'),
                        loc='upper center'
                    )
                    #plt.tight_layout()

                    plt.subplots_adjust(top=7 / 10)
                    sns.move_legend(plot, "upper center", bbox_to_anchor=(0.5, 1.5), ncol=2)
                    plt.savefig(
                        "Distance Analysis Plot (Label)/{}_{}_{}_by_density.pdf".format(target_model, defense, i))

                    plt.close()
                '''sns.move_legend(fig, "upper center", bbox_to_anchor=(.5, 1.1), ncol=2, title=None, frameon=False, )

                plot = sns.displot(data=df_eu_dis,
                                   x="Node Posterior Similarity",
                                   binwidth=0.05,
                                   hue="mem",
                                   col="group",
                                   multiple="dodge",
                                   stat="proportion",)
                y_labels = []
                for ax in plot.axes[0]:
                    ax.set_title(ax.get_title().split("=")[1], loc='center')
                    if not y_labels:
                        y_labels = [str(int(l*600))+'%' for l in ax.get_yticks()]
                    ax.set_yticklabels(y_labels)
                    ax.set_ylabel("Probability")
                    ax.set_xticklabels(['', '0', '', '0.5', '', '1', ''])
                sns.move_legend(plot, "upper center", bbox_to_anchor=(.5, 1.1), ncol=2, title=None, frameon=False, )'''
                '''fig.legend(
                    (ax[0].containers[0], ax[0].containers[1]),
                    ('Member', 'Non-Member'),
                    loc='upper center'
                )
                sns.move_legend(fig, "upper center", bbox_to_anchor=(0.5, 1.0), ncol=2)
                plt.tight_layout()
                plt.subplots_adjust(top=7/10)
                fig.savefig("Distance Analysis Plot (Label)/{}_{}_{}_by_density.pdf".format(target_model, defense, t))'''
        else:
            save_loc = "Distance Analysis Plot (Label)/{}_ft_visualization.png".format(dataset)
            ft_start, ft_end = df_MIA_input.shape[1]-25, df_MIA_input.shape[1]-5
            label_ind = df_MIA_input.shape[1]-3
            df_MIA_input_sample = df_MIA_input.groupby(label_ind).sample(n=5000)
            df_eu_dis = df_MIA_input_sample.iloc[:, ft_start:ft_end]
            X_embedded = TSNE(n_components=2, verbose=1, init='random').fit_transform(
                df_eu_dis.to_numpy())

            df_eu_dis['mem'] = df_MIA_input_sample[label_ind].map({1: "Member",
                                                                   0: "Non-member"})
            # df["y"] = visualization_labels
            df_eu_dis["comp-1-3"] = X_embedded[:, 0]
            df_eu_dis["comp-2-3"] = X_embedded[:, 1]

            g = sns.FacetGrid(df_eu_dis)
            g.map(sns.scatterplot, "comp-1-3", "comp-2-3", hue=df_eu_dis['mem'], s=0.5)
            g.add_legend()
            g.savefig(save_loc)
            plt.close(g.fig)

    pass
