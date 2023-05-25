import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
import numpy as np


def to_percent(y, position):
    s = str(round(100 * y, 2))
    return s + '%'


if __name__ == "__main__":
    '''target_acc_df = pd.read_csv("MIA acc visualization/Target acc.csv")
    mia_acc_df = pd.read_csv("MIA acc visualization/MIA acc.csv")

    df_merge = pd.merge(mia_acc_df, target_acc_df, left_on=["Dataset", "Target Model", "Defense", "defense level"],
                      right_on=["Dataset", "Target Model", "Defense", "defense level"])'''

    df_merge = pd.read_csv("MIA acc visualization/merge_table.csv", index_col=0)
    auc_list = []
    for tm in ["GAT", "GCN"]:
        for acc_type in ["Train", "Test"]:
            for attack in ["A", "B"]:
                for Dataset in ["Facebook", "Pokec", "Spammer"]:
                    df_query = df_merge.query(
                        "`Target Model`=='{}' & `Acc Type`=='{}' & Attack=='{}' & Dataset=='{}'".format(tm, acc_type, attack, Dataset)).copy()
                    g = sns.lineplot(data=df_query, x="Accuracy", y="Acc_test", hue="Defense", marker='o')
                    g.legend(fontsize=12)
                    formatter = FuncFormatter(to_percent)
                    plt.gca().yaxis.set_major_formatter(formatter)
                    plt.gca().xaxis.set_major_formatter(formatter)
                    g.set_xlabel("Target Model Accuracy")
                    g.set_ylabel("Attack Model Accuracy")

                    g.figure.savefig("MIA acc visualization/{}_{}_{}_{}.png".format(tm, acc_type, attack, Dataset))

                    plt.close(g.figure)
                    for dfs in ["Baseline", "M-In", "M-Pre"]:
                        area = 0
                        df_dfs = df_query[df_query["Defense"] == dfs].copy().reset_index(drop=True)
                        max_delta = df_dfs["Acc_test"].max() - min(0.5, df_query["Acc_test"].min())
                        max_T = df_dfs["Accuracy"].max()
                        for i in range(len(df_dfs) - 1):
                            tmp_area = (df_dfs.loc[i, "Accuracy"] + df_dfs.loc[i+1, "Accuracy"]) * (df_dfs.loc[i, "Acc_test"] - df_dfs.loc[i + 1, "Acc_test"]) / 2
                            area += tmp_area
                        area = area/max_delta / max_T
                        if area < 0:
                            print("pause")
                        print("Area for {} is {}".format(dfs, area))
                        auc_list.append([acc_type, tm, attack, Dataset, dfs, area])

    auc_df = pd.DataFrame(np.array(auc_list), columns=["Acc Type", "Target Model", "Attack", "Dataset", "Defense", "AUC"])
    #auc_df.to_csv("MIA acc visualization/auc_res.csv", index=False)
    #PLD AUC

    auc_list = []
    for tm in ["GAT", "GCN"]:
        for acc_type in ["Train", "Test"]:
            for attack in ["A", "B"]:
                for Dataset in ["Facebook", "Pokec", "Spammer"]:
                    df_query = df_merge.query(
                        "`Target Model`=='{}' & `Acc Type`=='{}' & Attack=='{}' & Dataset=='{}'".format(tm, acc_type, attack, Dataset)).copy()
                    g = sns.lineplot(data=df_query, x="Accuracy", y="PLD", hue="Defense", marker='o')
                    g.legend(fontsize=12)
                    formatter = FuncFormatter(to_percent)
                    plt.gca().yaxis.set_major_formatter(formatter)
                    plt.gca().xaxis.set_major_formatter(formatter)
                    g.set_xlabel("Target Model Accuracy")
                    g.set_ylabel("PLD")

                    g.figure.savefig("MIA acc visualization/{}_{}_{}_{}.png".format(tm, acc_type, attack, Dataset))

                    plt.close(g.figure)
                    for dfs in ["Baseline", "M-In", "M-Pre"]:
                        area = 0
                        df_dfs = df_query[df_query["Defense"] == dfs].copy().reset_index(drop=True)
                        max_PLD = (1 - df_dfs["PLD"]).max()
                        max_T = df_dfs["Accuracy"].max()
                        for i in range(len(df_dfs) - 1):
                            tmp_area = (2 - df_dfs.loc[i, "PLD"] - df_dfs.loc[i+1, "PLD"]) * (df_dfs.loc[i, "Accuracy"] - df_dfs.loc[i + 1, "Accuracy"]) / 2
                            area += tmp_area
                        area += (1 - df_dfs.loc[len(df_dfs)-1, "PLD"]) * df_dfs.loc[len(df_dfs)-1, "Accuracy"]
                        area = area/max_PLD / max_T
                        if area < 0:
                            print("pause")
                        print("Area for {} is {}".format(dfs, 1 - area))
                        auc_list.append([acc_type, tm, attack, Dataset, dfs, 1 - area])
    auc_df_pld = pd.DataFrame(np.array(auc_list), columns=["Acc Type", "Target Model", "Attack", "Dataset", "Defense", "AUC"])
    auc_df_pld.to_csv("MIA acc visualization/auc_pld_res.csv", index=False)



    # Barplot
    for tm in ["GAT", "GCN"]:
        for attack in ["A", "B"]:
            for Dataset in ["Spammer"]:
                df_query = mia_acc_df.query(
                        "`Target Model`=='{}' & Attack=='{}' & Dataset=='{}' & (`defense level`==0 | `defense level`==5)".format(tm,
                                                                                     attack,
                                                                                     Dataset)).drop("Acc_test", axis=1).copy()
                df_query.loc[df_query["defense level"] == 0, 'Defense'] = "Original"
                df_query.columns = ['Dataset', 'Target Model', 'Defense', 'Attack', '$G_0$', '$G_1$', '$G_2$', 'defense level']
                df_melt = df_query.melt(id_vars=["Dataset", "Target Model", "Defense", "defense level", "Attack"])
                g = sns.barplot(x="Defense", y="value", hue="variable", data=df_melt)
                if Dataset=="Spammer":
                    g.legend(fontsize=20, bbox_to_anchor=(0.95, 1.18), ncol=3)
                else:
                    plt.legend([],[], frameon=False)

                g.set_yticklabels(g.get_yticks(), size=15)
                g.set_xticklabels(g.get_xticklabels(), size=15)

                formatter = FuncFormatter(to_percent)
                plt.gca().yaxis.set_major_formatter(formatter)
                g.set_xlabel("Mitigation Method", fontsize=15)
                g.set_ylabel("Attack Model Accuracy", fontsize=15)

                plt.subplots_adjust(left=0.15, right=0.99)

                g.figure.savefig("MIA acc visualization/Bar_{}_{}_{}_{}.png".format(tm, acc_type, attack, Dataset))

                plt.close(g.figure)





        '''
    target_acc_df = pd.read_csv("MIA acc visualization/Target acc.csv")
    target_acc_df['D_A'] = target_acc_df['Acc Type'].astype(str) + '-' + target_acc_df['Defense']
for tm in ["GAT", "GCN"]:
        g = sns.FacetGrid(target_acc_df[target_acc_df["Target Model"]==tm], col="Dataset", margin_titles=True)
        g.map(sns.lineplot, "defense level", "Accuracy",
          hue=target_acc_df["Acc Type"], style=target_acc_df["Defense"], markers=True)
        g.add_legend(fontsize=12)
        formatter = FuncFormatter(to_percent)
        plt.gca().yaxis.set_major_formatter(formatter)

        # g.set_titles(size=20)
        g.set_xlabels("Defense Level")
        g.set_ylabels("Target Accuracy")
        # g.set_yticklabels(fontsize=25)
        # g.set_xticklabels(labels=[0, 1, 2, 3, 4, 5], step=1)
        for ax in g.axes.flat:
            ax.xaxis.set_major_locator(ticker.FixedLocator(list(range(6))))
            ax.xaxis.set_major_formatter(ticker.FixedFormatter(['Original'] + list(range(1, 6))))
        [plt.setp(ax.texts, text="") for ax in g.axes.flat]
        g.set_titles(row_template='{row_name}', col_template='{col_name}')
        g.savefig("MIA acc visualization/Target acc_{}.png".format(tm))

    mia_acc_df = pd.read_csv("MIA acc visualization/MIA acc.csv")
    mia_acc_df['D_A'] = "Attack" + mia_acc_df['Attack'].astype(str) + '-' + mia_acc_df['Defense']
    mia_acc_df['Attack'] = "Attack " + mia_acc_df['Attack'].astype(str)
    g1 = sns.FacetGrid(mia_acc_df, col="Target Model", row="Dataset", margin_titles=True)
    g1.map(sns.lineplot, "defense level", "Acc_test", hue=mia_acc_df["Attack"], style=mia_acc_df["Defense"], markers=True)
    g1.add_legend()
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    # g.set_titles(size=20)
    g1.set_xlabels("Defense Level")
    g1.set_ylabels("MIA Accuracy")
    # g.set_yticklabels(fontsize=25)
    for ax in g1.axes.flat:
        ax.xaxis.set_major_locator(ticker.FixedLocator(list(range(6))))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(['Original'] + list(range(1, 6))))
    [plt.setp(ax.texts, text="") for ax in g1.axes.flat]
    g1.set_titles(row_template='{row_name}', col_template='{col_name}')
    g1.savefig("MIA acc visualization/MIA acc.png")'''



    pass