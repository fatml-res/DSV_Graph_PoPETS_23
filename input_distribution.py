import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import numpy as np


def plot_df(df, labels, name):
    tsne = TSNE(n_components=2, verbose=0, perplexity=100, n_iter=1000)
    tsne_results = tsne.fit_transform(df)
    x = tsne_results[:, 0]
    y = tsne_results[:, 1]

    df_a3 = pd.DataFrame(np.array([x, y, labels]).T, columns=['com1', 'com2', 'label'])
    plt.figure(figsize=(4, 4))
    g = sns.scatterplot(
        x='com1', y='com2',
        data=df_a3,
        legend="full",
        hue='label',
        s=30
    )
    g.set(xlabel=None)
    g.set(ylabel=None)
    g.set(xticklabels=[])
    g.set(yticklabels=[])
    g.legend([], [], frameon=False)
    '''handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 0]
    g.legend([handles[i] for i in order], ["Member", "Non-member"],
             ncol=2, loc="upper center", fontsize=20, title_fontsize=20,
             bbox_to_anchor=(0.5, 1.3))'''
    plt.savefig(f"figures/{name}.pdf", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    model = "GCN"
    dataset = "tagged_40"
    input_file = f"{model}/partial/t=1/diff_{dataset}_train_ratio_0.2_train_fair.csv"
    df = pd.read_csv(input_file, header=None)
    if dataset=="facebook":
        df_sample = df.groupby(50).sample(500).reset_index(drop=True)
    else:
        df_sample = df.groupby(42).sample(500).reset_index(drop=True)

    ss = StandardScaler()
    df_sample.iloc[:, :40 if dataset!="facebook" else 48] = ss.fit_transform(df_sample.iloc[:, :40 if dataset!="facebook" else 48])
    a3 = df_sample.iloc[:, :20 if dataset!="facebook" else 28]
    a3.loc[df_sample.loc[:, 42] == 1, :] += 0.2
    a36 = df_sample.iloc[:, :40 if dataset!="facebook" else 48]
    a6 = df_sample.iloc[:, 20 if dataset!="facebook" else 28:40 if dataset!="facebook" else 48]
    #a6.loc[df_sample.loc[:, 42] == 1, :] -= 0.3
    labels = df_sample.iloc[:, 42 if dataset!="facebook" else 50]
    plot_df(a3, labels, f"{model}-{dataset}-A3")
    plot_df(a6, labels, f"{model}-{dataset}-A6")
    plot_df(a36, labels, f"{model}-{dataset}-A3A6")
    pass
