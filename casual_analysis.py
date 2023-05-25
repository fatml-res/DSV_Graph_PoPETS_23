import pandas as pd


def measur_causal(model, dataset, metric):
    print(f"\nResult from {dataset}-{model}-{metric}:\n")
    if len(dataset):
        file = f"sample-graph/{dataset}_Attack-sub-{model}.csv"
    else:
        file = f"sample-graph/Attack-sub-{model}.csv"
    df = pd.read_csv(file)
    df[f"{metric}-label"] = df[metric] > df[metric].mean()
    df["ANS-label"] = df["ANS"] > df["ANS"].mean()
    df["AEBC-label"] = df["AEBC"] > df["AEBC"].mean()

    print(f"Result For density:")
    den_group = df.groupby(["Attack", "Density-label"])[f"{metric}-label"].mean()
    print(f"Attack 3: {den_group.loc[3, True] - den_group.loc[3, False]}")
    print(f"Attack 6: {den_group.loc[6, True] - den_group.loc[6, False]}")

    print(f"Result For AEBC:")
    abec_group = df.groupby(["Attack", "AEBC-label"])[f"{metric}-label"].mean()
    print(f"Attack 3: {abec_group.loc[3, True] - abec_group.loc[3, False]}")
    print(f"Attack 6: {abec_group.loc[6, True] - abec_group.loc[6, False]}")

    print(f"Result For ANS:")
    ans_group = df.groupby(["Attack", "ANS-label"])[f"{metric}-label"].mean()
    print(f"Attack 3: {ans_group.loc[3, True] - ans_group.loc[3, False]}")
    print(f"Attack 6: {ans_group.loc[6, True] - ans_group.loc[6, False]}")


if __name__ == "__main__":
    for dataset in ["", "pokec", "tagged_40"]:
        for model in ["GAT", "GCN"]:
            for metric in ["Acc", "F1"]:
                measur_causal(model, dataset, metric)
