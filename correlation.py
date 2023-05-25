import pandas as pd


if __name__ == "__main__":
    corr_file = "correlation/correlation.csv"
    df = pd.read_csv(corr_file)

    corr_df = df.groupby(["Dataset", 'Target Model'])[['attack3', 'attack6', 'EBC', 'NS', 'attention', 'Density']].corr(method='pearson')
    corr_df.to_csv("correlation/corr_res.csv")
    pass