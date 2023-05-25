import json
from utils import *


if __name__ == "__main__":
    dataset = 'facebook'
    model_type = "GAT"


    with open('model_config.json', 'r') as f:
        config = json.load(f)[dataset][model_type]
    delta = config["delta"]
    adj, ft, gender, labels = load_data(datapath_str="datapath/", dataset_str=dataset, ego="107", dropout=0)
    if delta > 0:
        adj = pkl.load(open(config["adj_location"], "rb"))

    # load ft name
    ft_name = pkl.load("dataset/facebook/107.featnames", encoding='latin1')
    pass