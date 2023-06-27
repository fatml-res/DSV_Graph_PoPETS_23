import os

from GCN_dense import train_model
from utils import *
from stealing_link.partial_graph_generation import get_partial
from attack import attack_main
import argparse
import json
from run_target import run_target
import math

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="GAT", help='Model Type, GAT or GCN')
    parser.add_argument('--dataset', type=str, default="facebook", help='dataset, facebook or pokec')
    parser.add_argument('--datapath', type=str, default="dataset/", help='datapath for original data')
    parser.add_argument('--epoch', type=int, default=300, help='number of epoch')
    parser.add_argument('--gamma', type=float, default=math.inf, help='gamma for FairDefense')
    parser.add_argument('--FD', action="store_true", default=False,
                        help='True if FairDefense experiment is running.')

    parser.add_argument('--run_dense', action="store_true", default=False, help='True if dense experiment is required.')
    parser.add_argument('--run_attack', action="store_true", default=False,
                        help='True if attack experiment is required.')
    parser.add_argument('--run_Target', action="store_true", default=False,
                        help='True if Target experiment is required.')
    parser.add_argument('--run_partial', action="store_true", default=False,
                        help='True if new partial data is required.')
    args = parser.parse_args()

    model_type = args.model_type  # "GCN"
    dataset = args.dataset  # "facebook
    datapath = args.datapath  # "dataset/"
    epoch = args.epoch  # 300
    FairDefense = args.FD
    gamma = args.gamma

    run_dense = args.run_dense    # False
    run_attack = args.run_attack  # False
    run_Target = args.run_Target  # False
    run_partial = args.run_partial  # True

    with open('model_config.json', 'r') as f:
        config = json.load(f)[dataset][model_type]
    adj, ft, gender, labels = load_data(datapath, dataset, dropout=0)

    MIA_res_addon = ""

    if run_dense:
        train_model(gender, ft, adj, labels, dataset, num_epoch=epoch, model_type="dense", saving_path="dense")

    if not FairDefense:
        target_saving_path = model_type
        partial_path = config["partial_path"]
        attack_res_loc = model_type
    else:
        target_saving_path = model_type + "/FairDefense/gamma={}".format(gamma)
        partial_path = config["partial_path"] + "FairDefense/gamma={}/".format(gamma)
        attack_res_loc = model_type + "/FairDefense/gamma={}".format(gamma)

    for locs in [target_saving_path, partial_path, attack_res_loc]:
        if not os.path.exists(locs):
            os.makedirs(locs)

    if run_Target:
        run_target(model_type, config, gender, ft, adj, labels,
                   FairDefense=FairDefense, gamma=gamma,
                   epochs=epoch, dataset=dataset, saving_path=target_saving_path)

    df_acc = []
    partial_done = False
    agg_all = []
    for at in [3, 6]:
        a_all, p_all, r_all, roc_all = 0, 0, 0, 0
        for t in range(5):
            if run_partial and not partial_done:
                get_partial(adj=adj, model_type=model_type, datapath=config["datapath"], pred_path=target_saving_path,
                            partial_path=partial_path,
                            dataset=dataset, t=t)
            if not run_attack:
                continue
            a, p, r, roc, acc_list = attack_main(datapath=config["partial_path"],
                                                 dataset=dataset,
                                                 saving_path=partial_path,
                                                 ratio=0.2,
                                                 attack_type=at,
                                                 t=t)
            df_acc.append(acc_list)
            a_all += a
            p_all += p
            r_all += r
            roc_all += roc
        partial_done = True
        print("Average Performance of attack {} on model {} over {} time:\n" \
              " Accuracy = {},\n"
              " Precision={},\n"
              " Recall = {},\n"
              " ROC={}".format(at,
                               model_type,
                               t + 1,
                               a_all / (t + 1),
                               p_all / (t + 1),
                               r_all / (t + 1),
                               roc_all / (t + 1)))
        agg_all.append([at, a_all / (t + 1), p_all / (t + 1), r_all / (t + 1), roc_all / (t + 1)])
    df_res = pd.DataFrame(agg_all, columns=['Attack', 'Accuracy', 'Precision', 'Recall', 'ROC'])
    df_res.to_csv(attack_res_loc + f"{dataset}/attack_res.csv", index=False)
    df_acc = pd.DataFrame(df_acc, columns=['ACC_train_all', 'ACC_train_0', 'ACC_train_1', 'ACC_train_2',
                                           'ACC_test_all', 'ACC_test_0', 'ACC_test_1', 'ACC_test_2'])
    df_acc.to_csv(df_acc + f"{dataset}/attack_subgroups.csv", index=False)
