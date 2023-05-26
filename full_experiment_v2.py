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
    parser.add_argument('--model_type', type=str, default="GAT", help='Model Type, GAT or gcn')
    parser.add_argument('--dataset', type=str, default="facebook", help='dataset, facebook or cora')
    parser.add_argument('--datapath', type=str, default="dataset/", help='datapath for original data')
    parser.add_argument('--epoch', type=int, default=300, help='number of epoch')
    parser.add_argument('--gamma', type=float, default=math.inf, help='gamma for M-IN')
    parser.add_argument('--Min', action="store_true", default=False,
                        help='True if M-In experiment is running.')

    parser.add_argument('--run_dense', action="store_true", default=False, help='True if dense experiment is required.')
    parser.add_argument('--run_attack', action="store_true", default=False,
                        help='True if attack experiment is required.')
    parser.add_argument('--run_Target', action="store_true", default=False,
                        help='True if Target experiment is required.')
    parser.add_argument('--run_partial', action="store_true", default=False,
                        help='True if new partial data is required.')
    args = parser.parse_args()

    model_type = args.model_type  # "gcn"
    dataset = args.dataset  # "cora"
    shadow_dataset = args.shadow_dataset  # "cora"
    datapath = args.datapath  # "dataset/"
    epoch = args.epoch  # 300
    Min = args.min
    gamma = args.gamma

    run_dense = args.run_dense  # False
    fair_sample = args.fair_sample  # False
    run_attack = args.run_attack  # False
    run_Target = args.run_Target  # False
    run_partial = args.run_partial  # True

    with open('model_config.json', 'r') as f:
        config = json.load(f)[dataset][model_type]
    adj, ft, gender, labels = load_data(datapath, dataset, dropout=0)

    MIA_res_addon = ""


    if run_dense:
        train_model(gender, ft, adj, labels, dataset, num_epoch=epoch, model_type="dense", saving_path="dense")

    if not Min:
        target_saving_path = model_type
        partial_path = config["partial_path"]
        attack_res_loc = model_type
    else:
        target_saving_path = model_type + "/M_IN/gamma={}".format(gamma)
        partial_path = config["partial_path"] + "M_IN/gamma={}/".format(gamma)
        attack_res_loc = model_type + "/M_IN/gamma={}".format(gamma)

    for locs in [target_saving_path, partial_path, attack_res_loc]:
        if not os.path.exists(locs):
            os.makedirs(locs)

    if run_Target:
        run_target(model_type, config, gender, ft, adj, labels,
                   Min=Min, gamma=gamma,
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
                            dataset=dataset, fair_sample=fair_sample, t=t)
            if not run_attack:
                continue
            print("Start Attack {} with {} balanced sample for {} time.".format(at, fair_sample, t))
            a, p, r, roc, acc_list = attack_main(datapath=config["partial_path"],
                                                 dataset=dataset,
                                                 saving_path=partial_path,
                                                 ratio=0.2,
                                                 attack_type=at,
                                                 fair_sample=fair_sample,
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

    df_acc = pd.DataFrame(df_acc, columns=["Attack",
                                           "Acc_train", "Acc_train1", "Acc_train2", "Acc_train0",
                                           "Acc_test", "Acc_test1", "Acc_test2", "Acc_test0"])
    df_acc_agg = df_acc.groupby("Attack").mean()
    df_agg_performance = pd.DataFrame(agg_all, columns=["Attack Type", "Accuracy", "Precision", "Recall", "AUC"])

    if run_attack:
        df_acc_agg.to_csv(
            "{}/MIA_res/{}_attack{}acc_agg.csv".format(attack_res_loc, dataset, "_fair_" if fair_sample else "_"))
        df_acc.to_csv(
            "{}/MIA_res/{}_attack{}acc.csv".format(attack_res_loc, dataset, "_fair_" if fair_sample else "_"))
        df_agg_performance.to_csv(
            "{}/MIA_res/{}_attack{}performance.csv".format(attack_res_loc, dataset, "_fair_" if fair_sample else "_"))
