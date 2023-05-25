from utils import *
from stealing_link.partial_graph_generation import get_partial
from attack import attack_main
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="GAT", help='Model Type, GAT or gcn')
    parser.add_argument('--dataset', type=str, default="facebook", help='dataset, facebook or cora')
    parser.add_argument('--shadow_dataset', type=str, default="cora", help='shadow dataset, cora')
    parser.add_argument('--ego_user', type=str, default="107", help='ego user of target dataset')
    parser.add_argument('--datapath', type=str, default="dataset/", help='datapath for original data')

    parser.add_argument('--fair_sample', action="store_true", default=False, help='True if fair sample for MIA is required')
    parser.add_argument('--run_attack', action="store_true", default=False, help='True if attack experiment is required.')
    parser.add_argument('--prepare_new', action="store_true", default=False, help='True if prepare new attack input files')
    parser.add_argument('--run_partial', action="store_true", default=False, help='True if new partial data is required.')
    parser.add_argument('--betas', type=float, nargs='+', default=[0.1, 0.5, 1.0, 3.0, 5.0], help="Beta for Laplace noise")
    parser.add_argument('--att', type=int, nargs='+', default=[3, 6], help="Attack Types")
    parser.add_argument('--top_k', type=int, default=-1, help="top k output only, -1 if complete")

    args = parser.parse_args()

    model_type = args.model_type  # "gcn"
    dataset = args.dataset  # "cora"
    shadow_dataset = args.shadow_dataset  # "cora"
    ego_user = args.ego_user
    fair_sample = args.fair_sample  # False
    run_attack = args.run_attack  # False
    prepare_new = args.prepare_new  # True
    run_partial = args.run_partial  # True
    Beta_list = args.betas
    att_list = args.att
    top_k = args.top_k # -1

    att_res = []
    att_acc_res = []
    for beta in Beta_list:
        for t in range(3):
            datapath = model_type + "/DP/beta={}/t={}/".format(beta, t)
            if not os.path.exists(datapath):
                continue
            partial_done = False
            for at in att_list:
                if run_partial and not partial_done:
                    get_partial(model_type=model_type.lower(), datapath=datapath,
                                dataset=dataset, fair_sample=fair_sample, t=0, top_k=top_k)
                    partial_done = True
                if not run_attack:
                    continue
                for time_a in range(5):
                    print("Start Attack {} with {} balanced sample for {} time.".format(at, fair_sample, t))
                    a, p, r, roc, acc_list = attack_main(datapath=datapath,
                                                         dataset=dataset,
                                                         shadow_data=shadow_dataset,
                                                         ratio=0.2,
                                                         attack_type=at,
                                                         fair_sample=fair_sample,
                                                         t=time_a,
                                                         prepare_new=prepare_new,
                                                         top_k=top_k)
                    print("Performance of attack{} on model {} with beta={}:\n"
                          " Accuracy = {},\n"
                          " Precision={},\n"
                          " Recall = {},\n"
                          " ROC={}".format(at, model_type, beta, a, p, r, roc))
                    att_res.append([beta, at, a, p, r, roc])
                    att_acc_res.append([beta] + acc_list)
    df_acc = pd.DataFrame(att_acc_res, columns=["Beta", "Attack", "Acc_train", "Acc_train1", "Acc_train2",
                                                "Acc_train0", "Acc_test", "Acc_test1", "Acc_test2", "Acc_test0"])
    df_acc_agg = df_acc.groupby(["Attack", "Beta"]).mean()

    df_performance = pd.DataFrame(att_res, columns=["Beta", "Attack", "Accuracy", "Precsion", "Recall", "ROC"])
    df_agg_performance = df_performance.groupby(["Attack", "Beta"]).mean()
    if run_attack:
        df_acc_agg.to_csv("{}/DP/{}_{}_attack_acc_agg.csv".format(model_type, dataset, "K={}_".format(top_k) if top_k > 0 else ""))
        df_acc.to_csv(
            "{}/DP/{}_{}_attack_acc.csv".format(model_type, dataset, "K={}_".format(top_k) if top_k > 0 else ""))
        df_agg_performance.to_csv("{}/DP/{}_{}attack_performance.csv".format(model_type, dataset,
                                                                             "K={}_".format(top_k) if top_k > 0 else ""))
