from GAT import run_GAT
from GCN_dense import train_model
from GCN import run_GCN
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
    parser.add_argument('--epoch', type=int, default=300, help='number of epoch')
    parser.add_argument('--run_dense', action="store_true", default=False, help='True if dense experiment is required.')
    parser.add_argument('--fair_sample', action="store_true", default=False, help='True if fair sample for MIA is required')
    parser.add_argument('--run_attack', action="store_true", default=False, help='True if attack experiment is required.')
    parser.add_argument('--prepare_new', action="store_true", default=False, help='True if prepare new attack input files')
    parser.add_argument('--run_Target', action="store_true", default=False, help='True if Target experiment is required.')
    parser.add_argument('--run_partial', action="store_true", default=False, help='True if new partial data is required.')
    parser.add_argument('--DP', action="store_true", default=False,
                        help='True if DP is added to the target model')

    args = parser.parse_args()

    model_type = args.model_type  # "gcn"
    dataset = args.dataset  # "cora"
    shadow_dataset = args.shadow_dataset  # "cora"
    ego_user = args.ego_user  # "107"
    datapath = args.datapath  # "dataset/"
    epoch = args.epoch  # 300

    run_dense = args.run_dense  # False
    fair_sample = args.fair_sample  # False
    run_attack = args.run_attack  # False
    prepare_new = args.prepare_new  # True
    run_Target = args.run_Target  # False
    run_partial = args.run_partial  # True

    if "tagged" in dataset:
        delta = 0
    elif "facebook" in dataset and model_type == "gcn":
        delta = 0.05
    else:
        delta = 0.1
    adj, ft, gender, labels = load_data(datapath, dataset, ego_user, dropout=0)
    ft = torch.randn_like(ft)

    MIA_res_addon = ""
    if delta > 0:
        adj = pkl.load(open("{}/CNR/Group/Reduce/Delta={}/ind.{}.adj".format(model_type, delta, dataset), "rb"))
        MIA_res_addon = "CNR/Group/Reduce/Delta={}/".format(delta)
    if run_dense:
        train_model(gender, ft, adj, labels, dataset, num_epoch=epoch, model_type="dense", saving_path="dense")

    if model_type == "GAT":
        pass
        if run_Target:
            run_GAT(gender, ft, adj, labels, epochs=epoch, dataset=dataset, DP=args.DP)
        df_acc = []
        partial_done = False
        agg_all = []
        for at in [3, 6]:
            a_all, p_all, r_all, roc_all = 0, 0, 0, 0
            for t in range(3):
                if run_partial and not partial_done:
                    get_partial(model_type="gat", datapath="GAT/" + MIA_res_addon, dataset=dataset, fair_sample=fair_sample, t=t)
                if not run_attack:
                    continue
                print("Start Attack {} with {} balanced sample for {} time.".format(at, fair_sample, t))
                a, p, r, roc, acc_list = attack_main(datapath="GAT/" + MIA_res_addon,
                                                     dataset=dataset,
                                                     shadow_data=shadow_dataset,
                                                     ratio=0.2,
                                                     attack_type=at,
                                                     fair_sample=fair_sample,
                                                     t=t,
                                                     prepare_new=prepare_new)
                df_acc.append(acc_list)
                a_all += a
                p_all += p
                r_all += r
                roc_all += roc
            partial_done = True
            print("Average Performance of attack{} on model {} over {} time:\n"
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

    else:
        if run_Target:
            #train_model(gender, ft, adj, labels, dataset, num_epoch=epoch, model_type="gcn")
            run_GCN(gender, ft, adj, labels, epochs=epoch, dataset=dataset, DP=args.DP)
        df_acc = []
        partial_done = False
        agg_all = []
        for at in [3, 6]:
            a_all, p_all, r_all, roc_all = 0, 0, 0, 0
            for t in range(3):
                if run_partial and not partial_done:
                    get_partial(model_type="gcn", datapath="gcn/" + MIA_res_addon, dataset=dataset, fair_sample=fair_sample, t=0)
                if not run_attack:
                    continue
                print("Start Attack {} with {} balanced sample for {} time.".format(at, fair_sample, t))
                a, p, r, roc, acc_list = attack_main(datapath="gcn/" + MIA_res_addon,
                                                     dataset=dataset,
                                                     shadow_data=shadow_dataset,
                                                     ratio=0.2,
                                                     attack_type=at,
                                                     fair_sample=fair_sample,
                                                     t=t,
                                                     prepare_new=prepare_new)
                df_acc.append(acc_list)
                a_all += a
                p_all += p
                r_all += r
                roc_all += roc
                partial_done = True
            print("Average Performance of attack{} on model {} over {} time:\n"
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
    df_acc = pd.DataFrame(df_acc, columns=["Attack", "Acc_train", "Acc_train1", "Acc_train2", "Acc_train0",
                                               "Acc_test", "Acc_test1", "Acc_test2", "Acc_test0"])
    df_acc_agg = df_acc.groupby("Attack").mean()
    df_agg_performance = pd.DataFrame(agg_all, columns=["Attack Type", "Accuracy", "Precision", "Recall", "AUC"])
    if run_attack:
        df_acc_agg.to_csv("{}/MIA_res/{}_attack{}acc_agg.csv".format(model_type, dataset, "_fair_" if fair_sample else "_"))
        df_acc.to_csv(
            "{}/MIA_res/{}_attack{}acc.csv".format(model_type, dataset, "_fair_" if fair_sample else "_"))
        df_agg_performance.to_csv("{}/MIA_res/{}_attack{}performance.csv".format(model_type, dataset, "_fair_" if fair_sample else "_"))
