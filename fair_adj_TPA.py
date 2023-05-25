from utils import *
from GAT import run_GAT
from GCN import run_GCN
from attack import attack_main
from stealing_link.partial_graph_generation import get_partial

if __name__ == "__main__":
    datapath = "dataset/"
    dataset = "pokec"
    ego_user = "107"
    model_type = "GAT"
    fair_sample = True
    prepare_new = True
    run_attack = True
    run_partial = True
    run_Target = True
    _, ft, gender, labels = load_data(datapath, dataset, ego_user, dropout=0)

    adj = pkl.load(open("{}/fair_adj/ind.{}.adj".format(model_type, dataset), "rb"))
    acc_lists = []
    if run_Target:
        if model_type == "GAT":
            tmp_list = run_GAT(gender, ft, adj, labels, epochs=300, dataset=dataset, DP=False,
                                   saving_path="{}/fair_adj".format(model_type))
        else:
            tmp_list = run_GCN(gender, ft, adj, labels, epochs=300, dataset=dataset, DP=False,
                               saving_path="{}/fair_adj".format(model_type))
        print("Target Accuracy:", tmp_list)
        df_tar_acc = pd.DataFrame(tmp_list)
        df_tar_acc.to_csv("{}/fair_adj/MIA_res/{}_attack{}_acc_agg.csv".format(model_type, dataset,
                                                                      "_fair" if fair_sample else ""), index=False)
    df_acc = []
    agg_all = []
    partial_done = False
    for at in [3, 6]:
        a_all, p_all, r_all, roc_all = 0, 0, 0, 0
        for t in range(3):
            if run_partial and not partial_done:
                get_partial(model_type=model_type.lower(), datapath="{}/fair_adj/".format(model_type),
                            dataset=dataset, fair_sample=fair_sample, t=t)

            print("Start Attack {} with {} balanced sample for {} time.".format(at, fair_sample, t))
            a, p, r, roc, acc_list = attack_main(datapath=model_type + '/fair_adj/',
                                                 dataset=dataset,
                                                 shadow_data='cora',
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
    df_agg_performance = pd.DataFrame(agg_all, columns=["Attack Type", "Accuracy", "Precision", "Recall", "AUC"])

    if run_attack:
        df_acc.to_csv("{}/fair_adj/MIA_res/{}_attack{}acc.csv".format(model_type, dataset,
                                                                      "_fair_" if fair_sample else ""), index=False)
        df_agg_performance.to_csv(
            "{}/fair_adj/MIA_res/{}_attack{}performance.csv".format(model_type, dataset,
                                                                    "_fair_" if fair_sample else ""),
            index=False)

        df_acc_agg = df_acc.groupby("Attack").mean()
        df_acc_agg.to_csv("{}/fair_adj/MIA_res/{}_attack{}_acc_agg.csv".format(model_type, dataset,
                                                                      "_fair" if fair_sample else ""), index=False)