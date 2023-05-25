from utils import *
from sklearn.preprocessing import StandardScaler
from stealing_link.partial_graph_generation import get_partial
from attack import attack_main
from pyGAT.models import GAT


if __name__=="__main__":
    nhid = 32
    dropout = 0.5
    lr = 0.004
    train, val = 0.4, 0.3
    patience = 20
    loss_weight = torch.Tensor([1, 1])
    datapath = "dataset/"
    dataset = "facebook"
    ego_user = "107"
    _, ft, gender, labels = load_data(datapath, dataset, ego_user, dropout=0)
    delta = 0.1
    adj = pkl.load(open("{}/CNR/Group/Reduce/Delta={}/ind.{}.adj".format("gcn", delta, dataset), "rb"))
    acc_lists = []

    if adj.is_sparse:
        adj = adj.to_dense()
    if adj.dtype == torch.int64:
        adj = adj.float()

    ss = StandardScaler()
    ft = torch.FloatTensor(ss.fit_transform(ft))

    model = GAT(nfeat=ft.shape[1],
                nhid=5,
                nhead=9,
                nclass=labels.max().item() + 1,
                dropout=dropout)

    rand_list = []
    for t in range(10):
        for p in model.parameters():
            rand_list.append(torch.randn_like(p))
            p = torch.randn_like(p)

    saving_path = "Null_model/"
    oh_label = one_hot_trans(labels)

    outputs = model(ft, adj)

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    all_results = [ft, labels, outputs, adj, gender]

    with open('{}/{}_gat_target.pkl'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(all_results, f)

    with open('{}/ind.{}.allx'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(ft, f)

    with open('{}/ind.{}.ally'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(oh_label, f)

    with open('{}/{}_gat_pred.pkl'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(outputs, f)

    with open('{}/ind.{}.adj'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(1 * (adj > 0), f)

    with open('{}/ind.{}.gender'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(gender, f)

    datapath = "Null_model/"
    partial_done = False
    att_res = []
    att_acc_res = []
    for t in range(10):
        for at in [3, 6]:

            if not partial_done:
                get_partial(model_type="GAT".lower(),
                            datapath=datapath, dataset=dataset, fair_sample=True, t=0)
                partial_done = True
            print("Start Attack {} with {} balanced sample ".format(at, True))
            a, p, r, roc, acc_list = attack_main(datapath=datapath,
                                             dataset=dataset,
                                             shadow_data="cora",
                                             ratio=0.2,
                                             attack_type=at,
                                             fair_sample=True,
                                             t=0,
                                             prepare_new=True)
            print("Average Performance of attack{} on Null_model:\n"
              " Accuracy = {},\n"
              " Precision={},\n"
              " Recall = {},\n"
              " ROC={}".format(at, a, p, r, roc))
            att_res.append([at, a, p, r, roc])
            att_acc_res.append(acc_list)
    df_acc = pd.DataFrame(att_acc_res, columns=["Attack", "Acc_train", "Acc_train1", "Acc_train2", "Acc_train0",
                                            "Acc_test", "Acc_test1", "Acc_test2", "Acc_test0"])
    df_acc_agg = df_acc.groupby(["Attack"]).mean()

    df_performance = pd.DataFrame(att_res, columns=["Attack", "Accuracy", "Precsion", "Recall", "ROC"])
    df_agg_performance = df_performance.groupby(["Attack"]).mean()
    df_acc_agg.to_csv("Null_model/{}_attack_acc_gat.csv".format(dataset))
    df_agg_performance.to_csv("Null_model/{}_attack_performance_gat.csv".format(dataset))

    pass
