from utils import *
import numpy as np
import glob
from torch import optim
import time
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from LPGNN.transforms import FeaturePerturbation, LabelPerturbation, dataset_class
from LPGNN.models import NodeClassifier


def get_all_ids(database):
    files = glob.glob("dataset/" + database + '/*.pkl')
    ids = []
    for file in files:
        id = int(file.split('-')[0].split('\\')[1])
        ids.append(id)
    return ids


def map_labels(labels, c_list):
    new_label = []
    for i in range(len(labels)):
        current_id = labels[i]
        new_id = int(np.where(np.unique(c_list) == current_id)[0])
        new_label.append(new_id)
    return np.array(new_label)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def run_GCN(gender, ft, adj, labels, epochs, ego='', dataset="facebook", saving_path="gcn",
            DP=False, epsilon=0.1, x_steps=0, y_steps=0):
    if len(labels.shape) > 1:
        if torch.is_tensor(labels):
            labels = labels.argmax(dim=1)
        else:
            labels = torch.Tensor(labels.argmax(axis=1))

    if "tagged" in dataset:
        nhid = 20
        dropout = 0.5
        lr = 0.005
        train, val = 0.6, 0.2
        patience = 20
    else:
        nhid = 32
        dropout = 0.5
        lr = 0.01
        train, val = 0.6, 0.2
        patience = 20

    if adj.is_sparse:
        adj = adj.to_dense()
    if adj.dtype == torch.int64:
        adj = adj.float()

    idx_random = np.arange(len(labels))
    np.random.shuffle(idx_random)
    idx_train = torch.LongTensor(idx_random[:int(len(labels) * train)])
    idx_val = torch.LongTensor(idx_random[int(len(labels) * train):int(len(labels) * (train + val))])
    idx_test = torch.LongTensor(idx_random[int(len(labels) * (train + val)):])
    ss = StandardScaler()
    ft = torch.FloatTensor(ss.fit_transform(ft))


    bad_counter = 0
    best = 1e10

    xep = yep = 0.5 * epsilon
    # prepare data
    data = dataset_class(ft, labels, idx_train, idx_val, idx_test, adj, gender)
    # perturb feature
    ft_pert = FeaturePerturbation(x_eps=xep)
    data = ft_pert(data)

    # perturb label
    number_of_class = len(np.unique(labels))
    y_pert = LabelPerturbation(y_eps=yep)
    data = y_pert(data)
    ft = data.x
    labels = data.y.argmax(dim=1)

    clf = NodeClassifier(input_dim=ft.shape[1],
                         num_classes=number_of_class,
                         model='gcn',
                         hidden_dim=nhid,
                         dropout=dropout,
                         x_steps=x_steps,
                         y_steps=y_steps,
                         nhead=0)
    optimizer = optim.Adam(clf.gnn.parameters(),
                           lr=lr,
                           weight_decay=5e-4)
    for epoch in range(epochs):
        t = time.time()
        optimizer.zero_grad()
        train_loss, train_metrics = clf.training_step(data)
        train_loss.backward()
        optimizer.step()
        val_metrics = clf.validation_step(data)
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(train_metrics['train/loss']),
              'acc_train: {:.4f}%'.format(train_metrics['train/acc']),
              'loss_val: {:.4f}'.format(val_metrics['val/loss']),
              'acc_val: {:.4f}%'.format(val_metrics['val/acc']),
              'time: {:.4f}s'.format(time.time() - t))

        if val_metrics['val/loss'] < best:
            best = val_metrics['val/loss']
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter >= patience:
            break

    acc_list = clf.get_target_res(data, saving_path=saving_path, dataset=dataset)
    return [epsilon] + acc_list


if __name__ == "__main__":
    datapath = "dataset/"
    dataset = "pokec"
    ego_user = "107"
    adj, ft, gender, labels = load_data(datapath, dataset, ego_user, dropout=0)
    delta = 0.1
    adj = pkl.load(open("{}/CNR/Group/Reduce/Delta={}/ind.{}.adj".format("gcn", delta, dataset), "rb"))
    acc_lists = []
    KxKy_dict = {np.inf: [0, 0],
                 5.0: [8, 4],
                 3.0: [8, 4],
                 1.0: [8, 4],
                 0.5: [10, 4],
                 0.1: [12, 4]}
    for epsilon in [np.inf, 5.0, 3.0, 1.0, 0.5, 0.1]:
        if epsilon is np.inf:
            Kx, Ky = 0, 0
        else:
            Kx, Ky = 4, 4
        for t in range(5):
            tmp_list = [0, 0]
            count = 0
            while tmp_list[1] < 0.2 and count < 5:
                count += 1
                print("Try {} time for epsilon={}".format(count, epsilon))
                tmp_list = run_GCN(gender, ft, adj, labels, epochs=300, dataset=dataset,
                                   DP=True, epsilon=epsilon, saving_path="gcn/DP/ep={}/t={}".format(epsilon, t),
                                   x_steps=Kx, y_steps=Ky)
            if tmp_list[1] > 0.2:
                acc_lists.append(tmp_list)
            else:
                print("Target model with DP failed for epsilon={} and t={} for {} times".format(epsilon, t, count))


    acc_lists = np.array(acc_lists)
    df_acc = pd.DataFrame(acc_lists, columns=["epsilon", "train acc", "train acc 1", "train acc 2",
                                              "test acc", "test acc 1", "test acc 2"])
    df_acc_avg = df_acc.groupby(["epsilon"]).mean()
    df_acc_avg.to_csv("gcn/DP/target_acc_agg.csv")
    pass