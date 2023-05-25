from utils import *
import numpy as np
import glob
from igraph import *
from pyGAT.models import GAT
from torch import optim
import torch.nn.functional as F
import time
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
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


def run_GAT(gender, ft, adj, labels, epochs, ego='', dataset="facebook", saving_path="GAT",
            DP=False, epsilon=1.0, x_steps=0, y_steps=0):
    if len(labels.shape) > 1:
        if torch.is_tensor(labels):
            labels = labels.argmax(dim=1)
        else:
            labels = torch.Tensor(labels.argmax(axis=1))
    if "facebook" in dataset:
        dataset = "facebook"
        nhid = 5
        dropout = 0.6
        nheads = 9
        lr = 0.003
        train, val = 0.6, 0.2
        patience = 20
        batch_size = int(len(labels) * train * 1.0)
    elif "tagged" in dataset:
        nhid = 8
        dropout = 0.6
        nheads = 8
        lr = 0.005
        train, val = 0.2, 0.15
        patience = 20
        loss_weight = torch.Tensor([1, 1])
        batch_size = int(len(labels) * train)
    elif "pokec" in dataset:
        nhid = 8
        dropout = 0.65
        nheads = 8
        lr = 0.004
        train, val = 0.5, 0.2
        patience = 20
    else:
        nhid = 5
        dropout = 0.6
        nheads = 7
        lr = 0.05
        train, val = 0.6, 0.3
        patience = 20


    if "tagged" in dataset:
        idx_random = np.arange(len(labels))
        np.random.shuffle(idx_random)
        idx_train = torch.LongTensor(idx_random[:int(len(labels) * train)])
        idx_val = torch.LongTensor(idx_random[int(len(labels) * train):int(len(labels) * (train + val))])
        idx_test = torch.LongTensor(idx_random[int(len(labels) * (train + val)):])
    else:
        idx_random = np.arange(len(labels))
        np.random.shuffle(idx_random)
        idx_train = torch.LongTensor(idx_random[:int(len(labels) * train)])
        idx_val = torch.LongTensor(idx_random[int(len(labels) * train):int(len(labels) * (train + val))])
        idx_test = torch.LongTensor(idx_random[int(len(labels) * (train + val)):])
    ss = StandardScaler()
    ft = torch.FloatTensor(ss.fit_transform(ft))

    number_of_class = len(np.unique(labels))

    loss_values = []
    bad_counter = 0
    best = 1e10
    best_epoch = 0
    if DP:
        xep = yep = 0.5 * epsilon
        # prepare data
        data = dataset_class(ft, labels, idx_train, idx_val, idx_test, adj, gender)
        # perturb feature
        ft_pert = FeaturePerturbation(x_eps=xep)
        data = ft_pert(data)


        # perturb label
        y_pert = LabelPerturbation(y_eps=yep)
        data = y_pert(data)
        ft = data.x
        labels = data.y.argmax(dim=1)

        clf = NodeClassifier(input_dim=ft.shape[1],
                             num_classes=number_of_class,
                             model='gat',
                             hidden_dim=nhid,
                             dropout=dropout,
                             x_steps=x_steps,
                             y_steps=y_steps,
                             nhead=nheads)
        optimizer = optim.Adam(clf.gnn.parameters(),
                               lr=lr,
                               weight_decay=0.1)
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

            if val_metrics['val/loss'] <= best:
                best = val_metrics['val/loss']
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter >= patience:
                break

        acc_list = clf.get_target_res(data, saving_path=saving_path, dataset=dataset)
        return [epsilon] + acc_list
    else:
        model = GAT(nfeat=ft.shape[1],
                    nhid=nhid,
                    nclass=int(labels.max()) + 1,
                    dropout=dropout,
                    nhead=nheads,
                    alpha=0.1,
                    DP=DP
                    )
        if adj.is_sparse:
            adj = adj.to_dense()

        optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=0.1)



        def train(epoch):
            t = time.time()
            model.train()
            optimizer.zero_grad()

            output = model(ft, adj)

            loss_train = F.nll_loss(output[idx_train], labels[idx_train])

            loss_train.backward()
            optimizer.step()

            model.eval()
            output = model(ft, adj)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.data.item()),
                  'acc_train: {:.4f}'.format(acc_train.data.item()),
                  'loss_val: {:.4f}'.format(loss_val.data.item()),
                  'acc_val: {:.4f}'.format(acc_val.data.item()),
                  'time: {:.4f}s'.format(time.time() - t))

            return loss_val.data.item()

        for epoch in range(epochs):
            if DP:
                loss_values.append(train(epoch))
            else:
                loss_values.append(train(epoch))

            torch.save(model.state_dict(), 'GAT/{}.pkl'.format(epoch))
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == patience :
                break

        files = glob.glob('GAT/[0-9]*.pkl')
        for file in files:
            file = file.replace('\\', '/')
            epoch_nb = int(file.split('.')[0].split('/')[1])
            if epoch_nb < best_epoch:
                os.remove(file)

        files = glob.glob('GAT/[0-9]*.pkl')
        for file in files:
            file = file.replace('\\', '/')
            epoch_nb = int(file.split('.')[0].split('/')[1])
            if epoch_nb > best_epoch:
                os.remove(file)

        def compute_test():
            model.eval()
            output = model(ft, adj)
            loss_test = F.nll_loss(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
            print("Test set results:",
                  "loss= {:.4f}".format(loss_test.data.item()),
                  "accuracy= {:.4f}".format(acc_test.data.item()))

        def compute_acc_group():
            model.eval()
            output = model(ft, adj)
            acc_train = accuracy(output[idx_train], labels[idx_train])
            acc_test = accuracy(output[idx_test], labels[idx_test])
            if len(gender) == 0:
                acc_train1 = acc_train2 = acc_train
                acc_test1 = acc_test2 = acc_test
            else:
                if torch.is_tensor(gender):
                    gender_train = gender[idx_train]
                    gender_test = gender[idx_test]
                else:
                    gender_train = torch.Tensor(gender)[idx_train]
                    gender_test = torch.Tensor(gender)[idx_test]

                idx_train_1 = idx_train[gender_train == 1]
                idx_train_2 = idx_train[gender_train == 2]
                acc_train1, acc_train2 = accuracy(output[idx_train_1], labels[idx_train_1]), accuracy(
                    output[idx_train_2],
                    labels[idx_train_2])

                idx_test_1 = idx_test[gender_test == 1]
                idx_test_2 = idx_test[gender_test == 2]
                acc_test1, acc_test2 = accuracy(output[idx_test_1], labels[idx_test_1]), accuracy(output[idx_test_2],
                                                                                                  labels[idx_test_2])

            return [acc_train, acc_train1, acc_train2, acc_test, acc_test1, acc_test2]

        compute_test()
        acc_list = compute_acc_group()

        oh_label = one_hot_trans(labels)

        outputs = model(ft, adj)
        if outputs.shape[1] == 2:
            preds = outputs.argmax(axis=1)
            acc = accuracy_score(preds, labels)
            prec = precision_score(labels, preds)
            recall = recall_score(labels, preds)

            print("Accuracy = {:.2%}, Precision={:.2%}, Recall = {:.2%}".format(acc, prec, recall))


        print(np.unique(outputs.argmax(axis=1), return_counts=True))
        att = model.get_attentions(ft, adj)

        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        all_results = [ft, labels, outputs, adj, gender]

        save_target_results(saving_path, dataset, all_results, acc_list, att, ft, oh_label, outputs, adj, gender)

        return [10] + acc_list



if __name__ == "__main__":
    datapath = "dataset/"
    dataset = "facebook"
    ego_user = "107"
    delta = 0.1
    adj, ft, gender, labels = load_data(datapath, dataset, ego_user, dropout=0)
    adj = pkl.load(open("{}/CNR/Group/Reduce/Delta={}/ind.{}.adj".format("GAT", delta, dataset), "rb"))
    acc_lists = []
    tmp_list = [0, 0]
    '''for t in range(5):
        count = 0
        while tmp_list[1] < 0.3 and count < 5:
            count += 1
            print("Try {} round - {} time for No DP".format(count, t))
            tmp_list = run_GAT(gender, ft, adj, labels, epochs=200, dataset=dataset, DP=False,
                               saving_path="GAT/DP/nodp/t={}".format(t))

            if tmp_list[1] > 0.3:
                acc_lists.append(tmp_list)
            else:
                print("Target model with DP failed for no DP and t={} for {} times".format(t, count))
    '''
    for epsilon in [np.inf,]:
        if epsilon is np.inf:
            kx, ky = 0, 0
        else:
            kx, ky = 3, 3
        for t in range(2):
            tmp_list = [0, 0]
            count = 0
            while tmp_list[1] < 0.1 and count < 5:
                count += 1
                print("Try {} round - {} time for epsilon={}".format(count, t, epsilon))
                tmp_list = run_GAT(gender, ft, adj, labels, epochs=300, dataset=dataset, DP=True, epsilon=epsilon,
                                   saving_path="GAT/DP/ep={}/t={}".format(epsilon, t), x_steps=kx, y_steps=ky)
            if tmp_list[1] > 0.1:
                acc_lists.append(tmp_list)
            else:
                print("Target model with DP failed for epsilon={} and t={} for {} times".format(epsilon, t, count))

    acc_lists = np.array(acc_lists)
    df_acc = pd.DataFrame(acc_lists, columns=["epsilon", "train acc", "train acc 1", "train acc 2",
                                              "test acc", "test acc 1", "test acc 2"])
    df_acc_avg = df_acc.groupby(["epsilon"]).mean()
    df_acc_avg.to_csv("GAT/DP/target_acc_agg.csv")
    pass





