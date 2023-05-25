import numpy as np
from pyGAT.models import GAT
import copy
import torch
from sklearn.preprocessing import StandardScaler
import time


def run_GAT(gender, ft_ori, adj, labels, epochs, ego='', dataset="facebook", saving_path="GAT",
            DP=False, epsilon=1.0):
    ft = copy.deepcopy(ft_ori)
    if len(labels.shape) > 1:
        if torch.is_tensor(labels):
            labels = labels.argmax(dim=1)
        else:
            labels = torch.Tensor(labels.argmax(axis=1))
    if "facebook" in dataset:
        dataset = "facebook"
        nhid = 5
        dropout = 0.8
        nheads = 9
        lr = 0.004
        train, val = 0.4, 0.3
        patience = 20
    elif "tagged" in dataset:
        nhid = 8
        dropout = 0.6
        nheads = 8
        lr = 0.005
        train, val = 0.6, 0.2
        loss_weight = torch.Tensor([1, 1])
        patience = 20
        batch_size = 50 # this parameter is not used now
    elif "pokec" in dataset:
        nhid = 6
        dropout = 0.65
        nheads = 10
        lr = 0.004
        train, val = 0.5, 0.2
        patience = 10
    else:
        nhid = 6
        dropout = 0.6
        nheads = 6
        lr = 0.05
        train, val = 0.4, 0.3
        patience = 20


    idx_random = np.arange(len(labels))
    np.random.shuffle(idx_random)
    idx_train = torch.LongTensor(idx_random[:int(len(labels) * train)])
    idx_val = torch.LongTensor(idx_random[int(len(labels) * train):int(len(labels) * (train + val))])
    idx_test = torch.LongTensor(idx_random[int(len(labels) * (train + val)):])
    ss = StandardScaler()
    ft = torch.FloatTensor(ss.fit_transform(ft))

    loss_values = []
    bad_counter = 0
    best = epochs + 1
    best_epoch = 0



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

    optimizer = torch.optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=0.1)

    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(ft, adj)
        if "tagged" in dataset:
            loss_train = F.nll_loss(output[idx_train], labels[idx_train], weight=loss_weight)
        else:
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        output = model(ft, adj)
        if "tagged" in dataset:
            loss_train = F.nll_loss(output[idx_train], labels[idx_train], weight=loss_weight)
        else:
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        if "tagged" in dataset:
            loss_val = F.nll_loss(output[idx_val], labels[idx_val], weight=loss_weight)
        else:
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.data.item()),
              'acc_train: {:.4f}'.format(acc_train.data.item()),
              'loss_val: {:.4f}'.format(loss_val.data.item()),
              'acc_val: {:.4f}'.format(acc_val.data.item()),
              'time: {:.4f}s'.format(time.time() - t))

        return loss_val.data.item()

    def train_batch(epoch):
        t = time.time()
        model.train()
        epoch_ids = torch.randperm(idx_train.nelement())
        number_of_batch = math.floor(len(idx_train) / batch_size)
        optimizer.zero_grad()

        for batch in range(number_of_batch):
            idx_batch = idx_train[epoch_ids[batch * batch_size: (batch + 1) * batch_size]]
            output = model(ft, adj)
            if "tagged" in dataset:
                loss_train = F.nll_loss(output[idx_batch], labels[idx_batch], weight=loss_weight)
            else:
                loss_train = F.nll_loss(output[idx_batch], labels[idx_batch])
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        model.eval()
        output = model(ft, adj)
        if "tagged" in dataset:
            loss_train = F.nll_loss(output[idx_train], labels[idx_train], weight=loss_weight)
        else:
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        if "tagged" in dataset:
            loss_val = F.nll_loss(output[idx_val], labels[idx_val], weight=loss_weight)
        else:
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
        if "tagged" in dataset:
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

        if bad_counter == patience and not DP:
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
            acc_train1, acc_train2 = accuracy(output[idx_train_1], labels[idx_train_1]), accuracy(output[idx_train_2],
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

    with open('{}/{}_gat_target.pkl'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(all_results, f)

    with open('{}/{}_gat_acc.pkl'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(acc_list, f)

    with open('{}/ind.{}.att'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(att, f)
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
    if not DP:
        return [-1] + acc_list
    else:
        return [epsilon] + acc_list


if __name__ == "__main__":
