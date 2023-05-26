import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from pyGAT.models import GAT
from pygcn.models import GCN
from graphsage.model import init_GraphSAGE
from torch import optim
import time
import pickle as pkl
import torch.nn.functional as F
from utils import accuracy, one_hot_trans
import glob
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.autograd import Variable


def ARR_matrix(adj, ep):
    print("preprocess matrix with ARR! epsilon={}".format(ep))
    N1 = adj.sum()
    N2 = adj.shape[0]**2 - N1
    #mu = torch.exp(torch.Tensor([ep]))/(torch.exp(torch.Tensor([ep]))+1)
    #rho = torch.exp(torch.Tensor([-ep]))
    #mu = N1 / (N1 + N2 * rho)
    p1 = 1 / (1+torch.exp(torch.tensor([ep])))
    p2 = p1
    prob = adj * (1-p1) + (1 - adj) * p2
    adj_ptb = torch.bernoulli(prob)

    #clip with d max
    dmax=N1 // adj.shape[0]
    cliped_pos = torch.multinomial(adj_ptb, int(dmax))
    adj_final = torch.zeros_like(adj_ptb)
    for i in range(adj_ptb.shape[0]):
        adj_final[i, cliped_pos[i]] = 1
    adj_utri = torch.triu(adj_final)
    adj_ltri = torch.flip(torch.triu(adj_final, diagonal=1), [0,1])
    adj_final = adj_utri + adj_ltri
    return adj_final



def run_target(model_type, config, gender, ft, adj, labels,
               epochs, dataset="facebook", saving_path="GAT",
               Min=False, gamma=torch.inf):

    if len(labels.shape) > 1:
        if torch.is_tensor(labels):
            labels = labels.argmax(dim=1)
        else:
            labels = torch.LongTensor(labels.argmax(axis=1))

    if model_type == "GAT":
        nhid = config["nhid"]
        dropout = config["dropout"]
        nheads = config["nheads"]
        lr = config["lr"]
        train, val = config["train"], config["val"]
        patience = config["patience"]
    else:
        nhid = config["nhid"]
        dropout = config["dropout"]
        nheads = 0 # not used in this case
        lr = config["lr"]
        train, val = config["train"], config["val"]
        patience = config["patience"]

    idx_random = np.arange(len(labels))
    np.random.shuffle(idx_random)
    idx_train = torch.LongTensor(idx_random[:int(len(labels) * train)])
    idx_val = torch.LongTensor(idx_random[int(len(labels) * train):int(len(labels) * (train + val))])
    idx_test = torch.LongTensor(idx_random[int(len(labels) * (train + val)):])
    ss = StandardScaler()
    ft = torch.FloatTensor(ss.fit_transform(ft))
    if isinstance(labels, np.ndarray):
        labels = torch.LongTensor(labels)
    if isinstance(adj, np.ndarray):
        adj = torch.FloatTensor(adj)
    adj = adj.float()

    if model_type == "GAT":
        model = GAT(nfeat=ft.shape[1],
                    nhid=nhid,
                    nclass=labels.max().item() + 1,
                    dropout=dropout,
                    nhead=nheads,
                    Min=Ptb,
                    gamma=gamma)
    elif model_type == "GCN":
        model = GCN(nfeat=ft.shape[1],
                    nhid=nhid,
                    nclass=int(labels.max().item() + 1),
                    dropout=dropout,
                    Min=Ptb,
                    gamma=gamma)
    else:
        model, enc1, enc2 = init_GraphSAGE(ft, adj, labels.max().item() + 1)

    def train(epoch):
        adj_copy = adj
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.7)
        model.train()
        t = time.time()
        optimizer.zero_grad()
        loss = model.loss(idx_train, Variable(torch.LongTensor(labels[np.array(idx_train)])))
        loss.backward()
        optimizer.step()
        model.eval()

        output = model.forward(np.arange(len(labels)))
        loss_train = model.loss(idx_train, Variable(torch.LongTensor(labels[np.array(idx_train)])))
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_val = model.loss(idx_val, Variable(torch.LongTensor(labels[np.array(idx_val)])))
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.data.item()),
                  'acc_train: {:.4f}'.format(acc_train.data.item()),
                  'loss_val: {:.4f}'.format(loss_val.data.item()),
                  'acc_val: {:.4f}'.format(acc_val.data.item()),
                  'time: {:.4f}s'.format(time.time() - t))
        return loss_val.data.item()

    loss_values = []
    bad_counter = 0
    best = epochs + 1
    best_epoch = 0
    if epochs == 0:
        for t in range(10):
            for p in model.parameters():
                p = torch.randn_like(p)

    for epoch in range(epochs):
        loss_values.append(train(epoch))

        torch.save(model.state_dict(), '{}/{}.pkl'.format(model_type, epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == patience:
            break

        files = glob.glob(model_type + '/[0-9]*.pkl')
        for file in files:
            file = file.replace('\\', '/')
            epoch_nb = int(file.split('.')[0].split('/')[1])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob(model_type + '/[0-9]*.pkl')
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

        return [acc_train.detach().item(), acc_train1.detach().item(), acc_train2.detach().item(),
                acc_test.detach().item(), acc_test1.detach().item(), acc_test2.detach().item()]

    compute_test()
    def compute_acc_3group(gender):
        model.eval()
        if model_type == "GraphSAGE":
            output = model.forward(np.arange(len(labels)))
        else:
            output = model(ft, adj)
        indexs = [[], [], []]
        if torch.is_tensor(gender):
            gender = gender.detach().numpy()
        for i in range(len(adj)):
            if adj.is_sparse:
                neighbors = gender[adj._indices()[1][adj._indices()[0] == i]]
            else:
                neighbors = adj[i] * gender
            self_gender = int(gender[i])
            if self_gender in np.unique(neighbors):
                indexs[self_gender].append(i)
            if 2-self_gender in np.unique(neighbors):
                indexs[0].append(i)

        acc_lists = []
        for g in range(3):
            gi_train_idx = list(set(indexs[g]) & (set(idx_train.detach().numpy())).union(set(idx_val.detach().numpy())))
            gi_test_idx = list(set(indexs[g]) & (set(idx_test.detach().numpy())))

            acc_gi_train = accuracy(output[gi_train_idx], labels[gi_train_idx]).item()
            acc_gi_test = accuracy(output[gi_test_idx], labels[gi_test_idx]).item()
            acc_lists.append([acc_gi_train, acc_gi_test])

        return acc_lists

    if len(gender) > 0:
        acc_3groups = compute_acc_3group(gender)
        print("Target performance of 3 subroups are:", acc_3groups)

    if model_type == "GraphSAGE":
        outputs = model.forward(np.arange(len(labels)))
    else:
        outputs= model(ft, adj)
    print(np.unique(outputs.argmax(axis=1), return_counts=True))
    if outputs.shape[1] == 2:
        preds = outputs.argmax(axis=1)
        acc = accuracy_score(preds, labels)
        prec = precision_score(labels, preds)
        recall = recall_score(labels, preds)

        print("Accuracy = {:.2%}, Precision={:.2%}, Recall = {:.2%}".format(acc, prec, recall))

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    with open('{}/{}_{}_pred.pkl'.format(saving_path, dataset, model_type), 'wb') as f:
        pkl.dump(outputs, f)

if __name__ == "__main__":

    import json
    from utils import load_data
    # input

    dataset = "facebook"
    model_type = "GCN"
    datapath = "dataset/"
    epoch = 0

    with open('model_config.json', 'r') as f:
        config = json.load(f)[dataset][model_type]
    adj, ft, gender, labels = load_data(datapath, dataset, dropout=0)
    ss = StandardScaler()
    ft = torch.FloatTensor(ss.fit_transform(ft))

    MIA_res_addon = ""
    run_target(model_type, config, gender, ft, adj, labels, epochs=epoch, dataset=dataset, saving_path=model_type,
               ARR=True, epsilon=1.0)