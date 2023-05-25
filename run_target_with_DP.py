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
from find_sigma import bs_sigma, search_multi
from tqdm import tqdm
import argparse
from pyvacy_master.pyvacy import optim as dp_optim, analysis, sampling



def run_target(model_type, config, gender, ft, adj, labels,
               epochs, dataset="facebook", saving_path="GAT",
               ep=100, bs=20):

    if len(labels.shape) > 1:
        if torch.is_tensor(labels):
            labels = labels.argmax(dim=1)
        else:
            labels = torch.Tensor(labels.argmax(axis=1))

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
        nheads = 0  # not used in this case
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

    if model_type == "GAT":
        model = GAT(nfeat=ft.shape[1],
                    nhid=nhid,
                    nclass=labels.max().item() + 1,
                    dropout=dropout,
                    nhead=nheads)
    elif model_type == "GCN":
        model = GCN(nfeat=ft.shape[1],
                    nhid=nhid,
                    nclass=labels.max().item() + 1,
                    dropout=dropout,
                    gpu=False)
    else:
        model, enc1, enc2 = init_GraphSAGE(ft, adj, labels.max().item() + 1)
    #model.cuda()

    #sigma = bs_sigma(len(idx_train), bs, 100, ep) v1 related
    mu_ep = search_multi(len(idx_train), bs, epochs, 1e-6, ep) # v2 related
    print("multiplier = {} for epsilon={}".format(mu_ep, ep))
    mbs = bs

    def train_with_batch(epoch, bs=1, mbs=1):
        #optimizer = optim.Adam(model.parameters(),
#                                   lr=lr,
#                                   weight_decay=0.1)
        training_parameters = {
            #'N': len(idx_train),
            'l2_norm_clip': 1.0,
            # A coefficient used to scale the standard deviation of the noise applied to gradients.
            'noise_multiplier': mu_ep,
            # Each example is given probability of being selected with minibatch_size / N.
            # Hence this value is only the expected size of each minibatch, not the actual.
            'minibatch_size': bs,
            'microbatch_size': mbs,
            'lr':lr
        }
        optimizer = dp_optim.DPSGD(params=model.parameters(), **training_parameters) # v2 related
        t = time.time()
        model.train()
        for i in tqdm(range(0, len(idx_train), bs)):
            optimizer.zero_grad()
            for j in range(0, bs, mbs):
                ids = idx_train[i:i + bs][j:j+mbs]
                optimizer.zero_microbatch_grad()
                output = model(ft, adj)
                loss_train = F.nll_loss(output[ids], labels[ids])
                loss_train.backward()
                optimizer.microbatch_step()
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


    loss_values = []
    bad_counter = 0
    best = epochs + 1
    best_epoch = 0
    for epoch in range(epochs):
        #loss_values.append(train(epoch))
        loss_values.append(train_with_batch(epoch, bs, mbs))

        torch.save(model.state_dict(), '{}/{}.pkl'.format(saving_path, epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == patience:
            break

        files = glob.glob(saving_path + '/[0-9]*.pkl')
        for file in files:
            file = file.replace('\\', '/')
            epoch_nb = int(file.split('.')[0].split('/')[-1])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob(saving_path + '/[0-9]*.pkl')
    for file in files:
        file = file.replace('\\', '/')
        epoch_nb = int(file.split('.')[0].split('/')[-1])
        if epoch_nb > best_epoch:
            os.remove(file)

    def compute_test():
        model.eval()
        if model_type == "GraphSAGE":
            output = model.forward(np.arange(len(labels)))
        else:
            output = model(ft, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.data.item()),
              "accuracy= {:.4f}".format(acc_test.data.item()))

    def compute_acc_group():
        model.eval()
        if model_type == "GraphSAGE":
            output = model.forward(np.arange(len(labels)))
        else:
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
    acc_list = compute_acc_group()

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

    acc_3groups = compute_acc_3group(gender)
    print("Target performance of 3 subroups are:", acc_3groups)

    oh_label = one_hot_trans(labels)

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
    '''all_results = [ft, labels, outputs, adj, gender]

    with open('{}/{}_gat_target.pkl'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(all_results, f)

    with open('{}/{}_gat_acc.pkl'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(acc_list, f)

    with open('{}/ind.{}.allx'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(ft, f)

    with open('{}/ind.{}.ally'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(oh_label, f)
    '''

    with open('{}/{}_{}_pred_ep={}_bs={}.pkl'.format(saving_path, dataset, model_type, ep, bs), 'wb') as f:
        pkl.dump(outputs, f)

    '''
    with open('{}/ind.{}.adj'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(1 * (adj > 0), f)

    with open('{}/ind.{}.gender'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(gender, f)
    if not DP:
        return np.array([0] + acc_list)
    else:
        return np.array([lbd] + acc_list)'''


if __name__ == "__main__":

    import json
    from utils import load_data
    # input
    parser = argparse.ArgumentParser()
    parser.add_argument('--ep', type=float, default=100, help='epsilon for baseline')
    parser.add_argument('--bs', type=int, default=20, help='epsilon for baseline')
    parser.add_argument('--model_type', type=str, default="GAT", help='epsilon for baseline')
    args = parser.parse_args()

    ep = args.ep
    model_type = args.model_type
    bs = args.bs

    dataset = "tagged_40"
    datapath = "dataset/"
    ego_user = 107
    epoch = 60

    with open('model_config.json', 'r') as f:
        config = json.load(f)[dataset][model_type]
    delta = config["delta"]
    adj, ft, gender, labels = load_data(datapath, dataset, ego_user, dropout=0)
    ss = StandardScaler()
    ft = torch.FloatTensor(ss.fit_transform(ft))

    MIA_res_addon = ""
    if delta > 0:
        adj = pkl.load(open(config["adj_location"], "rb"))
        MIA_res_addon = "CNR/Group/Reduce/Delta={}/".format(delta)
    run_target(model_type, config, gender, ft, adj, labels, epochs=epoch, dataset=dataset, saving_path=model_type + "/DP_con", ep=ep, bs=bs)