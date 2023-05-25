import pandas as pd
import pickle as pkl
from utils import load_data


def accuracy(pred, labels, inds=[]):
    if len(pred.shape) > 1:
        pred_res = pred.argmax(dim=1)
    else:
        pred_res = pred

    if len(inds) == 0:
        inds = range(len(pred))

    acc = (pred_res[inds] == labels[inds]).numpy().mean()
    return acc


def overfitting(acc_list):
    # [acc_train, acc_train1, acc_train2, acc_test, acc_test1, acc_test2]
    oft_all = acc_list[0] - acc_list[3]
    oft_female = acc_list[1] - acc_list[4]
    oft_male = acc_list[2] - acc_list[5]

    oft_list = [oft_all, oft_female, oft_male]

    return oft_list



if __name__ == "__main__":
    dataset = "facebook"
    model = "gcn"
    datapath = "dataset/"
    ego_user = "107"

    pred_file = "{}/{}_{}_pred.pkl".format(model, dataset, model.lower())
    acc_file = "{}/{}_{}_acc.pkl".format(model, dataset, model.lower())

    pred = pkl.load(open(pred_file, 'rb'))
    adj, ft, gender, labels = load_data(datapath, dataset, ego_user, dropout=0)
    acc_list = pkl.load(open(acc_file, 'rb'))

    ofts = overfitting(acc_list)

    acc = accuracy(pred, labels)
    male_inds = gender == 2
    female_inds = gender == 1
    acc_male = accuracy(pred, labels, male_inds)
    acc_female = accuracy(pred, labels, female_inds)


    pass