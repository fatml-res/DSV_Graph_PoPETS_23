import pickle as pkl

import numpy as np
import setuptools.wheel
from igraph import *
import glob


def load_featname(file):
    '''
    :param file: featname file
    :return: list of featname
    '''
    f = open(file)
    tmp_feats = []
    for line in f:
        line = line.strip().split(' ')
        feats = ' '.join(line[1:])
        tmp_feats.append(feats)
    f.close()
    return tmp_feats


def map_feat(featname_all, featname_file, ft):
    '''
    :param featname_all: list of all featnames
    :param featname_file: list of featnames in current ego network
    :param ft: features in current ego domains, list
    :return map_ft: features in all features domain
    '''
    map_ft = [0]*len(featname_all)
    for i in range(len(featname_file)):
        ind = featname_all.index(featname_file[i])
        map_ft[ind] = int(ft[i])
    return map_ft


def count_locale(featname_all, ft_all):
    featname_all = np.array(featname_all)
    locale_ind = np.char.find(featname_all, "locale") >=0
    locale_info = ft_all[:, locale_ind]
    locale_sum = locale_info.sum(axis=0)
    print("localse info: ", locale_sum)
    # if max(locale_sum)>=0.8*max(locale_sum)


def handle_dup(ft_old, ft_new):
    for i in range(len(ft_old)):
        if ft_old[i] or ft_new[i]:
            ft_new[i] = 1
    return ft_new





if __name__ == "__main__":
    file = "facebook/facebook_combined.txt"
    f = open(file)
    links = []
    max_node = 0
    for line in f:
        line = line.strip().split(' ')
        max_node = max([int(line[0]), int(line[1]), max_node])
        links.append((int(line[0]), int(line[1])))
    f.close()
    g = Graph()
    g.add_vertices(max_node+1)
    g.add_edges(links)

    features = [[]] * (max_node+1)
    files = glob.glob('facebook/*featnames')
    ft_names = set([])
    for file in files:
        tmp_feats = load_featname(file)
        ft_names = set.union(set(tmp_feats), ft_names)
    ft_names = list(ft_names)
    ft_names.sort(key=lambda x: [x.split(" ")[0], int(x.split(" ")[-1])])



    files = glob.glob('facebook/*feat')
    feature_check = [False]*(max_node+1)
    for file in files:
        count = 0
        f = open(file)
        ego_id = file.split('\\')[1].split('.')[0]
        featname_file = "facebook/{}.featnames".format(ego_id)
        feat_name = load_featname(featname_file)
        ft_all = []
        for line in f:

            if "egofeat" in file:
                id, ft = int(file.split('\\')[1].split('.')[0]), line.strip().split(' ')
            else:
                id, ft = int(line.strip().split(' ')[0]), line.strip().split(' ')[1:]
            ft_map = map_feat(ft_names, feat_name, ft)
            if len(features[id]) >0 and features[id] != ft_map:
                print("duplicate unmatch for id={}".format(id))
                features[id] = handle_dup(features[id], ft_map)
            else:
                features[id] = ft_map
            feature_check[id] = True
            count+=1
            ft_all.append(ft_map)
        ft_all = np.array(ft_all)
        count_locale(ft_names, ft_all)
        print("There are {} lines in file {}".format(count, file))

    features = np.array(features)
    adj = g.get_adjacency_sparse()
    with open('facebook/combine-adj-feat.pkl', 'wb') as f:
        pkl.dump((adj, features), f)
    with open("facebook/combine.featnames", "a") as wf:
        for i in range(len(ft_names)):
            wf.write(
                "{} {}\n".format(i, ft_names[i])
            )




