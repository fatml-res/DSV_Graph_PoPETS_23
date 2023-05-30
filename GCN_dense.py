from __future__ import division
from __future__ import print_function

import time
import tensorflow.compat.v1 as tf

#from stealing_link.utils import *
from utils import *

from pygcn import GCN
from pygcn_tf.models import MLP
import pickle as pkl
tf.enable_eager_execution()
flags = tf.app.flags
FLAGS = flags.FLAGS # "data/dataset/walk2friends"
from sklearn.metrics import accuracy_score, precision_score, recall_score


def train_model(gender, ft, adj, labels, dataset, num_epoch, model_type, saving_path="gcn"):
    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)
    tf.compat.v1.disable_eager_execution()
    tf.keras.backend.clear_session()

    # Settings
    idx = list(range(len(labels)))
    np.random.shuffle(idx)

    # model parameters
    train_ratio = 0.6 # training set
    val_ratio = 0.2 # validation set
    dropout = 0.5 # dropout
    lr = 0.01 # learning rate
    wd = 5e-4 # weight decay
    hidden = 16 # number of hidden layer nodes
    patience = 20 # patience for early stopping



    idx_train = torch.LongTensor(idx[: int(len(labels) * train_ratio)])
    idx_val = torch.LongTensor(idx[int(len(labels) * train_ratio): int(len(labels) * (train_ratio +val_ratio))])
    idx_test = torch.LongTensor(idx[int(len(labels) * (train_ratio + val_ratio)):])

    train_mask = sample_mask(idx_train, labels.shape[0])  # index =1, others = 0
    val_mask = sample_mask(idx_val, labels.shape[0])  # index =1, others = 0
    test_mask = sample_mask(idx_test, labels.shape[0])
    if len(labels.shape) == 1:
        labels = one_hot_trans(labels)

    # index =1, others = 0

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[
                             train_mask, :]  # only the mask position has the true label, others are set to 0
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # Some preprocessing
    features = preprocess_features(ft)

    if model_type == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif model_type == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif model_type == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features':
            tf.sparse_placeholder(
                tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels':
            tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask':
            tf.placeholder(tf.int32),
        'dropout':
            tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero':
            tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features[2][1], lr=lr, wd=wd, hidden=hidden, logging=True)

    # Initialize session
    sess = tf.Session()

    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask,
                                            placeholders)
        outs_val = sess.run([model.loss, model.accuracy,
                             model.predict()],
                            feed_dict=feed_dict_val)
        preds = outs_val[2].argmax(axis=1)


        return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

    # Train model
    for epoch in range(num_epoch):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask,
                                        placeholders)
        feed_dict.update({placeholders['dropout']: dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy],
                        feed_dict=feed_dict)

        # Validation
        cost, acc, pred, duration = evaluate(features, support, y_val, val_mask,
                                             placeholders)
        cost_val.append(cost)
        # print(pred)
        # print(len(pred))

        # Print results
        print("Model:", model_type,  "Epoch:",
              '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=",
              "{:.5f}".format(cost), "val_acc=", "{:.5f}".format(acc), "time=",
              "{:.5f}".format(time.time() - t))

        if epoch > patience and cost_val[-1] > np.mean(
                cost_val[-(patience + 1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")


    # Testing
    test_cost, test_acc, test_pred, test_duration = evaluate(
        features, support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost), "accuracy=",
          "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    if y_test.shape[1] == 2:
        acc = accuracy_score(y_test.argmax(axis=1), test_pred.argmax(axis=1))
        prec = precision_score(y_test.argmax(axis=1), test_pred.argmax(axis=1))
        recall = recall_score(y_test.argmax(axis=1), test_pred.argmax(axis=1))

        print("Accuracy = {:.2%}, Precision={:.2%}, Recall = {:.2%}".format(acc, prec, recall))

    _, train_acc, _, _ = evaluate(
        features, support, y_train, train_mask, placeholders)
    if len(gender)>0:
        min_g, maj_g = np.unique(gender, return_counts=True)[0][np.unique(gender, return_counts=True)[1].argsort()]
        g1_idx = gender == min_g
        g2_idx = gender == maj_g
        g1_mask = sample_mask(g1_idx, labels.shape[0])
        g2_mask = sample_mask(g2_idx, labels.shape[0])

        _, train_g1_acc, _, _ = evaluate(
            features, support, y_train, train_mask * g1_mask, placeholders)

        _, train_g2_acc, _, _ = evaluate(
            features, support, y_train, train_mask * g2_mask, placeholders)

        _, test_g1_acc, _, _ = evaluate(
            features, support, y_test, test_mask * g1_mask, placeholders)

        _, test_g2_acc, _, _ = evaluate(
            features, support, y_test, test_mask * g2_mask, placeholders)

        acc_list = [train_acc, train_g1_acc, train_g2_acc, test_acc, test_g1_acc, test_g2_acc]
    else:
        acc_list = [train_acc, train_acc, train_acc, test_acc, test_acc, test_acc]
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    with open('{}/{}_{}_pred.pkl'.format(saving_path, dataset, model_type), 'wb') as f:
        pkl.dump(test_pred, f)
    with open('{}/{}_gcn_acc.pkl'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(acc_list, f)