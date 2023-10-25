#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training") #7/24
    parser.add_argument('--test_acc', type=float, default=0.99,
                        help="The test accuracy to be achieved (default: 0.99)") # add 7/24
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--num_clusters', type=int, default=0,
                        help='the number of cluster')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=str, default='10',
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', type=int,default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--one', type=int, default=0,
                        help='Default set to 2 shard in non-IID. Set to 1 for 1 shard in non-IID.')
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--load', type=int, default=0,
                        help='Default set to build new model. Set to 1 for loading exsiting model.')
    parser.add_argument('--load_model', type=str, default='global_model.pt',
                        help='load exsiting model name')
    parser.add_argument('--max_iters', type=int, default=1,
                    help='maximum rounds of counting distance for kmeans')
    parser.add_argument('--k_vote', type=str, default='eq',
                        help='the cluster voting method of kmeans')
    parser.add_argument('--w', type=int, default=1,
                        help='average or weighted average model weights in cluster')
    parser.add_argument('--algo', type=str, default='fedavg',
                        help='cluster algorithm') # 考量fedavg也適用此參數檔，方便共同記錄實驗細節
    # parser.add_argument('--dc_per', type=float, default=50,
    #                     help='cutoff distance percentile')
    # parser.add_argument('--near', type=int, default=10,
    #                     help='search nearest neighbor in dfs')
    parser.add_argument('--lam', type=float, default=0.5,
                        help='lambda controls the trade-off between local model and global model (range: [0,2], default: 0.5)')
    parser.add_argument('--add', type=int, default=0,
                        help='dynamic add external data')
    parser.add_argument('--add_ep', type=int, default=5,
                        help='dynamic add external data(epoch)')
    args = parser.parse_args()
    return args
