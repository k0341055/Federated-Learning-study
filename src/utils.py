#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_one, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid, cifar_noniid_one, cifar_noniid_unequal
from collections import OrderedDict, defaultdict
import random

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups, users_label = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups, users_label = cifar_noniid_unequal(train_dataset, args.num_users)
            elif args.one:
                # Chose one shard for every user
                user_groups , users_label = cifar_noniid_one(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups, users_label = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups , users_label = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups , users_label  = mnist_noniid_unequal(train_dataset, args.num_users)
            elif args.one:
                # Chose one shard for every user
                user_groups , users_label = mnist_noniid_one(train_dataset, args.num_users)
            else:
                # Chose euqal splits(2 shards) for every user
                user_groups , users_label = mnist_noniid(train_dataset, args.num_users)
    return train_dataset, test_dataset, user_groups , users_label

def average_weights(w , n): #weighted by client's train dataset
    ndata = sum(n)
    w_avg = OrderedDict()
    for key in copy.deepcopy(w[0]).keys():
        w_avg[key] = w[0][key]*(n[0]/ndata)
        for i in range(1,len(w)):
            w_avg[key] += w[i][key]*(n[i]/ndata)
    return w_avg

def org_average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        # print('{}.shape: {}'.format(key,w_avg[key].size()))
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

# def kmeans(X, clients_ndata,clients ,centers,args):
#     device = 'cuda:0' if args.gpu else 'cpu'
#     # 計算同一個cluster中的clients的model weights平均
#     new_centers = OrderedDict()
#     layers_labels = []
#     layers_weights =[]
#     if centers == 1:
#         centers = random.sample(copy.deepcopy(X), args.num_clusters)
#     num_clients = len(X)
#     K = len(centers)
#     for layer_name in X[0].keys():
#         clients_layer_weights = [client_weights[layer_name] for client_weights in X]
#         centers_layer_weights = [center_weights[layer_name] for center_weights in centers]
#         clients_lw = torch.stack(clients_layer_weights).cpu() # reduce gpu memory consumption
#         centers_lw = torch.stack(centers_layer_weights).cpu() # reduce gpu memory consumption
#         l_dim = tuple(range(1,len(list(clients_lw.size()))+1))[1:] # eg: (2,3,4,5) or (2,3,4) ...
#         # 計算每個client的某layer到各群集中心的歐式距離
#         for i in range(args.max_iters):
#             distances = torch.sqrt(torch.sum((clients_lw[:, None] - centers_lw) ** 2, dim=l_dim)) # 計算所有每個客戶端單個layer的weight和各個center對應layer的weight的歐式距離
#             labels = torch.argmin(distances, dim=1) # 找到每個client"在此layer"的weight距離最近的center(cluster)
#             layers_labels.append(labels.tolist()) #搜集每個client在此layer的weight所屬的cluster
#             new_centers[layer_name] = torch.stack([clients_lw[labels == k].mean(dim=0) for k in range(K)]) # 計算各個cluster中的client"在此layer"的weight的平均
#             # 上述算法是以"layer"為主，生成K個global model

#             if torch.all(torch.eq(centers_lw, new_centers[layer_name])):
#                 break
#             centers_lw = new_centers[layer_name]
#         centers_split = [cw for cw in centers_lw if not torch.isnan(cw).any()]
#         # centers_split = [centers_lw[i] for i in range(centers_lw.size(0))]
#         layers_weights.append(centers_split)
#     # new_centers['conv1.weight'].size() = torch.Size([3, 32, 1, 5, 5]) 要轉成 centers_ls[0]['conv1.weight'].size() = torch.Size([32, 1, 5, 5])
#     centers_ls = [] 
#     for i in range(len(min(layers_weights, key=len))):
#         od = OrderedDict()
#         for j,key in enumerate(X[0].keys()):
#             od[key] = layers_weights[j][i].to(device) #for layers_weights: row is key , col is centers (transfer to gpu)
#         centers_ls.append(od)
            
#     if len(centers_ls) == 1:
#         labels_revise = [0]*num_clients
#         weight_membership = defaultdict(list)
#         weight_membership[0] = clients
#     else:
#         # 可決定各layer的比重，判斷client統整個layer後，所屬的cluster
#         if args.k_vote != 'eq': 
#             v = [2 if 'fc' not in k and 'bias' not in k else 0 for k in X[0].keys()]
#             for i in range(len(v)):
#                 for j in range(v[i]):
#                     layers_labels.append(layers_labels[i])
#         #以"client"為主，將所選clients指派到K個cluster
#         weight_labels = [max(client,key=client.count) for client in zip(*layers_labels)]
#         #校正labels至int [0,len(centers)]
#         labels_ls = list(set(weight_labels))
#         labels_revise = [labels_ls.index(label) for label in weight_labels]
#         # 指派client至對應的cluster
#         weight_membership = defaultdict(list)
#         [weight_membership[label].append(idx) for idx, label in zip(clients,labels_revise)]
#     return centers_ls , labels_revise , dict(weight_membership)

def kmeans_flat(w , clients_ndata, clients , centers , args):
    device = 'cuda:0' if args.gpu else 'cpu'
    w_ls = []
    ndata = sum(clients_ndata)
    # gm = average_weights(w,clients_ndata)
    if args.km == 'w':
        for client_weights , client_ndata in zip(w,clients_ndata):
            w_flat = torch.cat([l.view(-1) for l in client_weights.values()]) * (client_ndata / ndata)
            w_ls.append(w_flat)
    else:
        for client_weights in w:
            w_flat = torch.cat([l.view(-1) for l in client_weights.values()])
            w_ls.append(w_flat)
    # gm_flat = torch.cat([l.view(-1) for l in gm.values()])
    # gm_flat = gm_flat.to('cpu')
    X = torch.stack(w_ls)
    X = X.to('cpu') # reduce gpu memory consumption
    
    # 隨機初始化群集中心
    if centers == 1:
        centers = copy.deepcopy(X[torch.randperm(X.size(0))[:args.num_clusters]])
        # centers = centers.tolist()
        # centers.append(gm_flat)
        # centers = torch.stack(centers)
        K = args.num_clusters
    else:
        K = len(centers)
        c_ls = []
        for centers_weights in centers:
            c_flat = torch.cat([l.view(-1) for l in centers_weights.values()])
            c_ls.append(c_flat)
        centers = torch.stack(copy.deepcopy(c_ls))
        centers = centers.to('cpu')
    # if K == 1:
    #     return [average_weights(w,clients_ndata)] , [0]*len(clients) , {0:clients}
    for i in range(args.max_iters):
        # X_cpu = X.to(device) # reduce gpu memory consumption
        # 計算每個client到各群集中心的歐式距離
        distances = torch.sqrt(torch.sum((X[:, None] - centers) ** 2, dim=2)) #X_cpu 
        
        # 找到最近的群集中心
        labels = torch.argmin(distances, dim=1)
        # 更新群集中心
        if args.km == 'w':
            new_centers = torch.stack([X[labels == k].sum(dim=0) for k in range(K)])
        else:
            new_centers = torch.stack([X[labels == k].mean(dim=0) for k in range(K)])

        # 檢查是否收斂
        if torch.all(torch.eq(centers, new_centers)):
            break
        centers = new_centers
    
    #校正labels至int[0,len(centers)]
    labels_ls = list(set(labels.tolist()))
    labels_revise = [labels_ls.index(label) for label in labels.tolist()]
    # 指派client至對應的cluster
    membership = defaultdict(list)
    [membership[label].append(idx) for idx, label in zip(clients,labels_revise)]
    
    #當centers存在nan時，則center_revise只存放無nan的centers
    center_revise = torch.stack([cw for cw in centers if not torch.isnan(cw).any()])
    
    # 將flatten過的centers按K(num_cluster)，逐一還原成OrderedDict([('layer',tensor(layer))])
    layer_name = w[0].keys()
    layer_size = [v.size() for v in w[0].values()]
    centers_ls = []
    for i in range(center_revise.size(0)):
        start_idx=0
        centers_od = OrderedDict()
        for k,s in zip(layer_name,layer_size):
            numel = s.numel()
            centers_od[k] = center_revise[i][start_idx:start_idx + numel].view(s).to(device) # transfer to gpu
            start_idx += numel
        centers_ls.append(centers_od)
        
    return centers_ls , labels_revise , dict(membership)

# def kmeans_flat(w , clients_ndata, clients , centers , args):
#     device = 'cuda:0' if args.gpu else 'cpu'
#     w_ls = []
#     for client_weights in w:
#         w_flat = torch.cat([l.view(-1) for l in client_weights.values()])
#         w_ls.append(w_flat)
#     X = torch.stack(w_ls)
#     X = X.to('cpu') # reduce gpu memory consumption
#     num_samples = X.size(0)
#     num_features = X.size(1)
    
#     # 隨機初始化群集中心
#     if centers == 1:
#         centers = copy.deepcopy(X[torch.randperm(num_samples)[:args.num_clusters]])
#         K = args.num_clusters
#     else:
#         K = len(centers)
#         c_ls = []
#         for centers_weights in centers:
#             c_flat = torch.cat([l.view(-1) for l in centers_weights.values()])
#             c_ls.append(c_flat)
#         centers = torch.stack(copy.deepcopy(c_ls))
#         centers = centers.to('cpu')
    
#     for i in range(args.max_iters):
#         # X_cpu = X.to(device) # reduce gpu memory consumption
#         # 計算每個client到各群集中心的歐式距離
#         distances = torch.sqrt(torch.sum((X[:, None] - centers) ** 2, dim=2)) #X_cpu 
        
#         # 找到最近的群集中心
#         labels = torch.argmin(distances, dim=1)
#         # 更新群集中心
#         new_centers = torch.stack([X[labels == k].mean(dim=0) for k in range(K)])

#         # 檢查是否收斂
#         if torch.all(torch.eq(centers, new_centers)):
#             break
#         centers = new_centers
    
#     #校正labels至int[0,len(centers)]
#     labels_ls = list(set(labels.tolist()))
#     labels_revise = [labels_ls.index(label) for label in labels.tolist()]
#     # 指派client至對應的cluster
#     membership = defaultdict(list)
#     [membership[label].append(idx) for idx, label in zip(clients,labels_revise)]
    
#     #當centers存在nan時，則center_revise只存放無nan的centers
#     center_revise = torch.stack([cw for cw in centers if not torch.isnan(cw).any()])
    
#     # 將flatten過的centers按K(num_cluster)，逐一還原成OrderedDict([('layer',tensor(layer))])
#     layer_name = w[0].keys()
#     layer_size = [v.size() for v in w[0].values()]
#     centers_ls = []
#     for i in range(center_revise.size(0)):
#         start_idx=0
#         centers_od = OrderedDict()
#         for k,s in zip(layer_name,layer_size):
#             numel = s.numel()
#             centers_od[k] = center_revise[i][start_idx:start_idx + numel].view(s).to(device) # transfer to gpu
#             start_idx += numel
#         centers_ls.append(centers_od)
        
#     return centers_ls , labels_revise , dict(membership)


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning rate : {args.lr}')
    # print(f'    Global Rounds   : {args.epochs}\n') 7/24
    print(f'    Target test accuracy  : {args.test_acc}') # 7/24
    
    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
        if args.unequal:
            print('    Unbalanced')
        elif args.one:
            print('    One shard')
    if args.num_clusters:
        print(f'    Lambda  : {args.lam}')
        print(f'    Number of clusters  : {args.num_clusters}')
    print(f'    Number of users  : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
