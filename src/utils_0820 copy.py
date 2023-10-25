#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import json
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal , mnist_noniid_one
from sampling import cifar_iid, cifar_noniid, cifar_noniid_one,cifar_noniid_unequal
from myDataLoader import bmtDataset
from collections import OrderedDict, defaultdict
from sklearn.cluster import DBSCAN
import numpy as np
import random

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        # apply_transform = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        apply_transform = transforms.Compose([
        transforms.RandomCrop(24),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(contrast=0.2, brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
# def org_dbscan(w , clients_ndata, clients, args):
#     device = 'cuda:0' if args.gpu else 'cpu'
#     w_ls = []
#     ndata = sum(clients_ndata)
#     num_vectors = len(w)
#     dist_matrix = np.zeros((num_vectors, num_vectors))
#     for i, client_ndata in enumerate(clients_ndata):
#         if args.w :
#             w_flat = torch.cat([l.view(-1) for l in w[i].values()]) * (client_ndata / ndata)
#         else:
#             w_flat = torch.cat([l.view(-1) for l in w[i].values()])
#         w_ls.append(w_flat)
#         for j in range(i + 1, num_vectors):
#             total_dist = 0
#             for (k,i_v),j_v in zip(w[i].items(),w[j].values()):
#                 total_dist += torch.norm(i_v - j_v)
#             dist_matrix[i, j] = total_dist.to('cpu').numpy()
#             dist_matrix[j, i] = dist_matrix[i, j]
#     X = torch.stack(w_ls).to('cpu')
#     # dc = round(np.percentile(dist_matrix[np.triu_indices(num_vectors, k=1)],1),4)
#     # 動態調整dc
#     mean_dist = np.mean(dist_matrix[np.triu_indices(num_vectors, k=1)])
#     std_dist = np.std(dist_matrix[np.triu_indices(num_vectors, k=1)])
#     dc = mean_dist + 0.5 * std_dist
    
#     # dbscan
#     min_samples = max(int(num_vectors/10),1)
#     dbscan = DBSCAN(eps=dc, min_samples=min_samples, metric='precomputed')
#     labels = dbscan.fit_predict(dist_matrix)
#     if args.w :
#         centers = torch.stack([X[labels == k].sum(dim=0) for k in range(max(labels)+1)])
#     else:
#         centers = torch.stack([X[labels == k].mean(dim=0) for k in range(max(labels)+1)])
#     #校正labels至int[0,len(centers)]，並保留-1(noise)
#     labels_ls = list(set([label for label in labels.tolist() if label != -1]))
#     labels_revise = [-1 if label == -1 else labels_ls.index(label) for label in labels.tolist()]
#     # 指派client至對應的cluster
#     membership = defaultdict(list)
#     [membership[label].append(idx) for idx, label in zip(clients,labels_revise)]
    
#     #當centers存在nan時，則center_revise只存放無nan的centers
#     filter_centers = [cw for cw in centers if not torch.isnan(cw).any()] #  and not (cw == 0).any()
#     if len(filter_centers) != 0:
#         center_revise = torch.stack(filter_centers)
#     else:
#         # Handle the case where the list is empty, maybe assign a default value or raise an error.
#         print(f'center has nan: {centers}')
#     # center_revise = torch.stack([cw for cw in centers if not torch.isnan(cw).any() and not (cw == 0).any()])
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
def dbscan_cluster(w , clients_ndata, clients, args):
    device = 'cuda:0' if args.gpu else 'cpu'
    w_ls = []
    ndata = sum(clients_ndata)
    num_vectors = len(w)
    dist_matrix = np.zeros((num_vectors, num_vectors))
    for i, client_ndata in enumerate(clients_ndata):
        if args.w :
            w_flat = torch.cat([l.view(-1) for l in w[i].values()]) * (client_ndata / ndata)
        else:
            w_flat = torch.cat([l.view(-1) for l in w[i].values()])
        w_ls.append(w_flat)
        for j in range(i + 1, num_vectors):
            total_dist = 0
            for (k,i_v),j_v in zip(w[i].items(),w[j].values()):
                lw = 0.8 if 'conv' in k else 0.2
                ww = 0.4 if 'weight' in k else 0.1
                total_dist += torch.norm(i_v - j_v)*ww*lw
            dist_matrix[i, j] = total_dist.to('cpu').numpy()
            dist_matrix[j, i] = dist_matrix[i, j]
    X = torch.stack(w_ls).to('cpu')
    # dc = round(np.percentile(dist_matrix[np.triu_indices(num_vectors, k=1)],args.dc_per),4)
    # 動態調整dc
    # mean_dist = np.mean(dist_matrix[np.triu_indices(num_vectors, k=1)])
    # std_dist = np.std(dist_matrix[np.triu_indices(num_vectors, k=1)])
    # dc = mean_dist + 0.5 * std_dist
    dc = max(np.min(np.where(dist_matrix != 0, dist_matrix, np.inf), axis=1))
    # dbscan
    min_samples = max(int(num_vectors/10),1)
    dbscan = DBSCAN(eps=dc, min_samples=min_samples, metric='precomputed')
    labels = dbscan.fit_predict(dist_matrix)
    if args.w :
        centers = torch.stack([X[labels == k].sum(dim=0) for k in range(max(labels)+1)])
    else:
        centers = torch.stack([X[labels == k].mean(dim=0) for k in range(max(labels)+1)])
    #校正labels至int[0,len(centers)]，並保留-1(noise)
    labels_ls = list(set([label for label in labels.tolist() if label != -1]))
    labels_revise = [-1 if label == -1 else labels_ls.index(label) for label in labels.tolist()]
    # 指派client至對應的cluster
    membership = defaultdict(list)
    [membership[label].append(idx) for idx, label in zip(clients,labels_revise)]
    
    #當centers存在nan時，則center_revise只存放無nan的centers
    filter_centers = [cw for cw in centers if not torch.isnan(cw).any() and not (cw == 0).all()] 
    if len(filter_centers) != 0:
        center_revise = torch.stack(filter_centers)
    else:
        # Handle the case where the list is empty, maybe assign a default value or raise an error.
        print(f'center has nan: {centers}')
    # center_revise = torch.stack([cw for cw in centers if not torch.isnan(cw).any() and not (cw == 0).any()])
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

# def dfs(node, visited, neighbors_id):
#     cluster = []
#     stack = [node]
#     while stack:
#         current = stack.pop()
#         if current not in visited:
#             visited.add(current)
#             cluster.append(current)
#             stack.extend(neighbors_id[current])
#     return cluster

def dfs(node, visited, adj_matrix):
    cluster = []
    stack = [node]
    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            cluster.append(current)
            neighbors = np.where(adj_matrix[current] == 1)[0]
            stack.extend(neighbors)
    return cluster

def min_distance_between_clusters(cluster1, cluster2, dist_matrix):
    distances = dist_matrix[np.ix_(cluster1, cluster2)]
    return distances.min()

def dfs_cluster(w , clients_ndata, clients, args):
    device = 'cuda:0' if args.gpu else 'cpu'
    w_ls = []
    ndata = sum(clients_ndata)
    num_vectors = len(w)
    labels = np.full(num_vectors, -1)
    dist_matrix = np.zeros((num_vectors, num_vectors))
    for i, client_ndata in enumerate(clients_ndata):
        if args.w :
            w_flat = torch.cat([l.view(-1) for l in w[i].values()]) * (client_ndata / ndata)
        else:
            w_flat = torch.cat([l.view(-1) for l in w[i].values()])
        w_ls.append(w_flat)
        for j in range(i + 1, num_vectors):
            total_dist = 0
            for (k,i_v),j_v in zip(w[i].items(),w[j].values()):
                lw = 0.8 if 'conv' in k else 0.2
                ww = 0.4 if 'weight' in k else 0.1
                total_dist += torch.norm(i_v - j_v)*ww*lw
            dist_matrix[i, j] = total_dist.to('cpu').numpy()
            dist_matrix[j, i] = dist_matrix[i, j]
    X = torch.stack(w_ls).to('cpu')

    # 動態調整dc(使用平均距離和標準差來動態調整dc)
    # 使 dc 更具適應性，以反映數據的實際分佈)
    dc = max(np.min(np.where(dist_matrix != 0, dist_matrix, np.inf), axis=1))
    # mean_dist = np.mean(dist_matrix[np.triu_indices(num_vectors, k=1)])
    # std_dist = np.std(dist_matrix[np.triu_indices(num_vectors, k=1)])
    # dc = mean_dist # + 0.5 * std_dist
    print(f"dc: {round(dc,4)}")
    # 考慮節點的密度和中心性(該點與其他點的平均距離)
    densities = np.sum(dist_matrix <= dc, axis=1)
    # centralities = np.mean(dist_matrix, axis=1)
    # centralities = np.zeros(num_vectors)
    # for i in range(num_vectors):
    #     # 找到距離小於或等於dc的鄰居，但排除自己
    #     neighbors_id = np.where((dist_matrix[i] <= dc) & (dist_matrix[i] > 0))[0]
    #     centralities[i] = np.mean(dist_matrix[i, neighbors_id])
    mask = (dist_matrix <= dc) & (dist_matrix > 0)
    centralities = np.sum(dist_matrix * mask, axis=1) / np.sum(mask, axis=1)
    neighbors_id = [np.where(row)[0] for row in mask]

    # 使用密度和局部中心性來更新鄰接矩陣
    adj_matrix = np.zeros_like(dist_matrix)
    for i in range(num_vectors):
        if densities[i] >= np.median(densities) and centralities[i] <= np.median(centralities):
            # 找到距離最近的10個點
            # closest_indices = np.intersect1d(dist_matrix[i].argsort()[1:args.near+1], np.where(dist_matrix[i] <= dc)[0])
            adj_matrix[i, neighbors_id[i]] = 1 # closest_indices
    # zero_indices = np.argwhere(adj_matrix == 0)
    # print(f'number of indices in adj_matrix is zero: {len(zero_indices)}')
    
    # DFS algo
    visited = set()
    clusters = []
    shuffle_idx = np.random.permutation(num_vectors) # add random start point
    for i in shuffle_idx:
        if i not in visited:
            cluster = dfs(i, visited, adj_matrix)
            clusters.append(cluster)
    # print(f'clusters: {clusters}')

    # 計算每個小群的中心
    # cluster_centers = [all_points[np.argmin(centralities[all_points])] for all_points in [np.append(neighbors, i) for i, neighbors in enumerate(neighbors_id)]]
    # cluster_centers = [cluster[np.argmin(np.mean(dist_matrix[cluster][:, cluster], axis=1))] for cluster in valid_clusters] # clusters
    # cluster_centers = [np.mean(dist_matrix[cluster], axis=0) for cluster in valid_clusters] # clusters
    # 合併距離較近的小群
    merged_clusters = []
    merged = set()
    random.shuffle(clusters) # add shuffle for merged
    for i, cluster1 in enumerate(clusters):
        if i in merged:
            continue
        close_clusters = [i]
        for j, cluster2 in enumerate(clusters):
            if j != i and j not in merged:
                distance = min_distance_between_clusters(cluster1, cluster2, dist_matrix) # distance = np.linalg.norm(center1 - center2)
                # print(f'clust_dist: {round(distance,4)}')
                if distance < dc:  # 使用先前計算的dc作為閾值
                    close_clusters.append(j)
                    merged.add(j)
        merged_cluster = []
        for idx in close_clusters:
            merged_cluster.extend(clusters[idx])
        merged_clusters.append(merged_cluster)
    clusters = merged_clusters
    # 後處理：合併小群和移除雜訊
    min_cluster_size = 3  # 可以根據需要調整
    valid_clusters = []
    noise = []
    for cluster in clusters:
        if len(cluster) >= min_cluster_size:
            valid_clusters.append(cluster)
        else:
            noise.extend(cluster)
    print(f'valid clusters: {valid_clusters}')
    print(f'noise: {noise}')
    for cluster_idx, cluster in enumerate(valid_clusters):
        for p_idx in cluster:
            labels[p_idx] = cluster_idx
    print(f'labels: {labels}')
    # 輸出centers
    if args.w :
        centers = torch.stack([X[labels == k].sum(dim=0) for k in range(max(labels)+1)])
    else:
        centers = torch.stack([X[labels == k].mean(dim=0) for k in range(max(labels)+1)])
    #校正labels至int[0,len(centers)]，並保留-1(noise)
    labels_ls = list(set([label for label in labels.tolist() if label != -1]))
    labels_revise = [-1 if label == -1 else labels_ls.index(label) for label in labels.tolist()]
    # 指派client至對應的cluster
    membership = defaultdict(list)
    [membership[label].append(idx) for idx, label in zip(clients,labels_revise)]
    
    #當centers存在nan時，則center_revise只存放無nan的centers
    filter_centers = [cw for cw in centers if not torch.isnan(cw).any() and not (cw == 0).all()]
    if len(filter_centers) != 0:
        center_revise = torch.stack(filter_centers)
    else:
        print(f'center has nan: {centers}')
    # center_revise = torch.stack([cw for cw in centers if not torch.isnan(cw).any() and not (cw == 0).any()])
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

# def dfs_cluster(w , clients_ndata, clients, args):
#     device = 'cuda:0' if args.gpu else 'cpu'
#     w_ls = []
#     ndata = sum(clients_ndata)
#     num_vectors = len(w)
#     labels = np.full(num_vectors, -1)
#     dist_matrix = np.zeros((num_vectors, num_vectors))
#     for i, client_ndata in enumerate(clients_ndata):
#         if args.w :
#             w_flat = torch.cat([l.view(-1) for l in w[i].values()]) * (client_ndata / ndata)
#         else:
#             w_flat = torch.cat([l.view(-1) for l in w[i].values()])
#         w_ls.append(w_flat)
#         for j in range(i + 1, num_vectors):
#             total_dist = 0
#             for (k,i_v),j_v in zip(w[i].items(),w[j].values()):
#                 lw = 0.8 if 'conv' in k else 0.2
#                 ww = 0.4 if 'weight' in k else 0.1
#                 total_dist += torch.norm(i_v - j_v)*ww*lw
#             dist_matrix[i, j] = total_dist.to('cpu').numpy()
#             dist_matrix[j, i] = dist_matrix[i, j]
#     X = torch.stack(w_ls).to('cpu')
    
#     dc = round(np.percentile(dist_matrix[np.triu_indices(num_vectors, k=1)],args.dc_per),4)
#     adj_matrix = np.zeros_like(dist_matrix)
#     for i in range(num_vectors):
#         # 找到距離最近的10個點
#         closest_indices = np.intersect1d(dist_matrix[i].argsort()[1:args.near+1],np.where(dist_matrix[i] <= dc)[0])
#         # closest_indices = org_dist[i].argsort()[1:4]
#         adj_matrix[i, closest_indices] = 1

#     visited = set()
#     clusters = []
#     for i in range(adj_matrix.shape[0]):
#         if i not in visited:
#             cluster = dfs(i, visited, adj_matrix)
#             clusters.append(cluster)
#     # 篩選雜訊
#     valid_clusters = []
#     noise = []
#     for cluster in clusters:
#         if len(cluster) >= 3:
#             valid_clusters.append(cluster)
#         else:
#             noise.extend(cluster)
#     for cluster_idx, cluster in enumerate(valid_clusters):
#         for p_idx in cluster:
#             labels[p_idx] = cluster_idx
#     # print("Valid Clusters:", valid_clusters)
#     # print("Noise:", noise)
#     if args.w :
#         centers = torch.stack([X[labels == k].sum(dim=0) for k in range(max(labels)+1)])
#     else:
#         centers = torch.stack([X[labels == k].mean(dim=0) for k in range(max(labels)+1)])
#     #校正labels至int[0,len(centers)]，並保留-1(noise)
#     labels_ls = list(set([label for label in labels.tolist() if label != -1]))
#     labels_revise = [-1 if label == -1 else labels_ls.index(label) for label in labels.tolist()]
#     # 指派client至對應的cluster
#     membership = defaultdict(list)
#     [membership[label].append(idx) for idx, label in zip(clients,labels_revise)]
    
#     #當centers存在nan時，則center_revise只存放無nan的centers
#     filter_centers = [cw for cw in centers if not torch.isnan(cw).any()] #  and not (cw == 0).any()
#     if len(filter_centers) != 0:
#         center_revise = torch.stack(filter_centers)
#     else:
#         # Handle the case where the list is empty, maybe assign a default value or raise an error.
#         print(f'center has nan: {centers}')
#     # center_revise = torch.stack([cw for cw in centers if not torch.isnan(cw).any() and not (cw == 0).any()])
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
def org_kmeans(w , clients_ndata, clients , centers , args):
    device = 'cuda:0' if args.gpu else 'cpu'
    w_ls = []
    ndata = sum(clients_ndata)
    for client_weights , client_ndata in zip(w,clients_ndata):
        if args.w :
            w_flat = torch.cat([l.view(-1) for l in client_weights.values()]) * (client_ndata / ndata)
        else:
            w_flat = torch.cat([l.view(-1) for l in client_weights.values()])
        w_ls.append(w_flat)
    X = torch.stack(w_ls).to('cpu') # reduce gpu memory consumption
    # 隨機初始化群集中心
    if centers == 1:
        centers = torch.empty(args.num_clusters, X.size(1)).uniform_(torch.min(X), torch.max(X))
        # centers = copy.deepcopy(X[torch.randperm(X.size(0))[:args.num_clusters]]) #0917
        K = args.num_clusters
    else:
        K = len(centers)
        c_ls = []
        for centers_weights in centers:
            c_flat = torch.cat([l.view(-1) for l in centers_weights.values()])
            c_ls.append(c_flat)
        centers = torch.stack(copy.deepcopy(c_ls)).to('cpu')
    dist_ls = [] #0917
    for i in range(args.max_iters): #0917 , 5
        print(f'kmeans iter: {i}') #0917
        for i, cw in enumerate(centers): #0917
            print(f'center{i} weight: {cw}')
        # 計算每個client到各群集中心的歐式距離
        distances = torch.sqrt(torch.sum((X[:, None] - centers) ** 2, dim=2)) #X_cpu
        dist_ls.append(distances.numpy().tolist())
        # 找到最近的群集中心
        labels = torch.argmin(distances, dim=1)
        # 更新群集中心
        if args.w :
            new_centers = torch.stack([X[labels == k].sum(dim=0) for k in range(K)])
        else:
            new_centers = torch.stack([X[labels == k].mean(dim=0) for k in range(K)])
        
        if all(cw in centers for cw in new_centers): #0917
            break
        # found_match = False
        # for perm in torch.permutations(torch.arange(K)):
        #     if torch.all(torch.eq(centers, new_centers[perm])):
        #         found_match = True
        #         break
        # # 檢查是否收斂
        # if found_match:
        #     break
        # # 檢查是否收斂
        # if torch.all(torch.eq(centers, new_centers)):
        #     break
        centers = new_centers
    json.dump({'distance_from_centers':dist_ls},open('../save/json/{}/{}/test/dist.json'.format(args.dataset, args.algo),'w')) #0917
    #校正labels至int[0,len(centers)]
    labels_ls = list(set(labels.tolist()))
    labels_revise = [labels_ls.index(label) for label in labels.tolist()]
    # print(f'labels_revise: {labels_revise}')
    # 指派client至對應的cluster
    membership = defaultdict(list)
    [membership[label].append(idx) for idx, label in zip(clients,labels_revise)]
    
    #當centers存在nan時，則center_revise只存放無nan的centers
    filter_centers = [cw for cw in centers if not torch.isnan(cw).any() and not (cw == 0).all()]
    if len(filter_centers) != 0:
        center_revise = torch.stack(filter_centers)
    else:
        print(f'center has nan: {centers}')
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

def kmeans_flat(w , clients_ndata, clients , centers , args):
    device = 'cuda:0' if args.gpu else 'cpu'
    w_ls = []
    ndata = sum(clients_ndata)
    for client_weights , client_ndata in zip(w,clients_ndata):
        if args.w :
            w_flat = torch.cat([l.view(-1) for l in client_weights.values()]) * (client_ndata / ndata)
        else:
            w_flat = torch.cat([l.view(-1) for l in client_weights.values()])
        w_ls.append(w_flat)
    X = torch.stack(w_ls).to('cpu') # reduce gpu memory consumption
    # 隨機初始化群集中心
    if centers == 1:
        centers = copy.deepcopy(X[torch.randperm(X.size(0))[:args.num_clusters]])
        # centers = torch.randn((args.num_clusters, X.size(1)))
        K = args.num_clusters
    else:
        K = len(centers)
        c_ls = []
        for centers_weights in centers:
            c_flat = torch.cat([l.view(-1) for l in centers_weights.values()])
            c_ls.append(c_flat)
        centers = torch.stack(copy.deepcopy(c_ls)).to('cpu')
    for i in range(args.max_iters):
        # 計算每個client到各群集中心的歐式距離
        distances = torch.sqrt(torch.sum((X[:, None] - centers) ** 2, dim=2)) #X_cpu 
        # print(f'kmeans distance: {distances}')
        # 找到最近的群集中心
        labels = torch.argmin(distances, dim=1)
        # print(f'kmeans labels: {labels}')
        # 更新群集中心
        if args.w :
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
    # print(f'labels_revise: {labels_revise}')
    # 指派client至對應的cluster
    membership = defaultdict(list)
    [membership[label].append(idx) for idx, label in zip(clients,labels_revise)]
    
    #當centers存在nan時，則center_revise只存放無nan的centers
    filter_centers = [cw for cw in centers if not torch.isnan(cw).any() and not (cw == 0).all()]
    if len(filter_centers) != 0:
        center_revise = torch.stack(filter_centers)
    else:
        print(f'center has nan: {centers}')
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
    print(f'    Dataset     : {args.dataset}')
    if args.algo == 'fedavg':
        print(f'    Aggregation Algorithm : FedAVG')
    elif args.algo == 'org_kmeans':
        print(f'    Aggregation Algorithm : KMeans(FedSEM)')
    elif args.algo in 'kmeans':
        print(f'    Aggregation Algorithm : KMeans')
    elif args.algo in 'dbscan':
        print(f'    Aggregation Algorithm : DBSCAN')
    elif args.algo in 'dfs':
        print(f'    Aggregation Algorithm : DFS')
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
