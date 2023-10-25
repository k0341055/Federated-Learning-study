#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
from collections import Counter

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    labels = np.array(dataset.targets)
    users_label ,dict_users, all_idxs = {}, {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = np.random.choice(all_idxs, num_items,
                                             replace=False)
        label_count = Counter(labels[dict_users[i]].tolist())
        label_count_dict = {int(label): count for label, count in label_count.items()} # if count / len(dict_users[i]) >= 0.1
        users_label[i] =  list(label_count_dict.keys())
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users , users_label

def mnist_noniid_one(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs --> assign each user 1 shard eg: 600 imgs/shard X 100 users(shards)
    num_imgs = int(len(dataset)/num_users)
    idx_shard = [i for i in range(num_users)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    users_label = {i: np.array([]) for i in range(num_users)}
    labels = np.array(dataset.targets)
    avg_len = int(len(dataset)/10)
    idxs = np.array([])
    # resample each labels dataset to 6000
    for i in range(10):
        label_len = Counter(labels)[i]
        label_idx = np.where(labels == i)[0]
        if label_len < avg_len:
            add_idx = np.random.choice(label_idx, avg_len - label_len, replace=False)
            label_idx = np.concatenate([label_idx, add_idx])
        else:
            label_idx = np.random.choice(label_idx, avg_len, replace=False)
        np.random.shuffle(label_idx)
        idxs = np.concatenate([idxs,label_idx])
    idxs = idxs.astype(int)
    id_labels = labels[idxs]
    # idxs = np.arange(len(dataset))

    # # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs = idxs_labels[0, :]
    # id_labels = idxs_labels[1, :]
    # divide and assign 1 shards to client
    for i in range(num_users):
        rnd_shard = np.random.choice(idx_shard, 1, replace=False)
        idx_shard = np.delete(idx_shard, np.where(idx_shard == rnd_shard))
        dict_users[i] = idxs[rnd_shard[0]*num_imgs:(rnd_shard[0]+1)*num_imgs]
        label_ls = id_labels[rnd_shard[0]*num_imgs:(rnd_shard[0]+1)*num_imgs].tolist()
        # 使用 Counter 計算元素的出現次數
        label_count = Counter(label_ls)
        # 將 Counter 轉換為字典格式
        label_count_dict = {int(label): count for label, count in label_count.items()} # if count / label_ls.shape[0] >= 0.1
        users_label[i] =  list(label_count_dict.keys())
    return dict_users , users_label

def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  assign each user 2 shard eg: 300 imgs/shard X 2 shards X 100 users
    num_imgs = int(len(dataset)/(2*num_users))
    idx_shard = [i for i in range(2*num_users)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    users_label = {i: np.array([]) for i in range(num_users)}
    labels = np.array(dataset.targets)
    
    avg_len = int(len(dataset)/10)
    idxs = np.array([])
    for i in range(10):
        label_len = Counter(labels)[i]
        label_idx = np.where(labels == i)[0]
        if label_len < avg_len:
            add_idx = np.random.choice(label_idx, avg_len - label_len, replace=False)
            label_idx = np.concatenate([label_idx, add_idx])
        else:
            label_idx = np.random.choice(label_idx, avg_len, replace=False)
        np.random.shuffle(label_idx)
        idxs = np.concatenate([idxs,label_idx])
    idxs = idxs.astype(int)
    id_labels = labels[idxs]
    
    # idxs = np.arange(len(dataset))
    # # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs = idxs_labels[0, :]
    # id_labels = idxs_labels[1, :]
    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        label_ls = np.array([])
        for rand in rand_set:
            label_ls = np.concatenate([label_ls, id_labels[rand*num_imgs:(rand+1)*num_imgs]],axis=0)
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        # 使用 Counter 計算元素的出現次數
        label_count = Counter(label_ls.tolist())
        # 將 Counter 轉換為字典格式
        label_count_dict = {int(label): count for label, count in label_count.items()} # if count / label_ls.shape[0] >= 0.1
        users_label[i] =  list(label_count_dict.keys())
    return dict_users , users_label


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, int(len(dataset)/1200)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    users_label = {}

    labels = np.array(dataset.targets)
    avg_len = int(len(dataset)/10)
    idxs = np.array([])
    # resample each labels dataset to 6000
    for i in range(10):
        label_len = Counter(labels)[i]
        label_idx = np.where(labels == i)[0]
        if label_len < avg_len:
            add_idx = np.random.choice(label_idx, avg_len - label_len, replace=False)
            label_idx = np.concatenate([label_idx, add_idx])
        else:
            label_idx = np.random.choice(label_idx, avg_len, replace=False)
        np.random.shuffle(label_idx)
        idxs = np.concatenate([idxs,label_idx])
    idxs = idxs.astype(int)
    id_labels = labels[idxs]
    # idxs = np.arange(len(dataset))
    # # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs = idxs_labels[0, :]
    # id_labels = idxs_labels[1, :]
    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    for i in range(num_users):
        index_ls = np.where(np.isin(idxs, dict_users[i]))[0]
        label_count = Counter(id_labels[index_ls].tolist())
        label_count_dict = {int(label): count for label, count in label_count.items()} # if count / len(dict_users[i]) >= 0.1
        users_label[i] =  list(label_count_dict.keys())
    return dict_users , users_label


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(60000/num_users)
    users_label, dict_users, all_idxs = {}, {}, [i for i in range(len(dataset))]
    labels = np.array(dataset.targets)
    add_idx = np.random.choice(all_idxs, 60000-len(dataset),replace=False)
    all_idxs = np.concatenate([all_idxs,add_idx])
    np.random.shuffle(all_idxs)
    for i in range(num_users):
        # dict_users[i] = set(np.random.choice(all_idxs, num_items,
        #                                      replace=False))
        dict_users[i] = all_idxs[i*num_items:(i+1)*num_items]
        label_count = Counter(labels[np.array(list(dict_users[i]))].tolist())
        label_count_dict = {int(label): count for label, count in label_count.items()} # if count / label_ls.shape[0] >= 0.1
        users_label[i] =  list(label_count_dict.keys())
        # all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users , users_label


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_imgs = int(60000/(num_users*2)) # len(dataset)
    idx_shard = [i for i in range(num_users*2)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    users_label = {i: np.array([]) for i in range(num_users)}
    labels = np.array(dataset.targets)
    idxs = np.array([])
    for i in range(10):
        label_idx = np.where(labels == i)[0]
        add_idx = np.random.choice(label_idx, 6000-Counter(labels)[i], replace=False)
        label_idx = np.concatenate([label_idx, add_idx])
        np.random.shuffle(label_idx)
        idxs = np.concatenate([idxs,label_idx])
    idxs = idxs.astype(int)
    id_labels = labels[idxs]
    
    # idxs = np.arange(len(dataset))
    # # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs = idxs_labels[0, :]
    # id_labels = idxs_labels[1, :]
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        label_ls = np.array([])
        for rand in rand_set:
            label_ls = np.concatenate([label_ls, id_labels[rand*num_imgs:(rand+1)*num_imgs]],axis=0)
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        label_count = Counter(label_ls.tolist())
        label_count_dict = {int(label): count for label, count in label_count.items()} # if count / label_ls.shape[0] >= 0.1
        users_label[i] =  list(label_count_dict.keys())
    return dict_users , users_label

def cifar_noniid_one(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_imgs = int(60000/num_users) # len(dataset)
    idx_shard = [i for i in range(num_users)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    users_label = {i: np.array([]) for i in range(num_users)}
    labels = np.array(dataset.targets)
    idxs = np.array([])
    for i in range(10):
        label_idx = np.where(labels == i)[0]
        add_idx = np.random.choice(label_idx, 6000-Counter(labels)[i], replace=False)
        label_idx = np.concatenate([label_idx, add_idx])
        np.random.shuffle(label_idx)
        idxs = np.concatenate([idxs,label_idx])
    idxs = idxs.astype(int)
    id_labels = labels[idxs]
    
    # idxs = np.arange(len(dataset))
    # # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs = idxs_labels[0, :]
    # id_labels = idxs_labels[1, :]
    # divide and assign one shard to client
    for i in range(num_users):
        rnd_shard = np.random.choice(idx_shard, 1, replace=False)
        idx_shard = np.delete(idx_shard, np.where(idx_shard == rnd_shard))
        dict_users[i] = idxs[rnd_shard[0]*num_imgs:(rnd_shard[0]+1)*num_imgs]
        label_ls = id_labels[rnd_shard[0]*num_imgs:(rnd_shard[0]+1)*num_imgs].tolist()
        # 使用 Counter 計算元素的出現次數
        label_count = Counter(label_ls)
        # 將 Counter 轉換為字典格式
        label_count_dict = {int(label): count for label, count in label_count.items()} # if count / label_ls.shape[0] >= 0.1
        users_label[i] =  list(label_count_dict.keys())
    return dict_users , users_label

def cifar_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 50,000 training imgs --> 50 imgs/shard X 1000 shards
    num_shards, num_imgs = 1000, int(60000/1000) # len(dataset)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    users_label = {}
    
    labels = np.array(dataset.targets)
    idxs = np.array([])
    for i in range(10):
        label_idx = np.where(labels == i)[0]
        add_idx = np.random.choice(label_idx, 6000-Counter(labels)[i], replace=False)
        label_idx = np.concatenate([label_idx, add_idx])
        np.random.shuffle(label_idx)
        idxs = np.concatenate([idxs,label_idx])
    idxs = idxs.astype(int)
    id_labels = labels[idxs]

    # idxs = np.arange(len(dataset))
    # # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs = idxs_labels[0, :]
    # id_labels = idxs_labels[1, :]
    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    for i in range(num_users):
        index_ls = np.where(np.isin(idxs, dict_users[i]))[0]
        label_count = Counter(id_labels[index_ls].tolist())
        label_count_dict = {int(label): count for label, count in label_count.items()} # if count / len(dict_users[i]) >= 0.1
        users_label[i] =  list(label_count_dict.keys())
    return dict_users , users_label


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d,u = mnist_noniid(dataset_train, num)
