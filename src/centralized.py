#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import json
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Subset
from options import args_parser
from update import local_test_inference #, DatasetSplit
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar,MLPv2
from utils_0820 import get_dataset
from torchsummary import summary

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Dataset     : {args.dataset}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning rate : {args.lr}')
    print(f'    Data distribution:')
    if args.iid:
        print('        IID')
    else:
        print('        Non-IID')
        if args.unequal:
            print('        Unbalanced')
        elif args.one:
            print('        One shard')
    print(f'    Number of users  : {args.num_users}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    if args.load :
        global_model = torch.load(f'./save_model/{args.load_model}')
        global_model.eval()
        # 載入模型後，怎麼把模型的輸入參數和模型結果讀出?
        # print('whole global model load')
        # print(f' \n Results after {args.epochs} global rounds of training:')
        # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        # print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
        for parm in global_model.parameters():
            print(parm)
    else:
        if args.gpu:
            torch.cuda.set_device(0) #old:args.gpu
        device = 'cuda:0' if args.gpu else 'cpu' #old: cuda

        # load dataset and user groups
        train_dataset, test_dataset, user_groups , users_label = get_dataset(args)

        # BUILD MODEL
        if args.model == 'cnn':
            # Convolutional neural netork
            if args.dataset == 'mnist':
                global_model = CNNMnist(args=args)
                # total_params = [p.numel() for p in global_model.parameters()]
                # print(f"total parameters:{total_params}")
            elif args.dataset == 'fmnist':
                global_model = CNNFashion_Mnist(args=args)
            elif args.dataset == 'cifar':
                global_model = CNNCifar(args=args)

        elif args.model == 'mlp':
            # Multi-layer preceptron
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
                global_model = MLP(dim_in=len_in, dim_hidden=64,
                                dim_out=args.num_classes)
        elif args.model == 'MLPv2':
            global_model = MLPv2(dim_in=30,dim_hidden=64,dim_out=3)
        else:
            exit('Error: unrecognized model')

        # Set the model to train and send it to device.
        global_model.to(device) #use gpu or cpu
        # global_model.train() # switch model to train status
        print(global_model)
        init_gm = copy.deepcopy(global_model)
        summary(global_model,input_size=train_dataset[0][0].shape) #7/25
        # copy weights
        global_weights = global_model.state_dict()

        # Training
        # Set optimizer and criterion
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                        momentum=0.5)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                        weight_decay=1e-4)
        
        idxs = list(range(len(train_dataset)))
        split_idx = int(0.8 * len(train_dataset))
        idxs_train = idxs[:split_idx]
        idxs_val = idxs[split_idx:]
        train_loader = DataLoader(Subset(train_dataset, idxs_train), batch_size = int(args.local_bs), shuffle=True)
        valid_loader = DataLoader(Subset(train_dataset, idxs_val),batch_size= int(args.local_bs),shuffle=True)

        criterion = torch.nn.NLLLoss().to(device)
        all_test_acc = 0.0 # for first round
        users_acc , users_loss = {} , {}
        # Training
        epoch_loss = []
        global_model.train()
        for iter in range(args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = global_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(np.mean(batch_loss))
        update_loss = np.mean(epoch_loss)
        
        global_model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(valid_loader): 
            images, labels = images.to(device), labels.to(device)
            # Inference
            outputs = global_model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        print(f"\nTrain accuracy: {100*correct/total:.2f}%")
        print(f"\nTrain loss: {round(loss,4)}")
        global_model.eval()
        # g_idxs = np.arange(args.num_users)
        for i in range(args.num_users):
            # if epoch % 5 == 0:
            #     np.random.shuffle(g_idxs)
            #     user_groups[i] = np.concatenate((user_groups[i], user_groups[g_idxs[i]][: len(user_groups[g_idxs[i]])// 10]))
            #     users_label[i].extend(users_label[g_idxs[i]])
            label_acc , label_loss , test_acc, test_lo = local_test_inference(args,global_model,test_dataset,users_label[i],len(user_groups[i])*0.2)
            users_acc[i] , users_loss[i] = test_acc ,test_lo
        all_test_acc = np.mean(list(users_acc.values()))
        print(f' \nAvg Testing Stats for all clients ever trained:')
        print(f"|---- Test Accuracy: {100*all_test_acc:.2f}%") # org_test_accuracy
        print(f"|---- Test Loss: {round(np.mean(list(users_loss.values())),4)}")
        # Test inference after completion of training
        # test_acc, test_loss = test_inference(args, global_model, test_dataset)
        print(f' \n Results of training:') # old:args.epochs
        print(f"|---- Train Accuracy: {100*correct/total:.2f}%")
        print(f'|---- Test Accuracy: {100*all_test_acc:.2f}%')
        print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))