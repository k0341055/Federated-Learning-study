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

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar,MLPv2
from utils import get_dataset, average_weights, kmeans, kmeans_flat, exp_details #cluster_aggregate
from torchsummary import summary

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)
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
        train_dataset, test_dataset, user_groups = get_dataset(args)

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
        global_model.train() # switch model to train status
        print(global_model)
        summary(global_model,input_size=train_dataset[0][0].shape) #7/25
        # copy weights
        # global_weights = global_model.state_dict()

        # Training
        train_loss, train_accuracy = [], []
        val_acc_list, net_list = [], []
        cv_loss, cv_acc = [], []
        print_every = 1
        val_loss_pre, counter = 0, 0
        test_acc = 0.0 #7/24 add
        test_accuracy ,test_loss = [],[] #7/24 add
        epoch = 1 #7/24 add
        m = max(int(args.frac * args.num_users), 1) #8/1
        idxs_users = list(range(args.num_users)) #8/1
        local_weights, local_losses = [], []
        print(f'\n| Global Training Round : {epoch} |\n')
        global_model.train()
        selected_users = np.random.choice(idxs_users, m, replace=False)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False) #select m clients from all users
        print('selected users:',selected_users)

        for idx in selected_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch) # send global model to client

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            # print('user',idx,'\'s','local weight:',w)
        ''' clustering clients before avgerage client's model parameters
            multi-center or HC or density peaks cluster
        '''
        if args.km == 'm': 
            centers , labels , membership = kmeans(local_weights, args.num_clusters, args.k_vote , args.max_iters)
        else:
            centers , labels , membership  = kmeans_flat(local_weights, args.num_clusters ,  args.max_iters)
    
        # for epoch in tqdm(range(args.epochs)):
        for cluster in range(args.num_clusters):
            while args.test_acc >= test_acc: # 7/24 add
                # update global weights
                global_weights = centers[cluster]
                global_model.load_state_dict(global_weights)

                loss_avg = sum(local_losses) / len(local_losses)
                train_loss.append(loss_avg)

                # Calculate avg training accuracy over all users at every epoch
                list_acc, list_loss = [], []
                global_model.eval()
                for c in range(args.num_users):
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                            idxs=user_groups[idx], logger=logger)
                    acc, loss = local_model.inference(model=global_model)
                    list_acc.append(acc)
                    list_loss.append(loss)
                train_accuracy.append(sum(list_acc)/len(list_acc))

                print(f' \nAvg Training Stats after {epoch} global rounds for cluster{cluster}:') #7/24 the following
                print(f'|---- Training Loss: {np.mean(np.array(train_loss))}')
                print('|---- Train Accuracy: {:.2f}%'.format(100*train_accuracy[-1]))
                test_acc, test_lo = test_inference(args, global_model, test_dataset) #old:test_loss
                print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
                print(f'|---- Test Loss: {test_lo}')
                test_accuracy.append(test_acc)
                test_loss.append(test_lo)
                # print global training loss after every 'i' rounds
                # if (epoch+1) % print_every == 0:
                #     print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                #     print(f'Training Loss : {np.mean(np.array(train_loss))}')
                #     print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
                # if (epoch+1) % 10 == 0:
                #     test_acc, test_lo = test_inference(args, global_model, test_dataset) #old:test_loss
                #     print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
                epoch += 1
            # Test inference after completion of training
            # test_acc, test_loss = test_inference(args, global_model, test_dataset)
            epoch -= 1
            print(f' \n Results after {epoch} global rounds of training:') # old:args.epochs
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

