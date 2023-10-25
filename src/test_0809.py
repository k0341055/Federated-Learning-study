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
from collections import OrderedDict

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
        global_model.train() # switch model to train status
        print(global_model)
        summary(global_model,input_size=train_dataset[0][0].shape) #7/25
        # copy weights
        # global_weights = global_model.state_dict()
        last_layer = list(global_model.state_dict().keys())[-1]

        # Training
        train_loss, train_accuracy = [], []
        val_acc_list, net_list = [], []
        cv_loss, cv_acc = [], []
        print_every = 1
        val_loss_pre, counter = 0, 0
        test_acc = 0.0 #7/24 add
        test_accuracy ,test_loss = [],[] #7/24 add
        test_acc_ls , ndata = [0] , 1 # for first round
        epoch = 1 #7/24 add
        dist = torch.tensor(0)
        m = max(int(args.frac * args.num_users), 1) #8/1
        idxs_users = list(range(args.num_users)) #8/1
        
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False) #select m clients from all users
        
        while args.test_acc > sum(test_acc_ls)/ndata:
            selected_users = np.random.choice(idxs_users, m, replace=False)
            print('selected users:',selected_users)
            list_acc_ls , list_loss_ls = [] , []# each cluster's train accuracy in a round
            test_acc_ls ,test_lo_ls = [] , [] # each cluster's test loss & accuracy in a round
            local_weights, local_losses = [], []
            print(f'\n| Global Training Round : {epoch} |\n')
            if epoch == 1:
                global_model.train()
                for i,idx in enumerate(selected_users):
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                            idxs=user_groups[idx], logger=logger)
                    w, loss = local_model.update_weights(
                        model = copy.deepcopy(global_model), dist = dist ,global_round=epoch) # send global model to client 
                    local_weights.append(copy.deepcopy(w))
                    local_losses.append(copy.deepcopy(loss))
                gm_ls = [copy.deepcopy(global_model) for i in range(args.num_clusters)]
                centers = 1
                # previous_centers = 1
            else:
                # previous_centers = copy.deepcopy(centers)
                for i in range(len(centers)):
                    gm_ls[i].train()
                for i,idx in enumerate(selected_users):
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                            idxs=user_groups[idx], logger=logger)
                    w, loss = local_model.update_weights(
                        model = copy.deepcopy(gm_ls[labels[i]]), dist = dist ,global_round=epoch) # send global model to client
                    # w = OrderedDict({key: w[key] + centers[labels[i]][key]*args.lam for key in w.keys()})
                    w = OrderedDict({key: w[key]*0.5*(2-args.lam) + centers[labels[i]][key]*0.5*args.lam for key in w.keys()})
                    local_weights.append(copy.deepcopy(w))
                    local_losses.append(copy.deepcopy(loss))

            # loss_avg = sum(local_losses) / len(local_losses)
            # train_loss.append(loss_avg)
            if args.km == 'm': 
                centers , labels , membership = kmeans(local_weights , selected_users , centers ,args )
            else:
                centers , labels , membership  = kmeans_flat(local_weights , selected_users , centers ,args )
            print(membership)
            for i in range(len(centers)): # for debug
                print(f"centers[{i}]['{last_layer}']:",centers[i][last_layer])
            ndata = 0
            for cluster in range(len(centers)):
                cluster_ndata = 0
                # update global weights
                global_weights = centers[cluster]
                gm_ls[cluster].load_state_dict(global_weights)
                # Calculate avg training accuracy over all users at every epoch
                list_acc, list_loss = [], []
                gm_ls[cluster].eval()
                for idx in membership[cluster]:
                    client_ndata = len(user_groups[idx]) # number of samples in client
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                            idxs=user_groups[idx], logger=logger)
                    acc, loss = local_model.inference(model=gm_ls[cluster])
                    list_acc.append(acc * client_ndata)
                    list_loss.append(loss * client_ndata)
                    cluster_ndata += client_ndata
                ndata += cluster_ndata
                cluster_acc = sum(list_acc)/cluster_ndata # train acc/loss in cluster
                cluster_loss = sum(list_loss)/cluster_ndata
                list_acc_ls.extend(list_acc)
                list_loss_ls.extend(list_loss) 
                label_acc , label_loss , test_acc, test_lo = test_inference(args, gm_ls[cluster], test_dataset,membership[cluster],users_label)
                test_acc_ls.append(test_acc * cluster_ndata)
                test_lo_ls.append(test_lo * cluster_ndata)
                print(f' \nTraining & Testing Stats after {epoch} global rounds for cluster{cluster}:')
                print(f"|---- Train Accuracy: {100*cluster_acc:.2f}%")
                print(f'|---- Train Loss: {cluster_loss}')
                print(f"|---- Test Accuracy: {100*test_acc:.2f}%")
                print(f'|---- Test Loss: {test_lo}')
                print(f"|---- Test Label Accuracy: {label_acc}")
                print(f"|---- Test Label Loss: {label_loss}")
            #以num_clusters中num_users數加權平均的各指標的作為每round的結果
            train_loss.append(sum(list_loss_ls)/ndata)
            train_accuracy.append(sum(list_acc_ls)/ndata)
            test_accuracy.append(sum(test_acc_ls)/ndata)
            test_loss.append(sum(test_lo_ls)/ndata)
            print(f' \nTraining & Testing Stats after {epoch} global rounds from multi-center:')
            print(f'|---- Training Loss: {train_loss[-1]}')
            print(f'|---- Train Accuracy: {100*train_accuracy[-1]:.2f}%')
            print(f'|---- Test Accuracy: {100*test_accuracy[-1]:.2f}%')
            print(f'|---- Test Loss: {test_loss[-1]}')
            # test_accuracy.append(test_acc)
            # test_loss.append(test_lo)
            
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
        print("|---- Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100*test_accuracy[-1]))
        ### save experiment result
        f=open("../save/exp/multi-center_experiment.txt", "a+")
        f.write('\nMulti-center FL Experimental details:')
        f.write(f'\n    Model     : {args.model}')
        f.write(f'\n    Optimizer : {args.optimizer}')
        f.write(f'\n    Learning rate : {args.lr}')
        f.write(f'\n    Lambda  : {args.lam}')
        f.write(f'\n    Global Rounds   : {epoch}\n')
        f.write(f'\n    Target test accuracy  : {args.test_acc}')
        f.write('    Federated parameters:')
        if args.iid:
            f.write('\n    IID')
        else:
            f.write('\n    Non-IID')
            if args.unequal:
                f.write('\n    Unbalanced')
            elif args.one:
                f.write('\n    One shard')
        f.write(f'\n    Number of clusters  : {args.num_clusters}')
        f.write(f'\n    Number of users  : {args.num_users}')
        f.write(f'\n    Fraction of users  : {args.frac}')
        f.write(f'\n    Local Batch size   : {args.local_bs}')
        f.write(f'\n    Local Epochs       : {args.local_ep}\n')
        f.write(f' \nTraining & Testing Stats after {epoch} global rounds from multi-center:')
        f.write(f'\n|---- Train Accuracy: {100*train_accuracy[-1]:.2f}%')
        f.write(f'\n|---- Training Loss: {train_loss[-1]}')
        f.write(f"\n|---- Test Accuracy: {100*test_accuracy[-1]:.2f}%")
        f.write(f"\n|---- Test Loss: {test_loss[-1]}")
        f.write('\n Total Run Time: {0:0.4f}\n'.format(time.time()-start_time))
        f.close()
        for i,gm in enumerate(gm_ls):
            torch.save(gm, './save_model/multi-center[{}]_model_{}_{}_{}_C[{}]_iid[{}]_unequal[{}]_one[{}]_E[{}]_B[{}].pt'.\
            format(i,args.dataset, args.model, epoch, args.frac, args.iid, args.unequal, args.one ,
                args.local_ep, args.local_bs))
        # Saving the objects train_loss and train_accuracy:
        # file_name = '../save/objects/multi-center_{}_{}_{}_C[{}]_Clusters[{}]_iid[{}]_unequal[{}]_E[{}]_B[{}].pkl'.\
        #     format(args.dataset, args.model, epoch, args.frac , args.num_clusters, args.iid, args.unequal,
        #         args.local_ep, args.local_bs) #old :args.epoch
        # with open(file_name, 'wb') as f:
        #     pickle.dump([train_loss, train_accuracy], f)
        json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
            open('../save/json/multi-center_test_result_{}_{}_round[{}]_C[{}]_Clusters[{}]_iid[{}]_unequal[{}]_one[{}]_E[{}]_B[{}].json'.
            format(args.dataset, args.model, epoch, args.frac,  args.num_clusters,
                    args.iid, args.unequal, args.one ,args.local_ep, args.local_bs),'w'))
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    j = json.load(open('../save/json/multi-center_test_result_{}_{}_round[{}]_C[{}]_Clusters[{}]_iid[{}]_unequal[{}]_one[{}]_E[{}]_B[{}].json'.
            format(args.dataset, args.model, epoch, args.frac, args.num_clusters,
                    args.iid, args.unequal, args.one ,args.local_ep, args.local_bs),'r'))
    matplotlib.use('Agg')
    # Plot Loss curve
    plt.figure()
    plt.title('Testing Loss vs Communication rounds')
    plt.plot(range(len(j['test_loss'])), j['test_loss'], color='r',ls=':',label = 'mc loss:{}_{}_round[{}]_C[{}]_Clusters[{}]_iid[{}]_unequal[{}]_one[{}]_E[{}]_B[{}]'.
                format(args.dataset, args.model, epoch, args.frac, args.num_clusters,
                       args.iid, args.unequal, args.one ,args.local_ep, args.local_bs))
    plt.ylabel('Testing loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/img/multi-center_{}_{}_round[{}]_C[{}]_Clusters[{}]_iid[{}]_unequal[{}]_one[{}]_E[{}]_B[{}]_loss.png'.
                format(args.dataset, args.model, epoch, args.frac, args.num_clusters,
                       args.iid, args.unequal, args.one ,args.local_ep, args.local_bs)) # old: args.epochs
    
    # Plot Testing Accuracy vs Communication rounds
    plt.figure()
    plt.title('Testing Accuracy vs Communication rounds')
    plt.plot(range(len(j['test_accuracy'])), j['test_accuracy'], c='b',ls=':',label = 'mc acc:{}_{}_round[{}]_C[{}]_Clusters[{}]_iid[{}]_unequal[{}]_one[{}]_E[{}]_B[{}]'.
                format(args.dataset, args.model, epoch, args.frac, args.num_clusters,
                       args.iid, args.unequal, args.one ,args.local_ep, args.local_bs))
    plt.legend(loc=0)
    plt.axhline(y=args.test_acc, c='gray',ls='--')
    plt.ylabel('Testing Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/img/multi-center_{}_{}_round[{}]_C[{}]_Clusters[{}]_iid[{}]_unequal[{}]_one[{}]_E[{}]_B[{}]_acc.png'.
                format(args.dataset, args.model, epoch, args.frac, args.num_clusters,
                       args.iid, args.unequal, args.one ,args.local_ep, args.local_bs)) # old: args.epochs