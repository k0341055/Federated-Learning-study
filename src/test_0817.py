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
from update import LocalUpdate, local_test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar,MLPv2
from utils import get_dataset, kmeans_flat, exp_details, average_weights
from torchsummary import summary
from collections import OrderedDict, defaultdict

def labels_eval(label_dict):
    merge_dict = defaultdict(list)
    for d in label_dict:
        for key, value in d.items():
            merge_dict[key].append(value)
    outdict = {key: round(np.mean(values),4) for key, values in merge_dict.items()}
    return outdict


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
        # global_model.train() # switch model to train status
        # init_gm = copy.deepcopy(global_model)
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
        clients_model, users_acc, users_loss = {} , {} , {} # users_dist = {}
        test_accuracy ,test_loss = [],[] #7/24 add
        all_test_acc , centers , dist = 0.0 , 1 , torch.tensor(0)# for first round # dist = torch.tensor(1)
        epoch = 1 #7/24 add
        m = max(int(args.frac * args.num_users), 1) #8/1
        idxs_users = list(range(args.num_users)) #8/1
        while args.test_acc > all_test_acc:
            print(f'\n| Global Training Round : {epoch} |\n')
            selected_users = np.random.choice(idxs_users, m, replace=False)
            print('selected users:',selected_users)
            clients_ndata, local_weights, local_losses = [], [], []
            for idx in selected_users:
                clients_ndata.append(len(user_groups[idx]))
                if idx not in clients_model: # initialize local model with global model
                    # gm = copy.deepcopy(init_gm)
                    # gm.train()
                    global_model.train()
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                        idxs=user_groups[idx], logger=logger)
                    w, loss = local_model.update_weights(
                        model = copy.deepcopy(global_model), dist = dist, global_round=epoch) # gm
                    local_weights.append(copy.deepcopy(w))
                    local_losses.append(copy.deepcopy(loss))
                    clients_model[idx] = copy.deepcopy(global_model) # gm
                else:
                    clients_model[idx].train()
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                            idxs=user_groups[idx], logger=logger)
                    w, loss = local_model.update_weights(
                        model = clients_model[idx], dist = dist, global_round=epoch) # assign center^k to client then train)
                    # update local model weights: W~k*(𝜆/m)+Wi where W~k:multi-center’s model, 𝜆:hyper parameter, m:number of clients, Wi:local model(have trained)
                    # w = OrderedDict({key: w[key]*0.5*(2-args.lam) + centers[labels[i]][key]*0.5*args.lam for key in w.keys()})
                    # w = OrderedDict({key: w[key] + centers[labels[i]][key]*args.lam for key in w.keys()})
                    local_weights.append(copy.deepcopy(w))
                    local_losses.append(copy.deepcopy(loss))
            global_weights = average_weights(local_weights,clients_ndata)
            global_model.load_state_dict(global_weights)
            #     centers , labels , membership = kmeans(local_weights, clients_ndata, selected_users , centers ,args )
            centers , labels , membership  = kmeans_flat(local_weights, clients_ndata, selected_users , centers ,args )
            print(membership)
            for i in range(len(centers)): # for debug
                print(f"centers[{i}]['{last_layer}']:",centers[i][last_layer])
            
            # Calculate avg training accuracy over all users at every epoch
            list_acc_ls , list_loss_ls = [] , []# each cluster's train accuracy in a round
            test_acc_ls , test_lo_ls = [] , []
            ndata = sum(clients_ndata)
            cluster_ndata_ls = []
            for cluster in range(len(centers)):
                cluster_ndata = 0
                list_acc, list_loss = [], []
                list_test_acc ,list_test_loss = [] , []
                label_acc_ls , label_loss_ls = [] , []
                for idx in membership[cluster]:
                    i = list(selected_users).index(idx)
                    # w = torch.cat([l.view(-1) for l in local_weights[i].values()])
                    # center = torch.cat([l.view(-1) for l in centers[labels[i]].values()])
                    # users_dist[idx] = torch.norm(w - center)
                    w = OrderedDict({key: local_weights[i][key]*0.5*(2-args.lam) + centers[labels[i]][key]*0.5*args.lam for key in local_weights[i].keys()})
                    clients_model[idx].load_state_dict(w) #initialize local model with center^k # clients_model[idx].load_state_dict(local_weights[i])
                    clients_model[idx].eval()
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                            idxs=user_groups[idx], logger=logger)
                    acc, loss = local_model.inference(model=clients_model[idx])
                    list_acc.append(acc * clients_ndata[i]) # weighted by train dataset in user
                    list_loss.append(loss * clients_ndata[i])
                    cluster_ndata += clients_ndata[i]
                    label_acc , label_loss , test_acc, test_lo = local_test_inference(args,clients_model[idx], test_dataset,users_label[idx],clients_ndata[i]*0.2)
                    users_acc[idx] = test_acc
                    users_loss[idx] = test_lo
                    list_test_acc.append(test_acc)
                    list_test_loss.append(test_lo)
                    label_acc_ls.append(label_acc)
                    label_loss_ls.append(label_loss)
                cluster_ndata_ls.append(cluster_ndata)
                cluster_acc = sum(list_acc)/cluster_ndata # train acc/loss in cluster
                cluster_loss = sum(list_loss)/cluster_ndata
                list_acc_ls.extend(list_acc)
                list_loss_ls.extend(list_loss)
                cluster_test_acc = np.mean(list_test_acc) # test acc/loss in cluster
                cluster_test_loss = np.mean(list_test_loss)
                test_acc_ls.extend(list_test_acc)
                test_lo_ls.extend(list_test_loss)
                print(f' \nTraining & Testing Stats after {epoch} global rounds for cluster{cluster}:')
                print(f"|---- Train Accuracy: {100*cluster_acc:.2f}%")
                print(f'|---- Train Loss: {round(cluster_loss,4)}')
                print(f"|---- Test Accuracy: {100*cluster_test_acc:.2f}%")
                print(f'|---- Test Loss: {round(cluster_test_loss,4)}')
                print(f"|---- Test Label Accuracy: {labels_eval(label_acc_ls)}")
                print(f"|---- Test Label Loss: {labels_eval(label_loss_ls)}")
            # global_model.train()
            # global_weights = average_weights(centers,cluster_ndata_ls)
            # global_model.load_state_dict(global_weights)
            #以num_clusters中各user的train sample數加權平均的train acc/loss作為每round的train結果
            all_test_acc = np.mean(list(users_acc.values()))
            train_loss.append(sum(list_loss_ls)/ndata)
            train_accuracy.append(sum(list_acc_ls)/ndata)
            test_accuracy.append(all_test_acc) # test acc/loss for "all clients ever trained"
            test_loss.append(np.mean(list(users_loss.values())))
            print(f' \nTraining & Testing Stats after {epoch} global rounds from multi-center:')
            print(f'|---- Training Loss: {round(train_loss[-1],4)}')
            print(f'|---- Train Accuracy: {100*train_accuracy[-1]:.2f}%')
            print(f'|---- Test Accuracy: {100*np.mean(test_acc_ls):.2f}%')
            print(f'|---- Test Loss: {round(np.mean(test_lo_ls),4)}')
            print(f' \nAvg Testing Stats for all clients ever trained after {epoch} global rounds:')
            print(f'|---- Test Accuracy: {100*all_test_acc:.2f}%')
            print(f'|---- Test Loss: {round(test_loss[-1],4)}')
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
        print("|---- Test Accuracy: {:.2f}%".format(100*all_test_acc))
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
        f.write(f'\n|---- Test Accuracy: {100*np.mean(test_acc_ls):.2f}%')
        f.write(f'\n|---- Test Loss: {np.mean(test_lo_ls):.4f}')
        f.write(f'\nAvg Testing Stats for all clients ever trained after {epoch} global rounds from multi-center:')
        f.write(f"\n|---- Test Accuracy: {100*all_test_acc:.2f}%")
        f.write(f"\n|---- Test Loss: {round(test_loss[-1],4)}")
        f.write('\n Total Run Time: {0:0.4f}\n'.format(time.time()-start_time))
        f.close()
        # for idx,cm in clients_model.items():
        #     torch.save(cm, './save_model/multi-center/mc_client{}_model_{}_{}_{}_C[{}]_iid[{}]_unequal[{}]_one[{}]_E[{}]_B[{}].pt'.\
        #     format(idx,args.dataset, args.model, epoch, args.frac, args.iid, args.unequal, args.one ,
        #         args.local_ep, args.local_bs))
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