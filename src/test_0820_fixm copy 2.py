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
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, OptimizedCNNCifar, MLPv2
from utils_0820 import get_dataset, org_kmeans, kmeans_flat, dfs_cluster, dbscan_cluster, exp_details, average_weights # org_dbscan
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
        global_model
        # copy weights
        # global_weights = global_model.state_dict()
        last_layer = list(global_model.state_dict().keys())[-1]
        # print('global weights before train:',global_model.state_dict()[last_layer])
        # Training
        train_loss, train_accuracy = [], []
        val_acc_list, net_list = [], []
        cv_loss, cv_acc = [], []
        print_every = 1
        val_loss_pre, counter = 0, 0
        users_acc , users_loss , users_dist = {} , {} , {}
        global_models = {}
        # init_local_weights = {}
        test_accuracy ,test_loss = [],[] #7/24 add
        all_test_acc , dist , centers = 0.0, torch.tensor(0), 1 # for first round
        epoch = 1 #7/24 add
        m = max(int(args.frac * args.num_users), 1) #8/1
        idxs_users = list(range(args.num_users)) #8/1
        selected_users = np.random.choice(idxs_users, m, replace=False)
        print('selected users:',selected_users)
        # while args.test_acc > all_test_acc:
        while epoch <= args.epochs:
            local_weights, local_losses, clients_ndata = [] , [] , []
            print(f'\n| Global Training Round : {epoch} |\n')
            # selected_users = np.random.choice(idxs_users, m, replace=False)
            # print('selected users:',selected_users)
            g_idxs = np.random.choice(idxs_users, m, replace=False)
            for i,idx in enumerate(selected_users):
                if args.add:
                    if epoch % args.add_ep == 0:
                        user_groups[idx] = np.concatenate((user_groups[idx], user_groups[g_idxs[i]][: len(user_groups[g_idxs[i]])// 10]))
                        users_label[idx].extend(users_label[g_idxs[i]])
                # if idx not in clients_model:
                if epoch == 1:
                    global_model.train()
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                        idxs=user_groups[idx], logger=logger)
                    w, loss = local_model.update_weights(
                        model = copy.deepcopy(global_model), dist=dist , global_round=epoch) # copy.deepcopy(global_model)
                    local_weights.append(copy.deepcopy(w))
                    local_losses.append(copy.deepcopy(loss))
                else:
                    global_models[labels[i]].train()
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                            idxs=user_groups[idx], logger=logger)
                    # global_models[labels[i]].load_state_dict(init_local_weights[idx])
                    w, loss = local_model.update_weights(
                            model = copy.deepcopy(global_models[labels[i]]), dist = users_dist[idx] ,global_round=epoch) # dist # assign global model to client then train)
                    local_weights.append(copy.deepcopy(w))
                    local_losses.append(copy.deepcopy(loss))
                clients_ndata.append(len(user_groups[idx]))
            # global_model.load_state_dict(global_weights)
            # print('global weights after train:',global_model.state_dict()[last_layer])
            #     centers , labels , membership = kmeans(local_weights, clients_ndata, selected_users , centers ,args )
            if args.algo == 'fedsem':
                centers , labels , membership  = org_kmeans(local_weights, clients_ndata, selected_users , centers ,args )
            elif args.algo == 'fedsem_r':
                centers , labels , membership  = kmeans_flat(local_weights, clients_ndata, selected_users , centers ,args )
            # elif args.algo == 'org_dbscan':
            #     centers , labels , membership  = org_dbscan(local_weights, clients_ndata, selected_users, args )
            elif args.algo == 'feddbscan':
                centers , labels , membership  = dbscan_cluster(local_weights, clients_ndata, selected_users, args )
            elif args.algo == 'feddfs':
                centers , labels , membership  = dfs_cluster(local_weights, clients_ndata, selected_users, args )
            print(f'membership: {membership}')
            print(f'labels: {labels}')
    
            for i in range(len(centers)): # for debug
                print(f"centers[{i}]['{last_layer}']:",centers[i][last_layer])
            
            # Calculate avg training accuracy over all users at every epoch
            list_acc_ls , list_loss_ls = [] , []
            test_acc_ls , test_lo_ls = [] , []
            ndata = sum(clients_ndata)
            # cluster_ndata_ls = []
            if -1 in membership:
                print(f'Noise client: {membership[-1]}')
                # noise_ndata = 0
                list_acc, list_loss = [], []
                list_test_acc ,list_test_loss = [] , []
                label_acc_ls , label_loss_ls = [] , []
                noise_ids = [list(selected_users).index(idx) for idx in membership[-1]]
                # lws = np.array(local_weights)
                cnd = np.array(clients_ndata)
                # noise_lws = lws[noise_ids].tolist()
                noise_cnd = cnd[noise_ids].tolist()
                # noise_avg = average_weights(noise_lws,noise_cnd)
                noise_ndata = sum(noise_cnd)
                w_avg = average_weights(local_weights,clients_ndata)
                global_models[-1] = copy.deepcopy(global_model)
                global_models[-1].load_state_dict(w_avg)
                # cluster_ndata_ls = [sum(clients_ndata[list(selected_users).index(idx)] for idx in membership[cluster]) for cluster in range(len(centers))]
                for idx in membership[-1]:
                    i = list(selected_users).index(idx)
                    w = torch.cat([l.view(-1) for l in local_weights[i].values()])
                    # noise_avg_flat = torch.cat([l.view(-1) for l in noise_avg.values()])
                    avg_flat = torch.cat([l.view(-1) for l in w_avg.values()])
                    # users_dist[idx] = torch.norm(w - noise_avg_flat)
                    users_dist[idx] = torch.norm(w - avg_flat)
                    # noise_ndata += clients_ndata[i]
                    # clients_model[idx].load_state_dict(noise_avg)
                    # w = OrderedDict({key: local_weights[i][key]*0.5*(2-args.lam) + noise_avg[key]*0.5*args.lam for key in local_weights[i].keys()})
                    # clients_model[idx].load_state_dict(average_weights(centers,cluster_ndata_ls))
                    # init_w = OrderedDict({key: local_weights[i][key]*0.5*(2-args.lam) + w_avg[key]*0.5*args.lam for key in local_weights[i].keys()})
                    # init_local_weights[idx] = init_w
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                                idxs=user_groups[idx], logger=logger)
                    acc, loss = local_model.inference(model=global_models[-1])
                    list_acc.append(acc * clients_ndata[i]) # weighted by train dataset in user
                    list_loss.append(loss * clients_ndata[i])
                    label_acc , label_loss , test_acc, test_lo = local_test_inference(
                        args, global_models[-1] ,test_dataset,users_label[idx],clients_ndata[i]*0.2)
                    # print(f'cluster {cluster} user {idx} labels:',users_label[idx])
                    users_acc[idx] = test_acc
                    users_loss[idx] = test_lo
                    list_test_acc.append(test_acc)
                    list_test_loss.append(test_lo)
                    label_acc_ls.append(label_acc)
                    label_loss_ls.append(label_loss)
                list_acc_ls.extend(list_acc)
                list_loss_ls.extend(list_loss)
                test_acc_ls.extend(list_test_acc)
                test_lo_ls.extend(list_test_loss)
                print(f' \nTraining & Testing Stats after {epoch} global rounds for noise:')
                print(f"|---- Train Accuracy: {100*round(sum(list_acc)/noise_ndata,4):.2f}%")
                print(f'|---- Train Loss: {round(sum(list_loss)/noise_ndata,4)}')
                print(f"|---- Test Accuracy: {100*round(np.mean(list_test_acc),4):.2f}%")
                print(f'|---- Test Loss: {round(np.mean(list_test_loss),4)}')
                print(f"|---- Test Label Accuracy: {labels_eval(label_acc_ls)}")
                print(f"|---- Test Label Loss: {labels_eval(label_loss_ls)}")
            for cluster in range(len(centers)):
                global_models[cluster] = copy.deepcopy(global_model)
                global_models[cluster].load_state_dict(centers[cluster])
                cluster_ndata = 0
                list_acc, list_loss = [], []
                list_test_acc ,list_test_loss = [] , []
                label_acc_ls , label_loss_ls = [] , []
                
                for idx in membership[cluster]:
                    i = list(selected_users).index(idx)
                    w = torch.cat([l.view(-1) for l in local_weights[i].values()])
                    center = torch.cat([l.view(-1) for l in centers[cluster].values()])
                    users_dist[idx] = torch.norm(w - center)
                    # if args.algo == 'fedsem':
                    #     init_w = centers[cluster]
                    # else:
                    #     init_w = OrderedDict({key: local_weights[i][key]*0.5*(2-args.lam) + centers[cluster][key]*0.5*args.lam for key in local_weights[i].keys()})
                    # init_local_weights[idx] = init_w
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                            idxs=user_groups[idx], logger=logger)
                    acc, loss = local_model.inference(model=global_models[cluster])
                    list_acc.append(acc * clients_ndata[i]) # weighted by train dataset in user
                    list_loss.append(loss * clients_ndata[i])
                    cluster_ndata += clients_ndata[i]
                    label_acc , label_loss , test_acc, test_lo = local_test_inference(
                        args, global_models[cluster] ,test_dataset,users_label[idx],clients_ndata[i]*0.2)
                    # print(f'cluster {cluster} user {idx} labels:',users_label[idx])
                    users_acc[idx] = test_acc
                    users_loss[idx] = test_lo
                    list_test_acc.append(test_acc)
                    list_test_loss.append(test_lo)
                    label_acc_ls.append(label_acc)
                    label_loss_ls.append(label_loss)
                # cluster_ndata_ls.append(cluster_ndata)
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
            print(f'|---- Train Accuracy: {100*train_accuracy[-1]:.2f}%')
            print(f'|---- Training Loss: {round(train_loss[-1],4)}')
            print(f'|---- Test Accuracy: {100*np.mean(test_acc_ls):.2f}%')
            print(f'|---- Test Loss: {round(np.mean(test_lo_ls),4)}')
            print(f' \nAvg Testing Stats for all clients ever trained after {epoch} global rounds:')
            print(f'|---- Test Accuracy: {100*all_test_acc:.2f}%')
            print(f'|---- Test Loss: {round(test_loss[-1],4)}')
            print('users_acc:',users_acc)
    
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
        
        if len(np.where(np.array(test_accuracy) >= args.test_acc)[0]) == 0:
            first_reach = 'nan'
        else:
            first_reach = int(np.where(np.array(test_accuracy) >= args.test_acc)[0][0])
        ### save experiment result
        f=open("../save/exp/multi-center_experiment.txt", "a+")
        f.write('\nMulti-center FL Experimental details:')
        f.write(f'\n    Model     : {args.model}')
        f.write(f'\n    Dataset     : {args.dataset}')
        if args.algo == 'fedavg':
            f.write(f'\n    Aggregation Algorithm : FedAVG')
        elif args.algo == 'fedsem':
            f.write(f'\n    Aggregation Algorithm : FedSEM')
        elif args.algo == 'fedsem_r':
            f.write(f'\n    Aggregation Algorithm : FedSEM_R')
        elif args.algo == 'feddbscan':
            f.write(f'\n    Aggregation Algorithm : FedDBSCAN')
        elif args.algo == 'feddfs':
            f.write(f'\n    Aggregation Algorithm : FedDFS')
        f.write(f'\n    Optimizer : {args.optimizer}')
        f.write(f'\n    Learning rate : {args.lr}')
        f.write(f'\n    Lambda  : {args.lam}')
        f.write(f'\n    Global Rounds   : {first_reach}\n')
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
        f.write(f'\n    Number of clusters  : {len(membership.keys())}')
        f.write(f'\n    Number of users  : {args.num_users}')
        f.write(f'\n    Fraction of users  : {args.frac}')
        f.write(f'\n    Local Batch size   : {args.local_bs}')
        f.write(f'\n    Local Epochs       : {args.local_ep}\n')
        f.write(f' \nTraining & Testing Stats after {epoch} global rounds from multi-center:')
        f.write(f'\n|---- Train Accuracy: {100*train_accuracy[-1]:.2f}%')
        f.write(f'\n|---- Training Loss: {round(train_loss[-1],4)}')
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
        if args.iid :
            json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
            open('../save/json/with_init/{}/{}/{}_{}_round[{}]_C[{}]_Clusters[{}]_E[{}]_B[{}]_iid.json'.format(args.dataset, args.algo, args.algo, args.dataset, first_reach, args.frac,  len(membership.keys()), args.local_ep, args.local_bs),'w'))
            json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
            open('../save/json/with_init/{}/iid/{}_{}_round[{}]_C[{}]_Clusters[{}]_E[{}]_B[{}]_iid.json'.format(args.dataset, args.algo, args.dataset, first_reach, args.frac,  len(membership.keys()), args.local_ep, args.local_bs),'w'))
        else:
            if args.one:
                json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                open('../save/json/with_init/{}/{}/{}_{}_round[{}]_C[{}]_Clusters[{}]_E[{}]_B[{}]_one.json'.format(args.dataset, args.algo, args.algo, args.dataset, first_reach, args.frac,  len(membership.keys()), args.local_ep, args.local_bs),'w'))
                json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                open('../save/json/with_init/{}/non-iid/one/{}_{}_round[{}]_C[{}]_Clusters[{}]_E[{}]_B[{}]_one.json'.format(args.dataset, args.algo, args.dataset, first_reach, args.frac,  len(membership.keys()), args.local_ep, args.local_bs),'w'))
            elif args.unequal:
                json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                open('../save/json/with_init/{}/{}/{}_{}_round[{}]_C[{}]_Clusters[{}]_E[{}]_B[{}]_unequal.json'.format(args.dataset, args.algo, args.algo, args.dataset, first_reach, args.frac,  len(membership.keys()), args.local_ep, args.local_bs),'w'))
                json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                open('../save/json/with_init/{}/non-iid/unequal/{}_{}_round[{}]_C[{}]_Clusters[{}]_E[{}]_B[{}]_unequal.json'.format(args.dataset, args.algo, args.dataset, first_reach, args.frac,  len(membership.keys()), args.local_ep, args.local_bs),'w'))
            else:
                json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                open('../save/json/with_init/{}/{}/{}_{}_round[{}]_C[{}]_Clusters[{}]_E[{}]_B[{}]_two.json'.format(args.dataset, args.algo, args.algo, args.dataset, first_reach, args.frac,  len(membership.keys()), args.local_ep, args.local_bs),'w'))
                json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                open('../save/json/with_init/{}/non-iid/two/{}_{}_round[{}]_C[{}]_Clusters[{}]_E[{}]_B[{}]_two.json'.format(args.dataset, args.algo, args.dataset, first_reach, args.frac,  len(membership.keys()), args.local_ep, args.local_bs),'w'))
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # j = json.load(open('../save/json/{}/{}/multi-center_test_result_{}_round[{}]_C[{}]_Clusters[{}]_iid[{}]_unequal[{}]_one[{}]_E[{}]_B[{}].json'.
    #         format(args.dataset, args.algo, args.model, epoch, args.frac,  args.num_clusters,
    #                 args.iid, args.unequal, args.one ,args.local_ep, args.local_bs),'r'))
    # matplotlib.use('Agg')
    # # Plot Loss curve
    # plt.figure()
    # plt.title('Testing Loss vs Communication rounds')
    # plt.plot(range(len(j['test_loss'])), j['test_loss'], color='r',ls=':',label = 'mc loss:{}_{}_round[{}]_C[{}]_Clusters[{}]_iid[{}]_unequal[{}]_one[{}]_E[{}]_B[{}]'.
    #             format(args.dataset, args.model, epoch, args.frac, args.num_clusters,
    #                    args.iid, args.unequal, args.one ,args.local_ep, args.local_bs))
    # plt.ylabel('Testing loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/img/multi-center_{}_{}_round[{}]_C[{}]_Clusters[{}]_iid[{}]_unequal[{}]_one[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, epoch, args.frac, args.num_clusters,
    #                    args.iid, args.unequal, args.one ,args.local_ep, args.local_bs)) # old: args.epochs
    
    # # Plot Testing Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Testing Accuracy vs Communication rounds')
    # plt.plot(range(len(j['test_accuracy'])), j['test_accuracy'], c='b',ls=':',label = 'mc acc:{}_{}_round[{}]_C[{}]_Clusters[{}]_iid[{}]_unequal[{}]_one[{}]_E[{}]_B[{}]'.
    #             format(args.dataset, args.model, epoch, args.frac, args.num_clusters,
    #                    args.iid, args.unequal, args.one ,args.local_ep, args.local_bs))
    # plt.legend(loc=0)
    # plt.axhline(y=args.test_acc, c='gray',ls='--')
    # plt.ylabel('Testing Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/img/multi-center_{}_{}_round[{}]_C[{}]_Clusters[{}]_iid[{}]_unequal[{}]_one[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, epoch, args.frac, args.num_clusters,
    #                    args.iid, args.unequal, args.one ,args.local_ep, args.local_bs)) # old: args.epochs