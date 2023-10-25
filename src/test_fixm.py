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
from utils_0820 import get_dataset, average_weights, exp_details
from torchsummary import summary
from collections import defaultdict

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
        train_dataset, test_dataset, user_groups , users_label= get_dataset(args)

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
        summary(global_model,input_size=train_dataset[0][0].shape) #7/25
        # copy weights
        global_weights = global_model.state_dict()

        # Training
        train_loss, train_accuracy = [], []
        val_acc_list, net_list = [], []
        cv_loss, cv_acc = [], []
        print_every = 1
        val_loss_pre, counter = 0, 0
        test_acc_ls , all_test_acc, dist = [0] , 0.0 , torch.tensor(0)# for first round
        users_acc , users_loss = {} , {}
        test_accuracy ,test_loss = [],[] #7/24 add
        epoch = 1 #7/24 add
        m = max(int(args.frac * args.num_users), 1) #8/1
        idxs_users = list(range(args.num_users)) #8/1
        selected_users = np.random.choice(idxs_users, m, replace=False)
        print('selected users:',selected_users)
        clients_ndata , users_dist = [] , {}
        # for epoch in tqdm(range(args.epochs)):
        # while args.test_acc > all_test_acc: # np.mean(test_acc_ls)
        while epoch <= args.epochs:
            local_weights, local_losses = [], []
            print(f'\n| Global Training Round : {epoch} |\n')
            g_idxs = np.random.choice(idxs_users, m, replace=False)
            global_model.train()
            for i,idx in enumerate(selected_users):
                if args.add:
                    if epoch % args.add_ep == 0:
                        user_groups[idx] = np.concatenate((user_groups[idx], user_groups[g_idxs[i]][: len(user_groups[g_idxs[i]])// 10]))
                        users_label[idx].extend(users_label[g_idxs[i]])
                if epoch == 1:
                    clients_ndata.append(len(user_groups[idx]))
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                        idxs=user_groups[idx], logger=logger)
                w, loss = local_model.update_weights(  # add kmeans cluster with local model weight(layers & dimensions)
                    model=copy.deepcopy(global_model), dist = dist, global_round=epoch) # send global model to client     
                local_weights.append(copy.deepcopy(w))

            # update global weights
            # global_weights = org_average_weights(local_weights)
            global_weights = average_weights(local_weights,clients_ndata)
            global_model.load_state_dict(global_weights)
            
            print(f"global['{list(global_weights.keys())[-1]}']:",list(global_weights.values())[-1])
            # loss_avg = sum(local_losses) / len(local_losses)
            # train_loss.append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            global_model.eval()
            label_acc_ls , label_loss_ls = [] , []
            test_acc_ls , test_lo_ls = [] , []
            # org_test_accuracy = []
            ndata = sum(clients_ndata)
            for i,idx in enumerate(selected_users):
                w = torch.cat([l.view(-1) for l in local_weights[i].values()])
                center = torch.cat([l.view(-1) for l in global_weights.values()])
                users_dist[idx] = round(torch.norm(w - center).item(),4)
                # print(f'round {epoch} distance between client{idx} and global model: {users_dist[idx]}')
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                        idxs=user_groups[idx], logger=logger)
                acc, loss = local_model.inference(model=global_model)
                list_acc.append(acc * clients_ndata[i])
                list_loss.append(loss * clients_ndata[i])
                label_acc , label_loss , test_acc, test_lo = local_test_inference(args,global_model,test_dataset,users_label[idx],clients_ndata[i]*0.2)
                users_acc[idx] , users_loss[idx] = test_acc ,test_lo
                test_acc_ls.append(test_acc)
                test_lo_ls.append(test_lo)
                label_acc_ls.append(label_acc)
                label_loss_ls.append(label_loss)
            train_accuracy.append(sum(list_acc)/ndata)
            train_loss.append(sum(list_loss)/ndata)
            
            # org_test_acc, org_test_lo = org_test_inference(args, global_model,test_dataset)
            
            # for i,idx in enumerate(selected_users):
            #     label_acc , label_loss , test_acc, test_lo = local_test_inference(args,global_model,test_dataset,users_label[idx],clients_ndata[i]*0.2)
            #     users_acc[idx] , users_loss[idx] = test_acc ,test_lo
            #     test_acc_ls.append(test_acc)
            #     test_lo_ls.append(test_lo)
            #     label_acc_ls.append(label_acc)
            #     label_loss_ls.append(label_loss)
            print(f' \nTraining & Testing Stats after {epoch} global rounds:') #7/24 the following
            print(f'|---- Train Accuracy: {100*train_accuracy[-1]:.2f}%')
            print(f'|---- Training Loss: {round(train_loss[-1],4)}')
            # print(f"|---- Test Accuracy: {100*org_test_acc:.2f}%")
            # print(f'|---- Test Loss: {round(org_test_lo,4)}')
            print(f"|---- Test Accuracy: {100*np.mean(test_acc_ls):.2f}%")
            print(f'|---- Test Loss: {round(np.mean(test_lo_ls),4)}')
            print(f"|---- Test Label Accuracy: {labels_eval(label_acc_ls)}")
            print(f"|---- Test Label Loss: {labels_eval(label_loss_ls)}")
            all_test_acc = np.mean(list(users_acc.values())) # test acc/loss for "all clients ever trained"
            test_loss.append(np.mean(list(users_loss.values())))
            test_accuracy.append(all_test_acc)
            # org_test_accuracy.append(org_test_acc)
            print(f' \nAvg Testing Stats for all clients ever trained after {epoch} global rounds:')
            print(f"|---- Test Accuracy: {100*all_test_acc:.2f}%") # org_test_accuracy
            print(f"|---- Test Loss: {round(test_loss[-1],4)}")
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
        # print('users distance:',users_dist)
        print(f' \n Results after {epoch} global rounds of training:') # old:args.epochs
        print(f"|---- Train Accuracy: {100*train_accuracy[-1]:.2f}%")
        print(f'|---- Test Accuracy: {100*all_test_acc:.2f}%')
        if len(np.where(np.array(test_accuracy) >= args.test_acc)[0]) == 0:
            first_reach = 'nan'
        else:
            first_reach = int(np.where(np.array(test_accuracy) >= args.test_acc)[0][0])
        f=open("../save/exp/fedavg_experiment.txt", "a+")
        f.write('\nFedAvg Experimental details:')
        f.write(f'\n    Model     : {args.model}')
        f.write(f'\n    Dataset     : {args.dataset}')
        f.write(f'\n    Optimizer : {args.optimizer}')
        f.write(f'\n    Learning rate : {args.lr}')
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
        f.write(f'\n    Number of users  : {args.num_users}')
        f.write(f'\n    Fraction of users  : {args.frac}')
        f.write(f'\n    Local Batch size   : {args.local_bs}')
        f.write(f'\n    Local Epochs       : {args.local_ep}\n')
        f.write(f' \nTraining & Testing Stats after {epoch} global rounds from Fedavg:')
        f.write(f'\n|---- Train Accuracy: {100*train_accuracy[-1]:.2f}%')
        f.write(f'\n|---- Training Loss: {round(train_loss[-1],4)}')
        f.write(f"\n|---- Test Accuracy: {100*np.mean(test_acc_ls):.2f}%")
        f.write(f'\n|---- Test Loss: {round(np.mean(test_lo_ls),4)}')
        f.write(f'\nAvg Testing Stats for all clients ever trained after {epoch} global rounds from Fedavg:')
        f.write(f"\n|---- Test Accuracy: {100*all_test_acc:.2f}%")
        f.write(f"\n|---- Test Loss: {round(test_loss[-1],4)}")
        f.write('\n Total Run Time: {0:0.4f}\n'.format(time.time()-start_time))
        f.close()
        # torch.save(global_model, './save_model/fedvg_global_model_{}_{}_{}_C[{}]_iid[{}]_unequal[{}]_one[{}]_E[{}]_B[{}].pt'.\
        #     format(args.dataset, args.model, epoch, args.frac, args.iid, args.unequal,
        #         args.one ,args.local_ep, args.local_bs))
        
        # Saving the objects train_loss and train_accuracy:
        # file_name = '../save/objects/fedavg_{}_{}_{}_C[{}]_iid[{}]_unequal[{}]_E[{}]_B[{}].pkl'.\
        #     format(args.dataset, args.model, epoch, args.frac, args.iid, args.unequal,
        #         args.local_ep, args.local_bs) #old :args.epochs
        # with open(file_name, 'wb') as f:
        #     pickle.dump([train_loss, train_accuracy], f)
        
        # json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
        #     open('../save/json/fedavg_test_result_{}_{}_round[{}]_C[{}]_iid[{}]_unequal[{}]_one[{}]_E[{}]_B[{}].json'.
        #     format(args.dataset, args.model, epoch, args.frac, args.iid,
        #              args.unequal, args.one ,args.local_ep, args.local_bs),'w'))
        if args.add:
            if args.iid:
                json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                open('../save/json/with_add/add_ep{}/{}/{}/{}_{}_round[{}]_C[{}]_E[{}]_B[{}]_iid.json'.format(args.add_ep,args.dataset, args.algo, args.algo, args.dataset, first_reach, args.frac, args.local_ep, args.local_bs),'w'))
                json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                open('../save/json/with_add/add_ep{}/{}/iid/{}_{}_round[{}]_C[{}]_E[{}]_B[{}]_iid.json'.format(args.add_ep,args.dataset, args.algo, args.dataset, first_reach, args.frac, args.local_ep, args.local_bs),'w'))
            else:
                if args.one:
                    json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                    open('../save/json/with_add/add_ep{}/{}/{}/{}_{}_round[{}]_C[{}]_E[{}]_B[{}]_one.json'.format(args.add_ep,args.dataset, args.algo, args.algo, args.dataset, first_reach, args.frac, args.local_ep, args.local_bs),'w'))
                    json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                    open('../save/json/with_add/add_ep{}/{}/non-iid/one/{}_{}_round[{}]_C[{}]_E[{}]_B[{}]_one.json'.format(args.add_ep,args.dataset, args.algo, args.dataset, first_reach, args.frac, args.local_ep, args.local_bs),'w'))
                elif args.unequal:
                    json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                    open('../save/json/with_add/add_ep{}/{}/{}/{}_{}_round[{}]_C[{}]_E[{}]_B[{}]_unequal.json'.format(args.add_ep,args.dataset, args.algo, args.algo, args.dataset, first_reach, args.frac, args.local_ep, args.local_bs),'w'))
                    json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                    open('../save/json/with_add/add_ep{}/{}/non-iid/unequal/{}_{}_round[{}]_C[{}]_E[{}]_B[{}]_unequal.json'.format(args.add_ep,args.dataset, args.algo, args.dataset, first_reach, args.frac, args.local_ep, args.local_bs),'w'))
                else:
                    json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                    open('../save/json/with_add/add_ep{}/{}/{}/{}_{}_round[{}]_C[{}]_E[{}]_B[{}]_two.json'.format(args.add_ep,args.dataset, args.algo, args.algo, args.dataset, first_reach, args.frac, args.local_ep, args.local_bs),'w'))
                    json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                    open('../save/json/with_add/add_ep{}/{}/non-iid/two/{}_{}_round[{}]_C[{}]_E[{}]_B[{}]_two.json'.format(args.add_ep,args.dataset, args.algo, args.dataset, first_reach, args.frac, args.local_ep, args.local_bs),'w'))
        else:
            if args.iid :
                json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                open('../save/json/{}/{}/{}_{}_round[{}]_C[{}]_E[{}]_B[{}]_iid.json'.format(args.dataset, args.algo, args.algo, args.dataset, first_reach, args.frac, args.local_ep, args.local_bs),'w'))
                json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                open('../save/json/{}/iid/{}_{}_round[{}]_C[{}]_E[{}]_B[{}]_iid.json'.format(args.dataset, args.algo, args.dataset, first_reach, args.frac, args.local_ep, args.local_bs),'w'))
            else:
                if args.one:
                    json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                    open('../save/json/{}/{}/{}_{}_round[{}]_C[{}]_E[{}]_B[{}]_one.json'.format(args.dataset, args.algo, args.algo, args.dataset, first_reach, args.frac, args.local_ep, args.local_bs),'w'))
                    json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                    open('../save/json/{}/non-iid/one/{}_{}_round[{}]_C[{}]_E[{}]_B[{}]_one.json'.format(args.dataset, args.algo, args.dataset, first_reach, args.frac, args.local_ep, args.local_bs),'w'))
                elif args.unequal:
                    json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                    open('../save/json/{}/{}/{}_{}_round[{}]_C[{}]_E[{}]_B[{}]_unequal.json'.format(args.dataset, args.algo, args.algo, args.dataset, first_reach, args.frac, args.local_ep, args.local_bs),'w'))
                    json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                    open('../save/json/{}/non-iid/unequal/{}_{}_round[{}]_C[{}]_E[{}]_B[{}]_unequal.json'.format(args.dataset, args.algo, args.dataset, first_reach, args.frac, args.local_ep, args.local_bs),'w'))
                else:
                    json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                    open('../save/json/{}/{}/{}_{}_round[{}]_C[{}]_E[{}]_B[{}]_two.json'.format(args.dataset, args.algo, args.algo, args.dataset, first_reach, args.frac, args.local_ep, args.local_bs),'w'))
                    json.dump({'test_accuracy':test_accuracy,'test_loss':test_loss},\
                    open('../save/json/{}/non-iid/two/{}_{}_round[{}]_C[{}]_E[{}]_B[{}]_two.json'.format(args.dataset, args.algo, args.dataset, first_reach, args.frac, args.local_ep, args.local_bs),'w'))
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))


    # # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # j = json.load(open('../save/json/fedavg_test_result_{}_{}_round[{}]_C[{}]_iid[{}]_unequal[{}]_one[{}]_E[{}]_B[{}].json'.
    #         format(args.dataset, args.model, epoch, args.frac, args.iid, 
    #                 args.unequal,args.one , args.local_ep, args.local_bs),'r'))
    # matplotlib.use('Agg')
    # # Plot Loss curve
    # plt.figure()
    # plt.title('Testing Loss vs Communication rounds')
    # plt.plot(range(len(j['test_loss'])), j['test_loss'], color='r',ls=':',label="fedavg loss:{}_{}_round[{}]_C[{}]_iid[{}]_unequal[{}]_one[{}]_E[{}]_B[{}]_test_loss".
    #             format(args.dataset, args.model, epoch, args.frac, args.iid, 
    #                   args.unequal, args.one , args.local_ep, args.local_bs))
    # plt.ylabel('Testing loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/img/fedavg_{}_{}_round[{}]_C[{}]_iid[{}]_unequal[{}]_one[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, epoch, args.frac, args.iid, 
    #                   args.unequal, args.one ,args.local_ep, args.local_bs)) # old: args.epochs
    
    # # Plot Testing Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Testing Accuracy vs Communication rounds')
    # plt.plot(range(len(j['test_accuracy'])), j['test_accuracy'], c='b',ls=':',label = 'fedavg acc:{}_{}_round[{}]_C[{}]_iid[{}]_unequal[{}]_one[{}]_E[{}]_B[{}]'.
    #             format(args.dataset, args.model, epoch, args.frac, args.iid,
    #                    args.unequal,args.one,args.local_ep, args.local_bs))
    # plt.legend(loc=0)
    # plt.axhline(y=args.test_acc, c='gray',ls='--')
    # plt.ylabel('Testing Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/img/fedavg_{}_{}_round[{}]_C[{}]_iid[{}]_unequal[{}]_one[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, epoch, args.frac, args.iid, 
    #                    args.unequal,args.one ,args.local_ep, args.local_bs)) # old: args.epochs
