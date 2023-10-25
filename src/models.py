#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F


# class MLP2NNMnist(nn.Module):
#     def __init__(self):
#         super(MLP2NNMnist, self).__init__()
#         self.fc1 = nn.Linear(28*28, 200)
#         self.fc2 = nn.Linear(200, 200)
#         self.fc3 = nn.Linear(200, 10)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = x.view(-1, 28*28)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         return nn.LogSoftmax(dim=1)(x)

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)
    
class MLPv2(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLPv2, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)

# class CNNMnist(nn.Module):
#     def __init__(self, args):
#         super(CNNMnist, self).__init__()
#         self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, args.num_classes)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# class CNNMnist(nn.Module):
#     def __init__(self, args):
#         super(CNNMnist, self).__init__()        
#         self.conv1 = nn.Conv2d(args.num_channels, 32, kernel_size=5)   # 第一個卷積層：32個輸出通道，卷積核大小為5x5
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5)        # 第二個卷積層：64個輸出通道，卷積核大小為5x5
#         # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) # 最大池化層：2x2最大池化
#         self.fc1 = nn.Linear(64*4*4, 512)           # 第一個全連接層：輸入維度為64*4*4（來自第二個卷積層的輸出），輸出維度為512
#         self.fc2 = nn.Linear(512, args.num_classes)  # 最後的輸出層：輸入維度為512，輸出維度為10（對應於MNIST數據集的10個類別）

#     def forward(self, x):        
#         x = F.max_pool2d(self.conv1(x),2) # 第一個卷積層 -> 最大池化
#         x = F.max_pool2d(self.conv2(x),2) # 第二個卷積層 -> 最大池化     
#         x = x.view(-1,  x.shape[1]*x.shape[2]*x.shape[3])     # 將特徵展平以進入全連接層 64*4*4
#         x = F.relu(self.fc1(x))    # 第一個全連接層 -> ReLU
#         x = self.fc2(x)    # 最後的全連接層（輸出層）
#         return F.log_softmax(x, dim=1)  # 使用 log_softmax 作為輸出

# class CNNMnist(nn.Module):
#     def __init__(self, args):
#         super(CNNMnist, self).__init__()        
#         self.conv1 = nn.Conv2d(args.num_channels, 32, kernel_size=5)   # 第一個卷積層：32個輸出通道，卷積核大小為5x5
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5)        # 第二個卷積層：64個輸出通道，卷積核大小為5x5
#         # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) # 最大池化層：2x2最大池化
#         self.fc1 = nn.Linear(64*4*4, 512)           # 第一個全連接層：輸入維度為64*4*4（來自第二個卷積層的輸出），輸出維度為512
#         self.fc2 = nn.Linear(512, 2078)              # 第二個全連接層：輸入維度為1024（來自第二個卷積層的輸出），輸出維度為2078
#         self.fc3 = nn.Linear(2078, args.num_classes)  # 最後的輸出層：輸入維度為2078，輸出維度為10（對應於MNIST數據集的10個類別）

#     def forward(self, x):        
#         x = F.max_pool2d(self.conv1(x),2) # 第一個卷積層 -> 最大池化
#         x = F.max_pool2d(self.conv2(x),2) # 第二個卷積層 -> 最大池化     
#         x = x.view(-1,  x.shape[1]*x.shape[2]*x.shape[3])     # 將特徵展平以進入全連接層 64*4*4
#         x = F.relu(self.fc1(x))    # 第一個全連接層 -> ReLU
#         x = F.relu(self.fc2(x))    # 第二個全連接層 -> ReLU
#         x = F.dropout(x, training=self.training)
#         x = self.fc3(x)    # 最後的全連接層（輸出層）
#         return F.log_softmax(x, dim=1)  # 使用 log_softmax 作為輸出

class CNNMnist(nn.Module): #total params: 1663370
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 32, kernel_size=5 , padding = 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding = 2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, args.num_classes)
        self.device = 'cuda:0' if args.gpu else 'cpu'

    def forward(self, x):
        x = x.to(self.device)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x))), 2)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# class CNNMnist(nn.Module): #total params: 1635210
#     def __init__(self, args):
#         super(CNNMnist, self).__init__()
#         self.conv1 = nn.Conv2d(args.num_channels, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(64*7*7, 512)
#         self.fc2 = nn.Linear(512, args.num_classes)

#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), 2)
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training = True)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# class CNNFashion_Mnist(nn.Module):
#     def __init__(self, args):
#         super(CNNFashion_Mnist, self).__init__()
#         self.conv1 = nn.Conv2d(args.num_channels, 32, kernel_size=5 , padding=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(64*7*7, 512)
#         self.fc2 = nn.Linear(512, args.num_classes)
#         self.device = 'cuda:0' if args.gpu else 'cpu'

#     def forward(self, x):
#         x = x.to(self.device)
#         x = F.max_pool2d(F.relu(self.conv1(x)), 2)
#         x = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x))), 2)
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)
        self.device = 'cuda:0' if args.gpu else 'cpu'
    def forward(self, x):
        x = x.to(self.device)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class OptimizedCNNCifar(nn.Module):
    def __init__(self, args):
        super(OptimizedCNNCifar, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Adjusted the input size due to the change in convolutional layers
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, args.num_classes)
        
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)
        self.device = 'cuda:0' if args.gpu else 'cpu'

    def forward(self, x):
        x = x.to(self.device)
        
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 4 * 4)  # Adjusted the view size due to the change in convolutional layers
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)


# class CNNCifar(nn.Module):
#     def __init__(self, args):
#         super(CNNCifar, self).__init__()
        
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
#         self.conv_drop = nn.Dropout2d()

#         # Fully connected layers
#         self.fc1 = nn.Linear(128 * 3 * 3, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, args.num_classes)
#         # self.dropout = nn.Dropout(0.5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.device = 'cuda:0' if args.gpu else 'cpu'

#     def forward(self, x):
#         x = x.to(self.device)
        
#         # Convolutional layers
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv_drop(self.conv2(x)))) # x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv_drop(self.conv3(x)))) # x = self.pool(F.relu(self.conv3(x)))
        
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]) # 128 * 3 * 3
        
#         # Fully connected layers
#         x = F.relu(self.fc1(x))
#         # x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         # x = self.dropout(x)
#         x = self.fc3(x)
        
#         return F.log_softmax(x, dim=1)

# class ModifiedCNNCifar(nn.Module):
#     def __init__(self, args):
#         super(ModifiedCNNCifar, self).__init__()
        
#         # Two convolutional layers
#         self.conv1 = nn.Conv2d(3, 32, 5, padding=2)  # Change kernel size to 5 and padding to 2
#         self.conv2 = nn.Conv2d(32, 64, 5, padding=2)  # Change kernel size to 5 and padding to 2
        
#         # Two fully connected layers and a linear transformation layer
#         self.fc1 = nn.Linear(64 * 6 * 6, 500)  # Adjusted to get close to 1M parameters
#         self.fc2 = nn.Linear(500, 250)  # Adjusted
#         self.fc3 = nn.Linear(250, args.num_classes)
        
#         self.device = 'cuda:0' if args.gpu else 'cpu'
#         self.pool = nn.MaxPool2d(2, 2)

#     def forward(self, x):
#         x = x.to(self.device)
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 6 * 6)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
        
#         return F.log_softmax(x, dim=1)

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, args.num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv_drop = nn.Dropout2d()
        self.device = 'cuda:0' if args.gpu else 'cpu'

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) # self.pool(F.relu(self.conv_drop(self.conv2(x))))
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
# class CNNCifar(nn.Module): #testing
#     def __init__(self, args):
#         super(CNNCifar, self).__init__()
        
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
#         self.fc1 = nn.Linear(64 * 6 * 6, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.fc3 = nn.Linear(128, args.num_classes)
#         self.pool = nn.MaxPool2d(2,2)
#         self.conv_drop = nn.Dropout2d()
#         self.device = 'cuda:0' if args.gpu else 'cpu'

#     def forward(self, x):
#         x = x.to(self.device)
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv_drop(self.conv2(x))))
#         x = x.view(-1, 64 * 6 * 6)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)

# class CNNCifar(nn.Module):  (with conv_drop [previous ver])
#     def __init__(self, args):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.conv_drop = nn.Dropout2d() # new
#         self.fc1 = nn.Linear(64 * 6 * 6, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, args.num_classes)
#         self.device = 'cuda:0' if args.gpu else 'cpu'
#         self.pool = nn.MaxPool2d(2, 2)

#     def forward(self, x):
#         x = x.to(self.device)
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv_drop(self.conv2(x)))) # x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])  # 64 * 6 * 6
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)

# class CNNCifar(nn.Module): # my model
#     def __init__(self, args):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(64 * 8 * 8, 512)
#         self.fc2 = nn.Linear(512, args.num_classes)
#         self.device = 'cuda:0' if args.gpu else 'cpu'

#     def forward(self, x):
#         x = x.to(self.device)
#         x = F.max_pool2d(F.relu(self.conv1(x)), 2)
#         x = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x))), 2)
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
    
# class CNNCifar(nn.Module): # org_ver batch_size 8
#     def __init__(self, args):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, args.num_classes)
#         self.device = 'cuda:0' if args.gpu else 'cpu'
#     def forward(self, x):
#         x = x.to(self.device)
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)

class modelC(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out
