from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import glob
import json
import numpy as np
import torch

class bmtDataset(Dataset):
    def __init__(self,split, transform):
        DataSet=[np.array(json.load(open(x))['x']) for x in glob.glob('../../myData/*')]
        DataLen=len(DataSet)
        labelDict=[1,2,3]
        tmp1=[]
        for i in DataSet:
            tmp2=[]
            for j in i:
                tmp2.append(np.array(j,dtype=np.float32))
            tmp1.append(np.array(tmp2))
        DataSet=np.array(tmp1)        
        local_label = []
        count=0
        for dataC in DataSet:
            for dataL in range(len(dataC)):
                local_label.append(labelDict[count%3])  
            count+=1                                                      
        self.lbls = np.array(local_label,dtype=np.float32)
        self.datas = np.vstack((DataSet[i] for i in range(DataLen)))  
        
        self.transform = transform
        #mat = loadmat(f'{root}/{split}_list.mat', squeeze_me=True)
        #self.datas = mat['file_list']
        #self.imgs = [f'{root}/Images/{i}' for i in self.imgs]
        #self.lbls = mat['labels']
        assert len(self.datas) == len(self.lbls), 'mismatched length!'
        #print('Total data in {} split: {}'.format(split, len(self.datas)))
        self.lbls = self.lbls - 1
        
    def __getitem__(self, index):
        #imgpath = self.imgs[index]
        data = self.datas[index]
        lbl = int(self.lbls[index])
        if self.transform is not None:
            data = self.transform(data)
        return data,lbl   
    def __len__(self):
        return len(self.datas)    
    
class bmtDataLoader:
    def __init__(self, dataSetName, isIID, numOfClients, dev):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.isClient=False
    def bmtDataloader(self):       
        DataSet=[np.array(json.load(open(x))['x']) for x in glob.glob('../../myData/*')]
        DataLen=len(DataSet)
        labelDict=[0,1,2]
        tmp1=[]
        for i in DataSet:
            tmp2=[]
            for j in i:
                tmp2.append(np.array(j,dtype=np.float32))
            tmp1.append(np.array(tmp2))
        DataSet=np.array(tmp1)
        inter=3
        if self.isClient:
            for i in range(0,(self.num_of_clients-1)*inter+1,inter):
                local_data=np.vstack((DataSet[i],DataSet[i+1],DataSet[i+2]))
                local_label=[]
                count=0
                for m in [len(DataSet[i]),len(DataSet[i+1]),len(DataSet[i+2])]:
                    for j in range(m):
                        local_label.append(labelDict[count])
                    count+=1
                local_label=np.array(local_label,dtype=np.float32)
                local_label = np.argmax(local_label, axis=1)
        else:
            local_label = []
            count=0
            for dataC in DataSet:
                for dataL in range(len(dataC)):
                    local_label.append(labelDict[count%3])  
                count+=1                                                      
            local_label = np.array(local_label,dtype=np.float32)
            local_data = np.vstack((DataSet[i] for i in range(DataLen)))
        test_data = torch.tensor(local_data)
        test_label = torch.tensor(local_label,dtype=torch.long)
        self.bmtData = DataLoader(TensorDataset( test_data, test_label), batch_size=32, shuffle=False)    