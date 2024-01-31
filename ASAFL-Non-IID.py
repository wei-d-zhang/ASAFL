#!/usr/bin/python
# -*- coding: UTF-8 -*-
import copy
import math
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib import pyplot as plt
import csv
import random
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch import optim
from torch.autograd._functions import tensor
from torch.nn import init
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import time
import numpy as np
from model import fashion_ResNet
from datasplit import dirichlet_split_noniid


client_n = 5  # num of clients
max_comunication = 50  # communication rounds
root = './' 
epochs = 1 #local training interaion
learn_rate = 0.0001  #learning rate
batch_size = 128 
train_loader, dataset_label, dataset_label_client, train_dataset_client = [], [], [], []
fedavg_loss = []
fedavg_accuracy = []
decay_rate =  0.5  ###learning rate decay
    
    
train_dataset = datasets.FashionMNIST(root="./FashionMNIST/", train=True, transform=transforms.ToTensor(), download=True)
train_labels = np.array(train_dataset.targets)
DIRICHLET_ALPHA = 0.5 #########Non-IID参数，越小，non-iid程度越大

#######客户端数据集划分索引
client_idcs = dirichlet_split_noniid(train_labels, alpha=DIRICHLET_ALPHA, n_clients=client_n) #############客户端数据划分
client_datas = []

for i in range(client_n):
    j = []
    client_datas.append(j)
    train_loader.append(j)
    
######将数据集按non-iid划分给10个客户端
for i in range(client_n):
    for num1 in client_idcs[i]:
        client_datas[i].append(train_dataset[num1])

for i in range(client_n):
    train_loader[i] = torch.utils.data.DataLoader(dataset=client_datas[i], batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.FashionMNIST(root='./FashionMNIST/', train=False, transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


current_model = fashion_ResNet()


model_c = []

class FL(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = {}
        self.model = self.get_model()
        self.server_pre_weight = {} #记录服务器上一轮模型参数

    def get_model(self):
        model = current_model
        model.to(self.device)
        return model

    def run(self):
        self.recv_data()

    def recv_data(self):
        comunication_n = 0
        #aggregation_times = 0
        #local_train_num = 0
        client_similarity = 0        #########客户端与服务器模型相似度
        simi_list = [[] for _ in range(client_n)]
        #local_iters = [0 for _ in range(5)]
        model = self.model
        cloud_weight = model.state_dict()
        for i in range(client_n):
            model_c.append(cloud_weight)
        
        while comunication_n < max_comunication:  # communication number
            if comunication_n == 0:
                self.server_pre_weight = copy.deepcopy(self.model.state_dict()) #第一轮时，服务器上一轮模型参数为初始化时的模型参数 

            for i in range(client_n):
                simi_list[i].append(0)
                #local_iters[i] = local_iters[i] + 1
                if model_c[i]:
                    # local traning
                    model_c[i],gradients = copy.deepcopy(self.train(epochs, train_loader[i],model_c[i],comunication_n))
                    #local_train_num = local_train_num + 1
                    
                    ####FedAA#######
                    #计算客户端模型与服务器模型相似度
                    local_weight = copy.deepcopy(model_c[i])
                    pre_server_weight = copy.deepcopy(self.server_pre_weight)
                    similarity = []
                    for name in cloud_weight:
                        if 'tracked' in name:
                            break
                        local_weight[name] = local_weight[name]#.unsqueeze(0)
                        pre_server_weight[name] = pre_server_weight[name]#.unsqueeze(0)
                        sim = torch.cosine_similarity(local_weight[name], pre_server_weight[name],dim =-1)
                        sim_mean = torch.mean(sim)
                        similarity.append(sim_mean)
                    client_similarity = sum(similarity)/len(similarity)
                    #print("client_similarity: ",client_similarity.item())
                    simi_list[i].append(client_similarity)
                    simi_tensor = torch.tensor(simi_list[i])
                    simi_average_tensor = torch.mean(simi_tensor)
                    #print(simi_average_tensor)
                    similarity.clear()
                    
                    if client_similarity > simi_average_tensor:
                        parameters = copy.deepcopy([param.data for param in self.model.parameters()])
                        simi_value = client_similarity.item()
                        lambd = 1/2*(np.tanh(2*simi_value))
                        self.sgd_update(parameters,gradients,lambd,comunication_n)
                        for param, new_param in zip(self.model.parameters(), parameters):
                            param.data = new_param.data
                        self.test(comunication_n)
                        for j in range(client_n):
                            model_c[j] = copy.deepcopy(self.model.state_dict())
                        self.server_pre_weight = copy.deepcopy(self.model.state_dict())
                        comunication_n = comunication_n + 1
                        if comunication_n > 50:
                            break
                    else:
                        print("pass")
                        #local_weight = copy.deepcopy(model_c[i])
                        #pre_server_weight = self.server_pre_weight
                        pass
                    #comunication_n = comunication_n + 1
                    
                
                else:
                    print("break")
                    break

        print('训练完毕')
        #confusion.plot()
        
        f = open('results/Non-IID/ASAFL-0.5-5.csv', 'w', encoding='utf-8', newline='')
        csv_write = csv.writer(f)
        csv_write.writerow(fedavg_accuracy)
            
    def sgd_update(self,parameters, gradients,lambdaa,iteration):
        """
        the SGD update rule.
        :param parameters: List of model parameters
        :param gradients: List of parameter gradients
        :param lr: Learning rate
        """
        decay_interval=10
        learn_ratee = lambdaa * learn_rate
        decayed_lr = learn_ratee * (decay_rate ** (iteration // decay_interval))
        #print("decayed_lr",decayed_lr)
        
        # Update parameters using SGD
        for param, grad in zip(parameters, gradients):
            param.data = param.data - decayed_lr * grad
        

    def train(self, epoch, t_dataset, model_para_client,comun):
        model_param = model_para_client
        model = current_model
        model.load_state_dict(model_param)
        model.train()  # 设置为trainning模式
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.SGD(model.parameters(), lr=learn_rate, weight_decay=0.0001)  # 初始化优化器
        #optimizer = optim.Adam(model.parameters(), lr=learn_rate, betas=(0.9, 0.999),eps=1e-8, amsgrad=False)
        for i in range(1, epoch + 1):
            for batch_idx, (data, target) in enumerate(t_dataset):
                data = data.to(self.device)
                target = target.to(self.device)
                data, target = Variable(data), Variable(target)  # 把数据转换成Variable
                #optimizer.zero_grad()  # 优化器梯度初始化为零
                output = model(data)  # 把数据输入网络并得到输出，即进行前向传播
                loss = criterion(output,target)
                loss.backward()  # 反向传播梯度
                gradients = []
                for param in model.parameters():
                    gradients.append(param.grad.data)
                parameters = copy.deepcopy([param.data for param in model.parameters()])
                self.sgd_update(parameters,gradients,1,comun)
                for param, new_param in zip(model.parameters(), parameters):
                    param.data = new_param.data
                #optimizer.step()  # 结束一次前传+反传之后，更新参数
        model_state = copy.deepcopy(model.state_dict())
        return model_state,gradients
    
    def test(self,n):
        self.model.eval()  # 设置为test模式
        test_loss = 0  # 初始化测试损失值为0
        correct = 0  # 初始化预测正确的数据个数为0
        
        for data, target in test_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            data, target = Variable(data), Variable(target)  # 计算前要把变量变成Variable形式，因为这样子才有梯度
            output = self.model(data)
            _,predicts = torch.max(output.data, 1)
            #confusion.update(predicts.cpu().numpy(), target.cpu().numpy())
                
            test_loss += F.cross_entropy(output, target,
                                             size_average=False).item()  # sum up batch loss 把所有loss值进行累加
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加
           
        test_loss /= len(test_loader.dataset)  # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
        nowtime = time.strftime("%Y-%m-%d %H:%M:%S")
        fedavg_loss.append(round(test_loss, 4))
        acc = (100. * correct / len(test_loader.dataset)).tolist()
        fedavg_accuracy.append(round(acc, 2))
        print('\n{} 第{}轮训练测试: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(nowtime, n + 1,
                                                                                        test_loss, correct,
                                                                                        len(test_loader.dataset),
                                                                                        100. * correct / len(test_loader.dataset)))

def main():
    fl = FL()
    fl.run()


# 当模块被直接运行时，以下代码块将被运行，当模块是被导入时，代码块不被运行。
if __name__ == "__main__":
    main()

