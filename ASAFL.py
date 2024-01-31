#!/usr/bin/python
# -*- coding: UTF-8 -*-
import copy
import math

import glob
import scipy.io as sio
import argparse
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib import pyplot as plt
import csv
import random
from socket import *
from threading import Thread
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch import optim
from torch.autograd._functions import tensor
from torch.nn import init
import torch.nn.functional as F
import pickle, struct
import torchvision
from torchvision import datasets, transforms
from time import sleep
import time
from collections import OrderedDict
import numpy as np
from prettytable import PrettyTable
#from sklearn.metrics import confusion_matrix
from data_process import Widar_Dataset,data_reader,cifar_10_load,data_reader1,SVHN_load
from model import cifar10_MLP,cifar10_LeNet,fashion_MLP,fashion_LeNet,fashion_ResNet,Widar_LeNet,Widar_ResNet18,Widar_MLP,SVHN_MLP,SVHN_ResNet18
from resnet import cifar10_ResNet18
from plot import ConfusionMatrix


client_n = 5  # num of clients
max_comunication = 50  # communication rounds
root = './' 
epochs = 1 #local training interaion
learn_rate = 0.001  #learning rate
batch_size = 128 
train_loader, dataset_label, dataset_label_client, train_dataset_client = [], [], [], []
fedavg_loss = []
fedavg_accuracy = []
decay_rate =  0.95  ###learning rate decay




for i in range(client_n):
    j = []
    train_loader.append(j)
    
#Import training data
train_dataset = datasets.FashionMNIST(root="./FashionMNIST/", train=True, transform=transforms.ToTensor(), download=True)
#train_dataset = datasets.CIFAR10(root='./cifar10', train=True, transform=transforms.ToTensor()) 

# Define a custom data set for distributing data to clients
class ClientDataset(Dataset):
    def __init__(self, dataset, client_indices):
        self.dataset = dataset
        self.client_indices = client_indices

    def __len__(self):
        return len(self.client_indices)

    def __getitem__(self, index):
        return self.dataset[self.client_indices[index]]

# Split FashionMNIST Data Integration 10 subsets (one subset per client)
data_split = random_split(train_dataset, [len(train_dataset) // client_n] * client_n)

# Create a list of 10 client datasets
client_datasets = [ClientDataset(train_dataset, split.indices) for split in data_split]

# Create a list of 10 client-side data loaders
train_loader = [DataLoader(client_dataset, batch_size=batch_size, shuffle=True) for client_dataset in client_datasets]
#Load the test datasets
test_dataset = torchvision.datasets.FashionMNIST(root='./FashionMNIST/', train=False, transform=torchvision.transforms.ToTensor())
#test_dataset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

#Loading model
#current_model = fashion_MLP()
#current_model = fashion_LeNet()
current_model = fashion_ResNet()
#current_model = cifar10_LeNet()
#current_model = cifar10_ResNet18()
#current_model = cifar10_MLP()
#current_model = cifar10_LeNet()
#current_model = SVHN_ResNet18()
#current_model = SVHN_MLP()

model_c = []

class FL(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = {}
        self.model = self.get_model()
        self.server_pre_weight = {} #Records the last round of model parameters on the server


    def get_model(self):
        model = current_model
        model.to(self.device)
        return model

    def run(self):
        self.recv_data()

    def recv_data(self):
        comunication_n = 0
        client_similarity = 0        #########Client and server model similarity
        simi_list = [[] for _ in range(client_n)]
        model = self.model
        
        #Distribute the server initial model to the clients
        cloud_weight = model.state_dict()
        for i in range(client_n):
            model_c.append(cloud_weight)
        #Start server iteration
        while comunication_n < max_comunication:  # communication number
            #In the first round, the model parameters on the server are the model parameters at the time of initialization
            if comunication_n == 0:
                self.server_pre_weight = copy.deepcopy(self.model.state_dict()) 

            for i in range(client_n):
                if model_c[i]:
                    # The local model is trained with SGD as gradient update mode, and the parameters and gradient of the local model are returned
                    model_c[i],gradients = copy.deepcopy(self.train(epochs, train_loader[i],model_c[i],comunication_n))
                    
                    ####ASAFL#######
                    #Calculate the similarity between client model and server model
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
                    #print(simi_list[i])
                    simi_tensor = torch.tensor(simi_list[i])
                    simi_average_tensor = torch.mean(simi_tensor)
                    #print(simi_average_tensor)
                    similarity.clear()
                    #The client upload decision is made when the similarity between the client epicycle and the server model is less than the threshold
                    if client_similarity <= simi_average_tensor:
                        parameters = copy.deepcopy([param.data for param in self.model.parameters()])
                        simi_value = client_similarity.item()
                        #Calculate the attenuation coefficient when the server is updated
                        lambd = 1/2*(np.tanh(2*simi_value))
                        #Parameter update based on SGD
                        self.sgd_update(parameters,gradients,lambd,comunication_n)
                        for param, new_param in zip(self.model.parameters(), parameters):
                            param.data = new_param.data
                        #The server tests the current model parameters on the test sets
                        self.test(comunication_n)
                        #The server model parameters are loaded to the client as initial model parameters for a new round of client local training
                        for j in range(client_n):
                            model_c[j] = copy.deepcopy(self.model.state_dict())
                        self.server_pre_weight = copy.deepcopy(self.model.state_dict())
                        comunication_n = comunication_n + 1
                        if comunication_n > 50:
                            break
                    else:
                        print("pass")
                        pass
                    
                
                else:
                    print("break")
                    break

        print('Complete training')
        '''
        f = open('results/ASAFL/Fashion-Mnist/Resnet-1.csv', 'w', encoding='utf-8', newline='')
        csv_write = csv.writer(f)
        csv_write.writerow(fedavg_accuracy)
        '''
    def sgd_update(self,parameters, gradients,lambdaa,iteration):
        """
        A simple manual implementation of the SGD update rule.
        :param parameters: List of model parameters
        :param gradients: List of parameter gradients
        :param lr: Learning rate
        """
        #Attenuation of learning rate
        decay_interval=10
        learn_ratee = lambdaa * learn_rate
        decayed_lr = learn_ratee * (decay_rate ** (iteration // decay_interval))
        
        # Update parameters using SGD
        for param, grad in zip(parameters, gradients):
            param.data = param.data - decayed_lr * grad
        

    def train(self, epoch, t_dataset, model_para_client,comun):
        model_param = model_para_client
        model = current_model
        #Loading of local model parameters
        model.load_state_dict(model_param)
        model.train()  # Set this parameter to trainning mode
        criterion = nn.CrossEntropyLoss() #Initialize the loss function
        for i in range(1, epoch + 1):
            for batch_idx, (data, target) in enumerate(t_dataset):
                data = data.to(self.device)
                target = target.to(self.device)
                data, target = Variable(data), Variable(target)  # Convert the data to Variable
                output = model(data)  # Input data into the network and get output, that is, forward propagation
                loss = criterion(output,target)
                loss.backward()  # Back propagation gradient
                gradients = []
                for param in model.parameters():
                    gradients.append(param.grad.data)
                parameters = copy.deepcopy([param.data for param in model.parameters()])
                self.sgd_update(parameters,gradients,1,comun)
                for param, new_param in zip(model.parameters(), parameters):
                    param.data = new_param.data
        model_state = copy.deepcopy(model.state_dict())
        return model_state,gradients
    
    def test(self,n):
        self.model.eval()  # Set this parameter to test mode
        test_loss = 0  #The initial test loss value is 0
        correct = 0  # Initialize the number of correctly predicted data to be 0
        
        for data, target in test_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            data, target = Variable(data), Variable(target)  # The Variable should be changed into variable form before calculation
            output = self.model(data)
            _,predicts = torch.max(output.data, 1)
            #confusion.update(predicts.cpu().numpy(), target.cpu().numpy())
                
            test_loss += F.cross_entropy(output, target,
                                             size_average=False).item()  # sum up batch loss 
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # Add up the number of data that predicted correctly
           
        test_loss /= len(test_loader.dataset)  # the average loss is obtained by dividing by the total data length

        nowtime = time.strftime("%Y-%m-%d %H:%M:%S")
        fedavg_loss.append(round(test_loss, 4))
        acc = (100. * correct / len(test_loader.dataset)).tolist()
        fedavg_accuracy.append(round(acc, 2))
        print('\n{} Round {} training test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(nowtime, n + 1,
                                                                                        test_loss, correct,
                                                                                        len(test_loader.dataset),
                                                                                        100. * correct / len(test_loader.dataset)))

def main():
    fl = FL()
    fl.run()


if __name__ == "__main__":
    main()

