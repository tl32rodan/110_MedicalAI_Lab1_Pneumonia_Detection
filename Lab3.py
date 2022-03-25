#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function 
from __future__ import division

from dataloader import RetinopathyLoader
from torch.utils import data
from ResNet import *
from run_model import *


import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms
from collections import OrderedDict
from PIL import Image
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


# In[ ]:


# Batch size for training (change depending on how much memory you have)
batch_size = 4

train_set = RetinopathyLoader(root='./data/',mode='train')
train_loader = data.DataLoader(dataset=train_set,batch_size=batch_size)
test_set = RetinopathyLoader(root='./data/',mode='test')
test_loader = data.DataLoader(dataset=test_set,batch_size=batch_size)

dataloaders_dict = {'train':train_loader,
                    'val' : test_loader}


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


# Number of classes in the dataset
num_classes = 5

# Number of epochs to train for 
num_epochs_18 = 10
num_epochs_50 = 5


# Setup the loss fxn
# Because the data imbalance, we use weight Tensor w to adjust
w = [200,40,20,7,6]
w = [1- (a/sum(w)) for a in w]
w = torch.Tensor(w).to(device)
print(w)
criterion = nn.CrossEntropyLoss(weight=w)


# In[ ]:


def get_pretrained_model(model_name, num_classes,use_pretrained=True):
    model_dict = {
        'resnet18':  models.resnet18,
        'resnet34':  models.resnet34,
        'resnet50':  models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152
    }
    
    model_ft = model_dict[model_name](pretrained=use_pretrained)          
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_ftrs,128)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(128,5))
    ]))
    input_size = 224
    
    return model_ft, input_size


# ## Run ResNet pretrained model which is from `torchvision`

# In[ ]:


MODE = 'new'

if MODE == 'new':
    model_ft, _ = get_pretrained_model("resnet18", num_classes=5)    # Pretrained ResNet18
    model_ft_50, _ = get_pretrained_model('resnet50', num_classes=5) # Pretrained ResNet50
    MyResNet18  = ResNet18(num_classes=5)
    MyResNet50  = ResNet50(num_classes=5)
    
elif MODE == 'old':
    model_ft    = torch.load('./models/vision_pretrained_resnet18')
    model_ft_50 = torch.load('./models/vision_pretrained_resnet50')
    MyResNet18  = torch.load('./models/my_resnet18')
    MyResNet50  = torch.load('./models/my_resnet50')
    
else:
    print('MODE value error')
    
model_ft    = model_ft.to(device)
model_ft_50 = model_ft_50.to(device)
MyResNet18  = MyResNet18.to(device)
MyResNet50  = MyResNet50.to(device)


# ## Run Pretrained ResNet18

# In[ ]:


# Setup optimizer
optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-3, weight_decay=1e-2, momentum=0.9)
lr_sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft)

# Train and evaluate pretrained model
pretrained_loss_list, pretrained_acc_train_list, pretrained_acc_test_list, best_acc = run(model_ft, dataloaders_dict, criterion, optimizer=optimizer_ft,        scheduler=lr_sch,num_epochs=num_epochs_18)


# In[ ]:


torch.save(model_ft,'./models/vision_pretrained_resnet18')
torch.save(pretrained_loss_list,'./result_list/pretrained_loss_list')
torch.save(pretrained_acc_train_list, './result_list/pretrained_acc_train_list')
torch.save(pretrained_acc_test_list, './result_list/pretrained_acc_test_list')


# ## Run ResNet implemented by myself

# In[ ]:


opt_MyResNet18 = optim.SGD(MyResNet18.parameters(), lr=1e-3,weight_decay=5e-4, momentum=0.9)
lr_sch = optim.lr_scheduler.ReduceLROnPlateau(opt_MyResNet18)


# In[ ]:


my_loss_list_18, my_acc_train_list_18, my_acc_test_list_18, best_acc = run(MyResNet18, dataloaders_dict, criterion, optimizer=opt_MyResNet18,        scheduler=lr_sch,num_epochs=num_epochs_18)


# In[ ]:


torch.save(MyResNet18,'./models/my_resnet18')
torch.save(my_loss_list_18,'./result_list/my_loss_list_18')
torch.save(my_acc_train_list_18, './result_list/my_acc_train_list_18')
torch.save(my_acc_test_list_18, './result_list/my_acc_test_list_18')


# ## Comparison

# In[ ]:


ohist_1  = [h.cpu().numpy() for h in pretrained_acc_train_list]
myhist_1 = [h.cpu().numpy() for h in my_acc_train_list_18]
ohist_2  = [h.cpu().numpy() for h in pretrained_acc_test_list]
myhist_2 = [h.cpu().numpy() for h in my_acc_test_list_18]

plt.figure(figsize=(24,16))
plt.title("Result Comparison(ResNet18)")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,num_epochs_18+1),ohist_1,label="Train(with pretraining)")
plt.plot(range(1,num_epochs_18+1),myhist_1,label="Train(w/o pretraining)")
plt.plot(range(1,num_epochs_18+1),ohist_2,label="Test(with pretraining)")
plt.plot(range(1,num_epochs_18+1),myhist_2,label="Test(w/o pretraining)")
plt.xticks(np.arange(1, num_epochs_18+1, 1.0))
plt.legend()
plt.show()


# ## ResNet 50

# In[ ]:


# Setup optimizer
optimizer_ft_50 = optim.SGD(model_ft_50.parameters(), lr=1e-3, weight_decay=1e-3, momentum=0.9)
lr_sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft_50)

# Train and evaluate pretrained model
pretrained_loss_list_50, pretrained_acc_train_list_50, pretrained_acc_test_list_50 , best_acc= run(model_ft_50, dataloaders_dict, criterion, optimizer=optimizer_ft_50,        num_epochs=num_epochs_50,scheduler=lr_sch,        model_path = './models/vision_pretrained_resnet50', best_acc = 0.)


# In[ ]:


#torch.save(model_ft_50,'./models/vision_pretrained_resnet50')
torch.save(pretrained_loss_list_50,'./result_list/pretrained_loss_list_50')
torch.save(pretrained_acc_train_list_50, './result_list/pretrained_acc_train_list_50')
torch.save(pretrained_acc_test_list_50, './result_list/pretrained_acc_test_list_50')


# ## Run ResNet50 implemented by myself

# In[ ]:


MyResNet50 = ResNet50(num_classes)
MyResNet50 = MyResNet50.to(device)


# In[ ]:


opt_MyResNet50 = optim.SGD(MyResNet50.parameters(), lr=1e-3,weight_decay=5e-4, momentum=0.9)
lr_sch = optim.lr_scheduler.ReduceLROnPlateau(opt_MyResNet50)


# In[ ]:


my_loss_list_50, my_acc_train_list_50, my_acc_test_list_50, best_acc = run(MyResNet50, dataloaders_dict, criterion,optimizer=opt_MyResNet50,        num_epochs=num_epochs_50,scheduler=lr_sch,        model_path = './models/vision_pretrained_resnet50', best_acc = 0.)


# In[ ]:


torch.save(MyResNet50,'./models/my_resnet50')
torch.save(my_loss_list_50,'./result_list/my_loss_list_50')
torch.save(my_acc_train_list_50, './result_list/my_acc_train_list_50')
torch.save(my_acc_test_list_50, './result_list/my_acc_test_list_50')


# ## Comparison

# In[ ]:


ohist_1  = [h.cpu().numpy() for h in pretrained_acc_train_list_50]
myhist_1 = [h.cpu().numpy() for h in my_acc_train_list_50]
ohist_2  = [h.cpu().numpy() for h in pretrained_acc_test_list_50]
myhist_2 = [h.cpu().numpy() for h in my_acc_test_list_50]

plt.figure(figsize=(24,16))
plt.title("Result Comparison(ResNet50)")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,num_epochs_50+1),ohist_1,label="Train(with pretraining)")
plt.plot(range(1,num_epochs_50+1),myhist_1,label="Train(w/o pretraining)")
plt.plot(range(1,num_epochs_50+1),ohist_2,label="Test(with pretraining)")
plt.plot(range(1,num_epochs_50+1),myhist_2,label="Test(w/o pretraining)")
plt.xticks(np.arange(1, num_epochs_50+1, 1.0))
plt.legend()
plt.show()

