import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import ChestXRayData
import torchvision 
from torchvision import datasets
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from torchvision import models
# from tqdm import tqdm_notebook as tqdm
import time
from tqdm import tqdm
import warnings
import copy
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.metrics import accuracy_score,classification_report, f1_score,roc_auc_score

def images_transforms(phase):
    if phase == 'training':
        data_transformation =transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            #transforms.RandomEqualize(10),
            transforms.RandomRotation(degrees=(-25,20)),
            transforms.RandomCrop(96),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    else:
        data_transformation=transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    
    return data_transformation

def imshow(img):
    plt.figure(figsize=(20, 20))
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def Convlayer(in_channels,out_channels,kernel_size,padding=1,stride=1):
    conv =  nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    return conv

class NeuralNet(nn.Module):
    def __init__(self,num_classes):
        super(NeuralNet,self).__init__()
        
        self.conv1 = Convlayer(in_channels=3,out_channels=32,kernel_size=3)
        self.conv2 = Convlayer(in_channels=32,out_channels=64,kernel_size=3)
        self.conv3 = Convlayer(in_channels=64,out_channels=128,kernel_size=3)
        self.conv4 = Convlayer(in_channels=128,out_channels=256,kernel_size=3)
        self.conv5 = Convlayer(in_channels=256,out_channels=512,kernel_size=3)
        
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=x.view(-1,512*4*4)
        x=self.classifier(x)

        return x

class ResNet50(nn.Module):
   def __init__(self,num_class,pretrained_option=False):
        super(ResNet50,self).__init__()
        self.model=models.resnet50(pretrained=pretrained_option)
        
        if pretrained_option==True:
            for param in self.model.parameters():
                param.requires_grad=False

        num_neurons=self.model.fc.in_features
        self.model.fc=nn.Sequential(
                        nn.Linear(num_neurons, 512),
                        nn.ReLU(),
                        nn.Linear(512, 128),
                        nn.ReLU(),
                        nn.Linear(128, num_class),
                      )
        
   def forward(self,X):
        out=self.model(X)
        return out

class ResNet101(nn.Module):
   def __init__(self,num_class,pretrained_option=False):
        super(ResNet101,self).__init__()
        self.model=models.resnet101(pretrained=pretrained_option)
        
        if pretrained_option==True:
            for param in self.model.parameters():
                param.requires_grad=False

        num_neurons=self.model.fc.in_features
        self.model.fc=nn.Sequential(
                        nn.Linear(num_neurons, 512),
                        nn.ReLU(),
                        nn.Linear(512, 128),
                        nn.ReLU(),
                        nn.Linear(128, num_class),
                      )
        
   def forward(self,X):
        out=self.model(X)
        return out

class ResNet152(nn.Module):
   def __init__(self,num_class,pretrained_option=False):
        super(ResNet152,self).__init__()
        self.model=models.resnet152(pretrained=pretrained_option)
        
        if pretrained_option==True:
            for param in self.model.parameters():
                param.requires_grad=False

        num_neurons=self.model.fc.in_features
        self.model.fc=nn.Sequential(
                        nn.Linear(num_neurons, 512),
                        nn.ReLU(),
                        nn.Linear(512, 128),
                        nn.ReLU(),
                        nn.Linear(128, num_class),
                      )
        
   def forward(self,X):
        out=self.model(X)
        return out

    
def plot_confusion_matrix(cm, title = 'Normalized confusion matrix', cmap=plt.cm.Blues, num_classes=5):
    classes = range(num_classes)
    # Normalized
#     for i in range(len(cm)):
#         cm[i] = cm[i]/cm[i].sum()

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    
def evaluate(model, device, test_loader, num_class):
    confusion_matrix = torch.zeros(num_class,num_class)
    correct=0
    TP=0
    TN=0
    FP=0
    FN=0
    with torch.set_grad_enabled(False):
        model.eval()
        for idx,(data,label) in enumerate(test_loader):
            data = data.to(device,dtype=torch.float)
            label = label.to(device,dtype=torch.long)
            predict = model(data)
            pred = torch.max(predict,1).indices
            #correct += pred.eq(label).cpu().sum().item()
            for j in range(data.size()[0]):
                #print ("{} pred label: {} ,true label:{}" .format(len(pred),pred[j],int(label[j])))
                if (int (pred[j]) == int (label[j])):
                    correct +=1
                if (int (pred[j]) == 1 and int (label[j]) ==  1):
                    TP += 1
                if (int (pred[j]) == 0 and int (label[j]) ==  0):
                    TN += 1
                if (int (pred[j]) == 1 and int (label[j]) ==  0):
                    FP += 1
                if (int (pred[j]) == 0 and int (label[j]) ==  1):
                    FN += 1
                    
            for t, p in zip(label.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        print ("TP : " , TP)
        print ("TN : " , TN)
        print ("FP : " , FP)
        print ("FN : " , FN)

        print ("num_correct :",correct ," / " , len(test_loader.dataset))
        Recall = TP/(TP+FN)
        print ("Recall : " ,  Recall )

        Precision = TP/(TP+FP)
        print ("Preecision : " ,  Precision )

        F1_score = 2 * Precision * Recall / (Precision + Recall)
        print ("F1 - score : " , F1_score)

        correct = (correct/len(test_loader.dataset))*100.
        print ("Accuracy : " , correct ,"%")

    return correct , Recall , Precision , F1_score, confusion_matrix

if __name__=="__main__":
    IMAGE_SIZE=(128,128)
    batch_size=128
    num_classes=2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    test_path='chest_xray/test'

    testset=ChestXRayData(test_path, trans=images_transforms('test'))

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model_names = {'ResNet50': ResNet50,
                   'ResNet101': ResNet101,
                   'ResNet152': ResNet152,}
    for model_name in model_names.keys(): 
        print(model_name)
        ckpt = './'+ model_name +'.pt'

        model = model_names[model_name](num_class=2).cuda()
        
        model.load_state_dict(torch.load(ckpt), strict=True)

        correct , Recall , Precision , F1_score, cm = evaluate(model, device, test_loader, 2)
        
        plot_confusion_matrix(cm, 'Confusion Matrix of Our ' + model_name, num_classes=2)
        plt.savefig(model_name+'_confusion_matrix.png')
        
        print('----------------')
    
