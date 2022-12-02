import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import json
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description = "Parser of training script")

parser.add_argument('data_dir', help = 'Data directory. Mandatory argument', type = str)
parser.add_argument('--save_dir', help = 'Saving directory. Optional argument', type = str)
parser.add_argument('--arch', help = 'Choose architecture (vgg13 or vgg16). Default is vgg16', type = str)
parser.add_argument('--lrn', help = 'Learning rate. Default is 0.001', type = float)
parser.add_argument('--hidden_units', help = 'Hidden units in classifier. Default is 5000', type = int)
parser.add_argument('--epochs', help = 'Number of epochs. Default is 5', type = int)
parser.add_argument('--gpu', help = "Option to use GPU. Default is CPU", type = str)

args = parser.parse_args()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
if args.gpu:
    if args.gpu.lower() == 'gpu':
        device = 'cuda'
    else:
        device = 'cpu'

if data_dir:
    data_transforms = {'train':transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]),
                      'test': transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]),
                      'valid': transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])}
    image_datasets = {'train':datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                     'test':datasets.ImageFolder(test_dir, transform=data_transforms['test']),
                     'valid':datasets.ImageFolder(valid_dir,transform=data_transforms['valid'])}
    dataloaders = {'train':torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                  'test':torch.utils.data.DataLoader(image_datasets['test'], batch_size=64),
                  'valid':torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64)}
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
def load_model(arch,hidden):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arch=='vgg13':
        arch='vgg13'
        model = models.vgg13(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden:
            classifier=nn.Sequential(
                            nn.Linear(25088, 4096),
                            nn.ReLU(),
                            nn.Dropout(p=0.3),
                            nn.Linear(4096, hidden),
                            nn.ReLU(),
                            nn.Dropout(p =0.3),
                            nn.Linear(hidden, 102),
                            nn.LogSoftmax(dim =1)
                            )
        else:
            classifier=nn.Sequential(
                        nn.Linear(25088, 4096),
                        nn.ReLU(),
                        nn.Dropout(p = 0.3),
                        nn.Linear(4096, 2048),
                        nn.ReLU(),
                        nn.Dropout(p = 0.3),
                        nn.Linear(2048, 102),
                        nn.LogSoftmax(dim =1)
                        )
    else:
        arch='vgg16'
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden:
            classifier=nn.Sequential(nn.Linear(25088, 5000),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(5000, hidden),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(hidden, 102),
                                             nn.LogSoftmax(dim=1))
        else:
            classifier=nn.Sequential(nn.Linear(25088, 5000),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(5000, 500),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(500, 102),
                                             nn.LogSoftmax(dim=1))
    model.classifier=classifier
    return model,arch

def validation(model, criterion):
    model.to(device)
    valid_loss = 0
    accuracy = 0
    for inputs, labels in dataloaders['valid']:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return valid_loss, accuracy

model, arch = load_model(args.arch, args.hidden_units)
criterion = nn.NLLLoss()
if args.lrn: 
    optimizer = optim.Adam(model.classifier.parameters(), lr = args.lrn)
else:
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)
model.to(device) 
if args.epochs:
    epochs = args.epochs
else:
    epochs = 3
    
print_every = 5
steps = 0

for e in range(epochs):
    running_loss = 0
    for inputs,labels in dataloaders['train']:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad () 

        outputs = model.forward(inputs) 
        loss = criterion(outputs, labels) 
        loss.backward()
        optimizer.step()
        running_loss += loss.item() 

        if steps % print_every == 0:
            model.eval() 

            with torch.no_grad():
                valid_loss, accuracy = validation(model, criterion)

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(dataloaders['valid'])),
                  "Valid Accuracy: {:.3f}%".format(accuracy/len(dataloaders['valid'])*100))

            running_loss = 0
            model.train()


model.to('cpu') 
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'arch': arch,
              'mapping':    model.class_to_idx
             }

if args.save_dir:
    torch.save(checkpoint, args.save_dir + '/checkpoint.pth')
else:
    torch.save(checkpoint, 'checkpoint.pth')