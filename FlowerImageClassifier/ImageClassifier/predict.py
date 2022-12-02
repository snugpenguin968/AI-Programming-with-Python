import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
from collections import OrderedDict
from PIL import Image
import argparse
import json

parser = argparse.ArgumentParser (description = "Parser of prediction script")

parser.add_argument ('image_dir', help = 'Path to image. Mandatory argument', type = str)
parser.add_argument ('load_dir', help = 'Path to checkpoint. Mandatory argument', type = str)
parser.add_argument ('--top_k', help = 'Top K most likely classes. Optional', type = int)
parser.add_argument ('--category_names', help = 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str)
parser.add_argument ('--gpu', help = "Option to use GPU. Optional", type = str)

def load_model(file_path):
    checkpoint = torch.load(file_path) 
    if checkpoint['arch'] == 'vgg13':
        model = models.vgg13 (pretrained = True)
    else: 
        model = models.vgg16 (pretrained = True)
    model.classifier = checkpoint ['classifier']
    model.load_state_dict (checkpoint ['state_dict'])
    model.class_to_idx = checkpoint ['mapping']
    for param in model.parameters():
        param.requires_grad = False 
    return model

def process_image(image_path):
    image = Image.open(image_path)
   
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transformed_image = transformations(image)
    transformed_image=np.array(transformed_image)
    return transformed_image

def predict(image_path,model,topk,device):
    model.to(device)
    image = process_image(image_path)
    if device == 'cuda':
        image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    else:
        image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze_(0)
    image = image.float()
    model.to(device)
    image.to(device)
    if device=='cuda':
        with torch.no_grad():
            output = model.forward(image.cuda())
    else:
        with torch.no_grad():
            output = model.forward(image)
            
    prob = F.softmax(output.data,dim=1)
    probs, classes=prob.topk(topk)
    probs=probs[0].tolist()
    classes=classes[0].tolist()
    for i in range(len(classes)):
        if classes[i]==0:
            classes[i]=1
    classes=[str(i) for i in classes]
    return probs,classes

args = parser.parse_args()
file_path = args.image_dir
device='cpu'
if args.gpu:
    if args.gpu.lower() == 'gpu':
        device = 'cuda'


if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

model = load_model(args.load_dir)

if args.top_k:
    nm_cl = args.top_k
else:
    nm_cl = 1

probs, classes = predict(file_path, model, nm_cl, device)
class_names = [cat_to_name[item] for item in classes]

for l in range(nm_cl):
     print("Number: {}/{}.. ".format(l+1, nm_cl),
            "Class name: {}.. ".format(class_names [l]),
            "Probability: {:.3f}..% ".format(probs [l]*100),
            )
    