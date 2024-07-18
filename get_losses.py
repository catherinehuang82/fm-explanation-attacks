import argparse
import os
import sys
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython import display
import copy

import pandas as pd
from sklearn.preprocessing import normalize

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Subset
from torch import linalg as LA
from statistics import mean

import random
import importlib

def main(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    print('==> Preparing data..')
                
    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
    ])

    if args.data in ['SVHN','Food101','GTSRB','FGVCAircraft']:
        trainset = getattr(torchvision.datasets,args.data)(root='data/', split='train', download=True, transform=transformation)
        testset = getattr(torchvision.datasets,args.data)(root='data/', split='test', download=True, transform=transformation)
    elif args.data in ['CIFAR10','CIFAR100']:
        trainset = getattr(torchvision.datasets,args.data)(root='data/', train=True, download=True, transform=transformation)
        testset = getattr(torchvision.datasets,args.data)(root='data/', train=False, download=True, transform=transformation)
    elif args.data=='CelebA':
        trainset = getattr(torchvision.datasets,args.data)(root='data/', split='train', download=False, target_type='attr', transform=transformation)
        testset = getattr(torchvision.datasets,args.data)(root='data/', split='test', download=False, target_type='attr',transform=transformation)
    elif args.data=='Places365':
        trainset = getattr(torchvision.datasets,args.data)(root='data/', split='train-standard', small=True, download=False, transform=transformation)
        testset = getattr(torchvision.datasets,args.data)(root='data/', split='val', small=True, download=False, transform=transformation)
    elif args.data=='INaturalist':
        trainset = getattr(torchvision.datasets,args.data)(root='data/', version='2021_train_mini', download=False, transform=transformation)
        testset = getattr(torchvision.datasets,args.data)(root='data/', version='2021_valid', download=False, transform=transformation)
    elif args.data=='ImageNet':
        trainset = getattr(torchvision.datasets,args.data)(root='data/', split='train', transform=transformation)
        testset = getattr(torchvision.datasets,args.data)(root='data/', split='val', transform=transformation)
        
    # combine train and test sets
    combinedset = torch.utils.data.ConcatDataset([trainset, testset])
    
    subset_indices = list(range(args.total_data_examples))
    combinedset = Subset(combinedset, subset_indices)
    
    if args.data in ['SVHN','CIFAR10']:
        num_classes=10
    elif args.data in ['CIFAR100','FGVCAircraft']:
        num_classes=100
    elif args.data in ['Food101']:
        num_classes=101
    elif args.data in ['GTSRB']:
        num_classes=43
    elif args.data in ['CelebA']:
        num_classes=40
    elif args.data in ['Places365']:
        num_classes=365
    elif args.data in ['ImageNet']:
        num_classes=1000
    elif args.data in ['INaturalist']:
        num_classes=10000
    elif args.data in ['EuroSAT']:
        num_classes=10
        
    model = timm.create_model(args.model, num_classes=num_classes)
    
    if 'nonDP' in args.clipping_mode:
        model_dir = f'attack_data/model_state_dicts_{args.data}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}'
    else:
        model_dir = f'attack_data/model_state_dicts_{args.data}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_epochs={args.epochs}'

    if device.type == 'cuda':
        model.load_state_dict(torch.load(f'{model_dir}/{args.experiment_no}.pt'))
    else:
        model.load_state_dict(torch.load(f'{model_dir}/{args.experiment_no}.pt',  map_location=torch.device('cpu')))

    model = model.eval()
    criterion = nn.CrossEntropyLoss()
    
    if args.verbose_flag:
        print(f'Starting loss collection, clipping mode = {args.clipping_mode}, eps = {args.epsilon}')
        
    # represent the combinedset as a dataloader (batch size 1 so we get individual per-example losses)
    combinedloader = torch.utils.data.DataLoader(
        combinedset, batch_size=1, shuffle=False, num_workers=1)

    losses = np.array([])
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(combinedloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses = np.append(losses, loss.item())

    if args.verbose_flag:
        print(f'Loss collection done for clipping mode = {args.clipping_mode}, eps = {args.epsilon}')
    
    if not args.dry_run:
        losses_df = pd.DataFrame({f'{args.experiment_no}': losses})
        if 'nonDP' not in args.clipping_mode:
            losses_dir = f'attack_data/losses_{args.data}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_epochs={args.epochs}'
        else:
            losses_dir = f'attack_data/losses_{args.data}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}'
        if not os.path.exists(losses_dir):
            os.makedirs(losses_dir)
            print(f'Created directory: {losses_dir}')
        
        losses_df.to_csv(f'{losses_dir}/{args.experiment_no}.csv', index=False)
        
         
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='CIFAR10')
    parser.add_argument('--epochs', default=30, type=int,
                        help='number of epochs')
    parser.add_argument('--epsilon', default=2.0, type=float, help='target epsilon')
    parser.add_argument('--clipping_mode', default='nonDP', type=str)
    parser.add_argument('--model', default='vit_small_patch16_224', type=str) # try: resnet18
    parser.add_argument('--dimension', type=int,default=224)
    parser.add_argument('--experiment_no', type=int, default=0)
    parser.add_argument('--dry_run', type=lambda x: x.lower() == 'true', default=False) # whether or not we want to save data
    parser.add_argument('--verbose_flag', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--total_data_examples', type=int, default=20000)
    args = parser.parse_args()
    main(args)
    
 