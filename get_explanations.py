'''For a single model, compute post-hoc explanations on all data 
examples and save per-example attack scores: explanation variance, L1 norm, L2 norm.'''

import argparse
import os
import sys
import multiprocessing

from captum.attr import InputXGradient, Saliency, IntegratedGradients, GradientShap, LRP, GuidedBackprop

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
    
    model_dir = f'attack_data/model_state_dicts_{args.data}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}'
    
    # load fine-tuned model
    if device.type == 'cuda':
        model.load_state_dict(torch.load(f'{model_dir}/{args.experiment_no}.pt'))
    else:
        model.load_state_dict(torch.load(f'{model_dir}/{args.experiment_no}.pt',  map_location=torch.device('cpu')))

    model = model.eval()
    
    if args.explanation_type == 'gs':
        explainer = GradientShap(model)
    elif args.explanation_type == 'ig':
        explainer = IntegratedGradients(model)
    elif args.explanation_type == 'ixg':
        explainer = InputXGradient(model)
    elif args.explanation_type == 'lrp':
        explainer = LRP(model)
    elif args.explanation_type == 'sl':
        explainer = Saliency(model)
    elif args.explanation_type == 'gb':
        explainer = GuidedBackprop(model)

    if args.verbose_flag:
        print(f'Starting explanations for type = {args.explanation_type}, clipping mode = {args.clipping_mode}, eps = {args.epsilon}')
        
    variances = np.array([])
    norms_l1 = np.array([])
    norms_l2 = np.array([])
    norms_linf = np.array([])
    if args.explanation_type not in ['ixg', 'sl', 'gb']:
        deltas = np.array([])
            
    for i, (x, y) in tqdm(enumerate(combinedset)):
        inp = torch.unsqueeze(x, dim=0) # [1, 3, 224, 224]
        inp = inp.float()
        output = model.forward(inp)
        pred = torch.argmax(output, axis=-1).item()
        if args.explanation_type == 'gs':
            baseline_dist = torch.randn(inp.size()) * 0.001
            attributions, delta = explainer.attribute(inp, n_samples=5, baselines=baseline_dist, target=pred,
                           return_convergence_delta=True) # shape: torch.Size([1, 3, 224, 224]
        elif args.explanation_type == 'ig':
            baseline = torch.zeros(inp.size())
            attributions, delta = explainer.attribute(inp, baseline, n_steps=args.nsamples,
                                                      target=pred, return_convergence_delta=True)
        elif args.explanation_type == 'lrp':
            baseline = torch.zeros(inp.size())
            attributions, delta = explainer.attribute(inp, target=pred, return_convergence_delta=True)
        elif args.explanation_type in ['ixg', 'sl', 'gb']:
            attributions = explainer.attribute(inp, target=pred) 

        if args.explanation_type not in ['ixg', 'sl', 'gb']:
            deltas = np.append(deltas, torch.mean(torch.abs(delta)).item())
        variances = np.append(variances, np.sum(torch.var(attributions, dim=(2, 3), unbiased=True).detach().numpy()))
        norms_l1 = np.append(norms_l1, torch.norm(attributions, p=1).detach().item())
        norms_l2 = np.append(norms_l2, torch.norm(attributions, p=2).detach().item())
        norms_linf = np.append(norms_linf, torch.norm(attributions, p=float('inf')).detach().item())

    if args.verbose_flag:
        print(f'Explanations done for type = {args.explanation_type}, clipping mode = {args.clipping_mode}, eps = {args.epsilon}, type={args.explanation_type}, nsamples={args.nsamples}')
    
    if not args.dry_run:
        variances_df = pd.DataFrame({f'{args.experiment_no}': variances})
        norms_l1_df = pd.DataFrame({f'{args.experiment_no}': norms_l1})
        norms_l2_df = pd.DataFrame({f'{args.experiment_no}': norms_l2})
        norms_linf_df = pd.DataFrame({f'{args.experiment_no}': norms_linf})
        if args.explanation_type not in ['ixg', 'sl', 'gb']:
            deltas_df = pd.DataFrame({f'{args.experiment_no}': deltas})
        variances_dir = f'attack_data/variances_{args.data}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}'
        norms_l1_dir = f'attack_data/norms_l1_{args.data}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}'
        norms_l2_dir = f'attack_data/norms_l2_{args.data}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}'
        norms_linf_dir = f'attack_data/norms_linf_{args.data}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}'
        deltas_dir = f'attack_data/deltas_{args.data}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_type={args.explanation_type}_nsamples={args.nsamples}_epochs={args.epochs}'
        if not os.path.exists(variances_dir):
            os.makedirs(variances_dir)
            print(f'Created directory: {variances_dir}')
        if not os.path.exists(norms_l1_dir):
            os.makedirs(norms_l1_dir)
            print(f'Created directory: {norms_l1_dir}')
        if not os.path.exists(norms_l2_dir):
            os.makedirs(norms_l2_dir)
            print(f'Created directory: {norms_l2_dir}')
        if not os.path.exists(norms_linf_dir):
            os.makedirs(norms_linf_dir)
            print(f'Created directory: {norms_linf_dir}')
        if args.explanation_type not in ['ixg', 'sl', 'gb'] and not os.path.exists(deltas_dir):
            os.makedirs(deltas_dir)
            print(f'Created directory: {deltas_dir}')
        
        variances_df.to_csv(f'{variances_dir}/{args.experiment_no}.csv', index=False)
        norms_l1_df.to_csv(f'{norms_l1_dir}/{args.experiment_no}.csv', index=False)
        norms_l2_df.to_csv(f'{norms_l2_dir}/{args.experiment_no}.csv', index=False)
        norms_linf_df.to_csv(f'{norms_linf_dir}/{args.experiment_no}.csv', index=False)

        if args.explanation_type not in ['ixg', 'sl', 'gb']:
            deltas_df.to_csv(f'{deltas_dir}/{args.experiment_no}.csv', index=False)

         
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='CIFAR10')
    parser.add_argument('--bs', default=1000, type=int, help='batch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='number of epochs')
    parser.add_argument('--mini_bs', type=int, default=50)
    parser.add_argument('--clipping_mode', default='nonDP', type=str)
    parser.add_argument('--model', default='vit_small_patch16_224', type=str) # try: resnet18
    parser.add_argument('--dimension', type=int,default=224)
    parser.add_argument('--origin_params', nargs='+', default=None)
    
    parser.add_argument('--explanation_type', default='ixg', choices=['gs', 'ig', 'ixg', 'lrp', 'sl', 'gb'])
    parser.add_argument('--nsamples', type=int, default=20)
    parser.add_argument('--experiment_no', type=int, default=0)
    parser.add_argument('--dry_run', type=lambda x: x.lower() == 'true', default=False) # whether or not we want to save data
    parser.add_argument('--verbose_flag', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--total_data_examples', type=int, default=20000)
    args = parser.parse_args()
    main(args)
    
 