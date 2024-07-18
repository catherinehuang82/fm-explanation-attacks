'''Fine-tune a single vision transformer model and save its state dictionary.'''

from fastDP import PrivacyEngine

import os
import gc
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Dataset, TensorDataset, Subset
import timm
from opacus.validators import ModuleValidator
from opacus.accountants.utils import get_noise_multiplier
from tqdm import tqdm
from statistics import mean
import warnings; warnings.filterwarnings("ignore")


def main(args):
    if args.clipping_mode not in ['nonDP', 'BK-ghost', 'BK-MixGhostClip', 'BK-MixOpt', 'nonDP-BiTFiT', 'BiTFiT']:
        print("Mode must be one of 'nonDP', 'BK-ghost', 'BK-MixGhostClip', 'BK-MixOpt', 'nonDP-BiTFiT', 'BiTFiT'")
        return None

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    ####### data ######
    print('==> Preparing data..')

    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.dimension),
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
         
    print('==> Splitting data..')
    
    # combine train and test sets
    combinedset = torch.utils.data.ConcatDataset([trainset, testset])
    
    # take the first 20000 data examples
    subset_indices = list(range(args.total_data_examples))
    combinedset = Subset(combinedset, subset_indices)

    half_examples = args.total_data_examples // 2

    # use random_split to split the dataset in half: half train, half test
    inset, outset = random_split(combinedset, [half_examples, args.total_data_examples - half_examples])

    # compute binary vector that records indices of train data
    # vector[i] = 1 means data example i is in train; vector[i] = 0 means example i is in test 
    in_indices = inset.indices
    out_indices = [i for i in range(args.total_data_examples) if i not in in_indices]

    binary_vector = np.zeros(args.total_data_examples)
    binary_vector[in_indices] = 1

    trainloader = torch.utils.data.DataLoader(
        inset, batch_size=args.mini_bs, shuffle=True, num_workers=4)

    testloader = torch.utils.data.DataLoader(
        outset, batch_size=100, shuffle=False, num_workers=4)
    
    # record binary vector of indices
    indices_dir = f'attack_data/indices_{args.data}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}'
    df_indices = pd.DataFrame({f'{args.experiment_no}': np.array(binary_vector)})
    if not os.path.exists(indices_dir):
        os.makedirs(indices_dir)
    df_indices.to_csv(f'{indices_dir}/{args.experiment_no}.csv', index=False)
    
    ####### training #######
    n_acc_steps = args.bs // args.mini_bs # gradient accumulation steps

    # build model
    print('==> Building model..', args.model,'; BatchNorm is replaced by GroupNorm. Mode: ', args.clipping_mode)
    net = timm.create_model(args.model,pretrained=True,num_classes=num_classes)    
    net=net.to(device)

    print('Number of total parameters: ', sum([p.numel() for p in net.parameters()]))
    print('Number of trainable parameters: ', sum([p.numel() for p in net.parameters() if p.requires_grad]))
    
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
        
    def train(epoch):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
   
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                optimizer.step()
                optimizer.zero_grad()
                
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Epoch: ', epoch, len(trainloader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # record train accuracy after the last epoch of training
        if epoch == args.epochs - 1:
            if not args.dry_run:
                accuracies_dir = f'attack_data/accuracies_train_{args.data}'
                if not os.path.exists(accuracies_dir):
                    os.makedirs(accuracies_dir)
                    print(f"Created directory: {accuracies_dir}")
                accuracies_file_path = f'{accuracies_dir}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}.txt'
                with open(accuracies_file_path, 'a') as file:
                    file.write(str(100.*correct/total) + '\n')
                    print(f"File updated with the integer value: {100.*correct/total}")

    def test(epoch):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print('Epoch: ', epoch, len(testloader), 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            # record test accuracy
            if epoch == args.epochs - 1:
                if not args.dry_run:
                    accuracies_dir = f'attack_data/accuracies_{args.data}'
                    if not os.path.exists(accuracies_dir):
                        os.makedirs(accuracies_dir)
                        print(f"Created directory: {accuracies_dir}")
                    accuracies_file_path = f'{accuracies_dir}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}.txt'
                    with open(accuracies_file_path, 'a') as file:
                        file.write(str(100.*correct/total) + '\n')
                        print(f"File updated with the integer value: {100.*correct/total}")

    for epoch in range(args.epochs):
        train(epoch)
        test(epoch)
        
    # save model state dict (for the next step of the pipeline: computing post-hoc explanations)
    if not args.dry_run:
        model_state_dict = net.state_dict()
        for key in list(model_state_dict.keys()):
            model_state_dict[key.replace('_module.', '')] = model_state_dict.pop(key)
        model_dir = f'attack_data/model_state_dicts_{args.data}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Created directory: {model_dir}")
        torch.save(model_state_dict, f'{model_dir}/{args.experiment_no}.pt')
    
    # garbage collection for good memory usage
    net.cpu()
    del net
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--epochs', default=5, type=int,
                        help='number of epochs')
    parser.add_argument('--bs', default=1000, type=int, help='batch size')
    parser.add_argument('--mini_bs', type=int, default=50)
    parser.add_argument('--clipping_mode', default='nonDP', type=str)
    parser.add_argument('--model', default='vit_small_patch16_224', type=str)
    parser.add_argument('--data', type=str, default='CIFAR10')
    parser.add_argument('--dimension', type=int,default=224)
    parser.add_argument('--origin_params', nargs='+', default=None)
    parser.add_argument('--dry_run', default=False)
    parser.add_argument('--experiment_no', type=int, default=1)
    parser.add_argument('--total_data_examples', type=int, default=20000)

    args = parser.parse_args()
    
    torch.manual_seed(args.experiment_no)

    main(args)
