''For a single attack parameter setting, run the Loss LiRA baseline attack and generate metrics and plots.'''

import argparse
import os
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython import display
import random
import scipy.stats
from sklearn import metrics
from sklearn.metrics import roc_curve

def get_data(args, valid_experiment_list):
    indices = np.empty((args.total_data_examples, len(valid_experiment_list)))
    for i, exp in enumerate(valid_experiment_list):
        if 'nonDP' not in args.clipping_mode:
            indices_dir = f"attack_data/indices_{args.data}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_epochs={args.epochs}"
        else:
            indices_dir = f"attack_data/indices_{args.data}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}"
        df = pd.read_csv(f"{indices_dir}/{exp}.csv")
        indices[:,i] = df[str(exp)][:args.total_data_examples]
    indices = indices.astype(bool)
    scores = np.empty((args.total_data_examples, args.num_experiments))

    for i, exp in enumerate(valid_experiment_list):
        if 'nonDP' not in args.clipping_mode:
            scores_dir = f"attack_data/losses_{args.data}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_epochs={args.epochs}"
        else:
            scores_dir = f"attack_data/losses_{args.data}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}"
        df = pd.read_csv(f"{scores_dir}/{exp}.csv")
        scores[:,i] = df[str(exp)][:args.total_data_examples]
    return indices, scores


def split_scores(scores, ind_mask_all, min_keep=3, attack_model_index=0):
    n_keep = min_keep - 1
    # overwrite this
    n_keep = n_keep - 1
    M = ind_mask_all.shape[1]
    # t0 data with arbitrary (idx to be attacked)
    mask_t0 =  ind_mask_all[:, attack_model_index]
    t0 = scores[:, attack_model_index]
    # remaining data
    other_inds = np.delete(range(M), attack_model_index).tolist()
    ind_mask_all = ind_mask_all[:,other_inds]
    scores = scores[:, other_inds]
    # compute ins and outs
    maskin = ind_mask_all * 1
    maskout = (1-ind_mask_all) * 1
    scores_in = scores * maskin
    scores_out = scores * maskout
    # make sure to have at least min_keep
    nzi = np.zeros(scores_in.shape[0], dtype=bool)
    non_zeros_in = np.where(np.sum(maskin, axis=1) > n_keep)[0]
    nzi[non_zeros_in] = True
    nzo = np.zeros(scores_out.shape[0], dtype=bool)
    non_zeros_out = np.where(np.sum(maskout, axis=1) > n_keep)[0]
    nzo[non_zeros_out] = True
    ind_keep = np.where(((nzi*1)*(nzo*1))!=0)[0]
    # update everything
    scores_in = scores_in[ind_keep]   # update scores in
    scores_out = scores_out[ind_keep] # update scores out
    t0 = t0[ind_keep]                 # update t0
    mask_t0 = mask_t0[ind_keep]       # update mask
    # compute averages
    normalizer_in = np.count_nonzero(scores_in, axis=1)
    normalizer_out = np.count_nonzero(scores_out, axis=1)
    in_scores = np.sum(scores_in, axis=1)/normalizer_in 
    out_scores = np.sum(scores_out, axis=1)/normalizer_out
    return in_scores, out_scores, t0, mask_t0

def get_lrt_scores(owner_scores, owner_indices, in_scores, out_scores, global_var):
    # compute lrt scores
    in_means = np.mean(in_scores) 
    out_means = np.mean(out_scores)
    in_stds = np.std(in_means)
    out_stds = np.std(out_means)
    if global_var:
        overall_std = np.std(np.array([in_scores, out_scores]))
        in_stds = overall_std
        out_stds = overall_std
    pr_in_lrt = -scipy.stats.norm.logpdf(owner_scores, in_scores, in_stds+1e-15)
    pr_out_lrt = -scipy.stats.norm.logpdf(owner_scores, out_scores, out_stds+1e-15)
    scores_lrt = pr_in_lrt - pr_out_lrt
    in_scores = scores_lrt[owner_indices]
    out_scores = scores_lrt[~owner_indices]
    return in_scores, out_scores

def compute_curve(in_scores: np.array, out_scores: np.array, pos_label=1):
    y = np.r_[np.ones(np.shape(in_scores)[0]), np.zeros(np.shape(out_scores)[0])]
    fs, ts, thresholds = roc_curve(y, np.r_[in_scores, out_scores], pos_label=pos_label)
    acc = np.max(1-(fs+(1-ts))/2)
    return ts, fs, metrics.auc(ts, fs), acc

def main(args):
    
    if args.model=='vit_relpos_small_patch16_224.sw_in1k':
        model_short = 'vit_relpos_small'
    elif args.model=='vit_small_patch16_224':
        model_short='vit_small'
    elif 'vit_relpos_base_patch16_224' in args.model:
        model_short='vit_relpos_small'
    elif 'beit_base_patch16_224' in args.model:
        model_short='beit_base'
    elif 'beitv2_base_patch16_224' in args.model:
        model_short='beitv2_base'
       
    results = {
    args.clipping_mode: {
        'Losses': {'tprs': [], 'aucs': []},
        }
    }

    mean_fpr = np.linspace(0, 1, 10000)

    tpr_001_dir = f'attack_data/tpr001_{args.data}_Losses'
    tpr_01_dir = f'attack_data/tpr01_{args.data}_Losses'
    auc_dir = f'attack_data/auc_{args.data}_Losses'
    if not os.path.exists(tpr_001_dir):
        os.makedirs(tpr_001_dir)
        print(f"Created directory: {tpr_001_dir}")
    if not os.path.exists(tpr_01_dir):
        os.makedirs(tpr_01_dir)
        print(f"Created directory: {tpr_01_dir}")
    if not os.path.exists(auc_dir):
        os.makedirs(auc_dir)
        print(f"Created directory: {auc_dir}")
    tpr_001_file_path = f'{tpr_001_dir}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}.txt'
    tpr_01_file_path = f'{tpr_01_dir}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}.txt'
    auc_file_path = f'{auc_dir}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}.txt'
    with open(tpr_001_file_path, 'w') as file:
        pass
    with open(tpr_01_file_path, 'w') as file:
        pass
    with open(auc_file_path, 'w') as file:
        pass

    valid_experiment_list = []
    for attack_model_index in range(args.num_experiments):
        if 'nonDP' not in args.clipping_mode:
            lira_dir = f"attack_data/losses_{args.data}/model={args.model}_mode={args.clipping_mode}_eps={args.epsilon}_epochs={args.epochs}"
        else:
            lira_dir = f"attack_data/losses_{args.data}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}"
        if os.path.exists(f"{lira_dir}/{attack_model_index}.csv"):
            valid_experiment_list.append(attack_model_index)
    print(valid_experiment_list)

    for attack_model_index, _ in tqdm(enumerate(valid_experiment_list)):

        tpr_001_dir = f'attack_data/tpr001_{args.data}_Losses'
        tpr_01_dir = f'attack_data/tpr01_{args.data}_Losses'
        tpr_001_file_path = f'{tpr_001_dir}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}.txt'
        tpr_01_file_path = f'{tpr_01_dir}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}.txt'
        auc_dir = f'attack_data/auc_{args.data}_Losses'
        auc_file_path = f'{auc_dir}/model={args.model}_mode={args.clipping_mode}_epochs={args.epochs}.txt'

        indices, scores = get_data(args, valid_experiment_list)

        in_scores, out_scores, t0, mask_t0 = split_scores(scores, indices, min_keep=3, attack_model_index=attack_model_index)

        in_scores, out_scores = get_lrt_scores(t0, mask_t0, in_scores, out_scores, global_var=True)

        num_rows = int(np.ceil(args.num_experiments / 4))
        num_cols = 4

        fpr, tpr, auc, acc = compute_curve(in_scores, out_scores)
        tpr = np.interp(mean_fpr, fpr, tpr)
        tpr[0] = 0.0
        low_001 = tpr[np.where(fpr<=.001)[0][-1]]
        low_01 = tpr[np.where(fpr<=.01)[0][-1]]
        with open(tpr_001_file_path, 'a') as file:
            file.write(str(low_001) + '\n')
        with open(tpr_01_file_path, 'a') as file:
            file.write(str(low_01) + '\n')
        with open(auc_file_path, 'a') as file:
            file.write(str(auc) + '\n')

        results[args.clipping_mode]['Losses']['tprs'].append(tpr)
        results[args.clipping_mode]['Losses']['aucs'].append(auc)   

    plt.figure()
    tprs = np.array(results[args.clipping_mode]['Losses']['tprs']).T
    aucs = np.array(results[args.clipping_mode]['Losses']['aucs']).T
    mean_tpr = np.mean(tprs, axis=1)
    std_tpr = np.std(tprs, axis=1)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs).round(2)
    std_auc = np.std(aucs).round(2)

    plt.loglog(mean_fpr, mean_tpr, label=f"AUC = %0.2f $\pm$ %0.2f" % (mean_auc, std_auc), lw=2, alpha=0.8)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=0.2)
    title = f'Losses LiRA, {args.data}, {model_short}'
    plt.plot([0,1], [0,1], linestyle='dotted', color='black')
    plt.legend(loc=4, fontsize=12)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.xlim([0.001, 1])
    plt.ylim([0.001, 1])
    plt.title(title, fontsize=14)
    plt.tight_layout()
    if args.save_fig:
        plt.savefig(f'plots/{args.data}/{title}.png')
    plt.show()

    #### plotting non-Log scaled curves ####
    plt.figure()
    tprs = np.array(results[args.clipping_mode]['Losses']['tprs']).T
    aucs = np.array(results[args.clipping_mode]['Losses']['aucs']).T
    mean_tpr = np.mean(tprs, axis=1)
    std_tpr = np.std(tprs, axis=1)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs).round(2)
    std_auc = np.std(aucs).round(2)

    plt.plot(mean_fpr, mean_tpr, label=f"AUC = %0.2f $\pm$ %0.2f" % (mean_auc, std_auc), lw=2, alpha=0.8)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=0.2) #label=r"$\pm$ 1 std. dev.")

    title = f'Losses LiRA, {args.data}, Non-Log, {model_short}'
    plt.plot([0,1], [0,1], linestyle='dotted', color='black')
    plt.legend(loc=4)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.xlim([0.001, 1])
    plt.ylim([0.001, 1])
    plt.title(title)
    plt.tight_layout()
    if args.save_fig:
        if not os.path.exists(f'plots/{args.data}'):
            os.makedirs(f'plots/{args.data}')
            print(f'Created directory: plots/{args.data}')
        plt.savefig(f'plots/{args.data}/{title}.png')
    plt.show()

         
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='CIFAR10')
    parser.add_argument('--epochs', default=30, type=int,
                        help='number of epochs')
    parser.add_argument('--num_experiments', default=33, type=int,
                        help='number of experiments')
    parser.add_argument('--epsilon', default=2.0, type=float, help='target epsilon')
    parser.add_argument('--clipping_mode', default='nonDP', type=str)
    parser.add_argument('--model', default='vit_small_patch16_224', type=str) # try: resnet18
    parser.add_argument('--explanation_type', default='ixg', choices=['gs', 'ig', 'ixg', 'lrp', 'sl', 'gb'])
    parser.add_argument('--save_fig', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--total_data_examples', type=int, default=20000)
    args = parser.parse_args()
    main(args)