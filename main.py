import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from load_dataset import load_dataset
from model import MODEL, MODEL_BN, adskip_RecurrentContainer, skip_RecurrentContainer, BatchNorm1d
from neuron import PLIFNode, LIFNODE
from utils import *

def train(train_loader, model, criterion, optimizer, epoch, args,  writer, perm):
    train_loss = 0
    correct    = 0
    total      = 0

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        if args.task == 'PSMNIST' or args.task == 'SMNIST':
            inputs = inputs.view(-1, args.time_window, args.input_dim)  # input_im:[bs, 784, 1]
            if args.task == 'PSMNIST':
                inputs = inputs[:, perm, :]

        reset_states(model=model)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # compute gradient
        for opt in optimizer: opt.zero_grad()
        loss.backward()
        for opt in optimizer: opt.step()

        progress_bar(
            batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total)
            )
        
    avg_loss = train_loss / len(train_loader)
    avg_acc = 100. * correct / total
    writer.add_scalar('loss / train', avg_loss, epoch)
    writer.add_scalar('acc / train', avg_acc, epoch)
    visualize_grad_norms(model, writer, epoch, args)
    visualize_weight_norms(model, writer, epoch, args)
    log_tau(model, writer, epoch)
    #visualize skip
    if args.model_version == 'v2':
        cnt = 4 if (args.use_batchnorm or args.use_layernorm) else 3
        for i in range(len(args.layers)):
            temp = torch.nn.functional.softmax(model.features[cnt * (i + 1) - 1].skip_para / model.features[cnt * (i + 1) - 1].tau, dim = 0)
            writer.add_scalars('skip_weight / layer' + str(i), {f'p{j}': temp[j][0][0] for j in range(args.skip[i])}, epoch)  

    return avg_acc, avg_loss

def test(test_loader, model, criterion, args, writer, perm):
    test_loss = 0
    correct   = 0
    total     = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            if args.task == 'PSMNIST' or args.task == 'SMNIST':
                inputs = inputs.view(-1, args.time_window, args.input_dim)  # input_im:[bs, 784, 1]
                if args.task == 'PSMNIST':
                    inputs = inputs[:, perm, :]

            reset_states(model=model)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total)
            )
    
    avg_loss = test_loss / len(test_loader)
    avg_acc = 100. * correct / total
    writer.add_scalar('loss / test', avg_loss, epoch)
    writer.add_scalar('acc / test', avg_acc, epoch)

    return avg_acc, avg_loss

def init_optim_sche(model, args):
    #init optimizer and shceduler
    weights_skip = []
    weights_neuron = []
    weights_norm = []
    weights_other = []

    if args.model_version == 'ASRC' and args.network_wise:
        weights_skip.append(model.skip_para)

    for m in model.features:
        if isinstance(m, nn.Linear):
            weights_other.append(m.weight)
            if args.bias:
                weights_other.append(m.bias)
        elif isinstance(m, skip_RecurrentContainer):
            weights_other.append(m.mlp.weight)
            if args.neuron == 'plif':
                weights_neuron.append(m.sub_module.w)
            elif args.neuron == 'glif':
                weights_neuron.append(m.sub_module.alpha)
                weights_neuron.append(m.sub_module.beta)
                weights_neuron.append(m.sub_module.gamma)
                weights_neuron.append(m.sub_module.v_threshold)
                weights_neuron.append(m.sub_module.linear_decay)
                weights_neuron.append(m.sub_module.v_subreset)
                weights_neuron.append(m.sub_module.conduct)
        elif isinstance(m, adskip_RecurrentContainer):
            weights_other.append(m.mlp.weight)
            if args.neuron == 'plif':
                weights_neuron.append(m.sub_module.w)
            elif args.neuron == 'glif':
                weights_neuron.append(m.sub_module.alpha)
                weights_neuron.append(m.sub_module.beta)
                weights_neuron.append(m.sub_module.gamma)
                weights_neuron.append(m.sub_module.v_threshold)
                weights_neuron.append(m.sub_module.linear_decay)
                weights_neuron.append(m.sub_module.v_subreset)
                weights_neuron.append(m.sub_module.conduct)
            if not (args.model_version == 'ASRC' and args.network_wise):
                weights_skip.append(m.skip_para)
        elif isinstance(m, BatchNorm1d):
            weights_norm.append(m.bn.weight)
            weights_norm.append(m.bn.bias)
        elif isinstance(m, torch.nn.LayerNorm):
            weights_norm.append(m.weight)
            weights_norm.append(m.bias)

    optimizer = []
    scheduler = []

    if args.optim == 'sgd':
        optimizer .append(torch.optim.SGD([{'params':weights_other, 'lr':args.lr, 'weight_decay':args.wd},
                                           {'params':weights_norm, 'lr':args.lr, 'weight_decay':0},]))
        optimizer .append(torch.optim.SGD([{'params':weights_neuron, 'lr':args.lr_neuron, 'weight_decay':0.}]))
        optimizer .append(torch.optim.SGD([{'params':weights_skip, 'lr':args.lr_skip, 'weight_decay':0.}]))
    elif args.optim == 'adam':
        optimizer.append(torch.optim.Adam([{'params':weights_other, 'lr':args.lr, 'weight_decay':args.wd},
                                           {'params':weights_norm, 'lr':args.lr, 'weight_decay':0},]))
        optimizer.append(torch.optim.Adam([{'params':weights_neuron, 'lr':args.lr_neuron, 'weight_decay':0}]))
        optimizer.append(torch.optim.Adam([{'params':weights_skip, 'lr':args.lr_skip, 'weight_decay':0}]))
    elif args.optim == 'adamw':
        optimizer.append(torch.optim.AdamW([{'params':weights_other, 'lr':args.lr, 'weight_decay':args.wd},
                                           {'params':weights_norm, 'lr':args.lr, 'weight_decay':0},]))
        optimizer.append(torch.optim.AdamW([{'params':weights_neuron, 'lr':args.lr_neuron, 'weight_decay':0}]))
        optimizer.append(torch.optim.AdamW([{'params':weights_skip, 'lr':args.lr_skip, 'weight_decay':0}]))
    else:
        raise NotImplementedError

    if args.scheduler == 'cos':
        scheduler.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[0], T_max=args.epochs))
        scheduler.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[1], T_max=args.epochs))    
        scheduler.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[2], T_max=args.epochs))
    elif args.scheduler == 'onecycle':
        scheduler.append(torch.optim.lr_scheduler.OneCycleLR(optimizer[0], max_lr=args.lr, total_steps=args.epochs))
        scheduler.append(torch.optim.lr_scheduler.OneCycleLR(optimizer[1], max_lr=args.lr_neuron,total_steps=args.epochs))
        scheduler.append(torch.optim.lr_scheduler.OneCycleLR(optimizer[2], max_lr=args.lr_skip, total_steps=args.epochs))
    else:
        raise NotImplementedError
    
    return optimizer, scheduler

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int, metavar='N', help='seed')
parser.add_argument('--task', default='SMNIST', type=str, help='SMNIST, PSMNIST, GSC, SSC')
parser.add_argument('--dataset-dir', default='./datasets', type=str, metavar='PATH', help='data path')
parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')
parser.add_argument("--num-workers", type=int, default=8, help='for dataloader')
parser.add_argument("--n-bins", type=int, default=5, help='for binning in SSC')
parser.add_argument('--device', default='cuda:0', type=str, help='device')
# training setting
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='max learning rate',
                    dest='lr')
parser.add_argument('--lr-neuron', default=0.0001, type=float, help='max learning rate for neuron paramerter')
parser.add_argument('--lr-skip', default=0.1, type=float, help='max learning rate for skip connections paramerter')
parser.add_argument('--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--optim', default='adam', type=str, help='optimizer (default: adam)') 
parser.add_argument('--scheduler', default='onecycle', type=str, help='scheduler: cos, onecycle')
# options for nueron
parser.add_argument('--neuron', default='lif', type=str, help='neuron: lif, plif, glif')
parser.add_argument('--neuron-channel-wise', action='store_true', default=False, help='paramerter of neuron use channel-wise')
parser.add_argument('--threshold', default=1.0, type=float, help='')
parser.add_argument('--detach-reset', action='store_true', default=False, help='')
parser.add_argument('--hard-reset', action='store_true', default=False, help='')
parser.add_argument('--decay-para', default=0.5, type=float, help='decay factor for lif, plif')
parser.add_argument('--sg', default='triangle', type=str, help='sg: triangle, exp, gau, rectangle and sigmoid')
# options for SNNs
parser.add_argument('--model-version', default='SRC', type=str, help='SRC, ASRC')
parser.add_argument('--layers', default=[64, 128, 128] , type=int, nargs='+', help='model layer hidden')
parser.add_argument('--bias', action='store_true', default=False, help='for feedforward mlp')
parser.add_argument('--drop', default=0., type=float, help='drop rate')
parser.add_argument('--use-batchnorm', action='store_true', default=False, help='')
parser.add_argument('--use-layernorm', action='store_true', default=False, help='')
parser.add_argument('--input-dim', default=1, type=int, help='auto set for task')
parser.add_argument('--output-dim', default=1, type=int, help='auto set for task')
#options for model SRC, ASRC
parser.add_argument('--de-tau', default=0.96, type=float, help='decrease tau every epoch for tau in ASRC')
parser.add_argument('--skip', default=[21, 21, 21], type=int, nargs='+', help='skip coefficient for SRC-SNN; max skip coefficient for ASRC-SNN')
parser.add_argument('--network-wise', action='store_true', default=False, help='In ASRC-SNN, skip paramerters shared for all layes')
#other
parser.add_argument('--time-window', default=784, type=int, help='auto set for task')
parser.add_argument('--total-params', default=0, type=int, help='parameter number')

args = parser.parse_args()
seed_everything(seed=args.seed, is_cuda=True)

if not torch.cuda.is_available():
    args.device = 'cpu'
    print('GPU is not available')

# init dataloaders
train_loader, test_loader = load_dataset(args)
# init model
model = None
if args.use_batchnorm:
    model = MODEL_BN(args).to(args.device)
else:
    model = MODEL(args).to(args.device)
# init optimizer and scheduler
optimizer, scheduler = init_optim_sche(model, args)
# calculate the number of parameters
args.total_params = count_parameters(model)
# for logs
if args.results_dir == '':
    args.results_dir = './exp/' + args.task + '/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
Path(args.results_dir).mkdir(parents=True, exist_ok=True)

with open(args.results_dir + '/args.json', 'w') as fid:
    json.dump(args.__dict__, fid, indent=2)

Path(args.results_dir + '/logs').mkdir(parents=True, exist_ok=True) 
writer = SummaryWriter(args.results_dir + '/logs')

if args.task == 'PSMNIST':
    perm = torch.randperm(784)
else:
    perm = None

# For storing results
train_res = pd.DataFrame()
test_res = pd.DataFrame()
best_acc = 0
criterion = nn.CrossEntropyLoss().to(args.device)

for epoch in range(args.epochs):
    # shuffle
    if args.task == 'SSC':
        train_loader.reset()
        test_loader.reset()

    train_acc, tarin_loss = train(train_loader, model, criterion, optimizer, epoch, args, writer, perm)
    test_acc, test_loss = test(test_loader, model, criterion, args, writer, perm = perm)

    for sc in scheduler: sc.step()

    model.decrease_tau()
    #for logs
    train_res[str(epoch)] = [train_acc, tarin_loss]
    test_res[str(epoch)] = [test_acc, test_loss]

    train_res.to_csv(os.path.join(args.results_dir, 'train_res.csv'), index=True)
    test_res.to_csv(os.path.join(args.results_dir, 'test_res.csv'), index=True)

    state = {'net': model.state_dict(), 'acc': test_acc, 'epoch': epoch,}
    torch.save(state, os.path.join(args.results_dir, 'last.pth'))

    if test_acc >= best_acc:
        torch.save(state, os.path.join(args.results_dir, 'best.pth'))
        best_acc = test_acc

    print('Test Epoch: [{}/{}], lr: {:.6f},  lr_skip: {:.6f}, acc: {:.4f}, best: {:.4f}'.format(epoch, args.epochs,
                                                                                     optimizer[0].param_groups[0]['lr'],
                                                                                     optimizer[2].param_groups[0]['lr'],
                                                                                     test_acc, best_acc))
