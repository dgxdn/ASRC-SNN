import torch
import numpy as np
import random
import os
import sys
import time
from prettytable import PrettyTable

import torch.nn as nn
from model import adskip_RecurrentContainer, skip_RecurrentContainer
from neuron import PLIFNode, LIFNODE
import base

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def reset_states(model):
    for m in model.modules():
        if hasattr(m, 'reset'):
            m.reset()

def seed_everything(seed=0, is_cuda=False):
    """Some configurations for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if is_cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def visualize_grad_norms(model, writer, epoch, args):
    for index, m in enumerate(model.features):
        if isinstance(m, nn.Linear):
            writer.add_histogram(f'gradients/layer{index}_linear_weight', m.weight.grad, epoch)
            if args.bias:
                writer.add_histogram(f'gradients/layer{index}_linear_bias', m.bias.grad, epoch)
        elif isinstance(m, adskip_RecurrentContainer):
            writer.add_histogram(f'gradients/layer{index}_mlp_weight', m.mlp.weight.grad, epoch)
            writer.add_histogram(f'gradients/layer{index}_skip_para', m.skip_para.grad, epoch)
            if args.neuron == 'plif':
                writer.add_histogram(f'gradients/layer{index}_plif_para', m.sub_module.w.grad, epoch)
        elif isinstance(m, skip_RecurrentContainer):
            if args.neuron == 'plif':
                writer.add_histogram(f'gradients/layer{index}_plif_para', m.sub_module.w.grad, epoch)
        elif isinstance(m, PLIFNode):
            writer.add_histogram(f'gradients/layer{index}_plif_para', m.w.grad, epoch)

def visualize_weight_norms(model, writer, epoch, args):    
    for index, m in enumerate(model.features):
        if isinstance(m, nn.Linear):
            writer.add_histogram(f'weights/layer{index}_linear_weight', m.weight, epoch)
            if args.bias:
                writer.add_histogram(f'weights/layer{index}_linear_bias', m.bias, epoch)
        elif isinstance(m, adskip_RecurrentContainer):
            writer.add_histogram(f'weights/layer{index}_mlp_weight', m.mlp.weight, epoch)
            writer.add_histogram(f'weights/layer{index}_skip_para', m.skip_para, epoch)
            if args.neuron == 'plif':
                writer.add_histogram(f'weights/layer{index}_plif_para', m.sub_module.w, epoch)
        elif isinstance(m, skip_RecurrentContainer):
            if args.neuron == 'plif':
                writer.add_histogram(f'weights/layer{index}_plif_para', m.sub_module.w, epoch)
        elif isinstance(m, PLIFNode):
            writer.add_histogram(f'weights/layer{index}_plif_para', m.w, epoch)

def log_tau(model, writer, epoch):
    tau = []
    for m in model.features:
        if isinstance(m, adskip_RecurrentContainer):
            tau.append(m.tau)

    for i in range(len(tau)):
            writer.add_scalar(f'tau / layer_{i}', tau[i], epoch)
