import torch
import torch.nn as nn

from functools import partial
from spikingjelly.activation_based import layer

import base
from neuron import LIFNODE, PLIFNode, GatedLIFNode

class adskip_RecurrentContainer(base.MemoryModule):
    def __init__(self, sub_module: nn.Module, step_mode='s', hid_dim=64, skip=21, skip_para=None, de_tau=0.96):
        super().__init__()
        assert not hasattr(sub_module, 'step_mode') or sub_module.step_mode == 's'

        self.step_mode = step_mode
        self.sub_module = sub_module
        self.skip = skip    # skip span
        self.tau = torch.nn.Parameter(torch.ones(1), requires_grad=False)   # temperature parameter
        self.de_tau = de_tau    # decay factor for an exponential decay strategy

        if skip_para is not None:   # for parameter layer shared
            self.skip_para = skip_para
        else:
            self.skip_para = torch.nn.Parameter(torch.zeros(self.skip, 1, 1), requires_grad=True)
        self.mlp = nn.Linear(hid_dim, hid_dim, bias = False)    # rnn parameter
        self.register_memory('y', None)

        torch.nn.init.orthogonal_(
            self.mlp.weight,
            gain=1.0,
        )

    def single_step_forward(self, x: torch.Tensor):
        if self.y is None:
            self.y = torch.zeros(self.skip, x.shape[0], x.shape[1]).to(x.device)

        s_sum = None
        if self.training:
            s_sum = (torch.nn.functional.softmax(self.skip_para / self.tau, dim = 0) * self.y).sum(0)
        elif not self.training:
            max_index = torch.argmax(self.skip_para)
            s_sum = self.y[max_index]
        
        spike = self.sub_module(self.mlp(s_sum) + x)

        self.y = torch.roll(self.y, 1, 0)
        self.y[0] = spike

        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[1]
        y_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[:,t,:])
            y_seq.append(y)

        return torch.stack(y_seq).permute(1, 0, 2)

    def forward(self, x: torch.Tensor):
        if self.step_mode == 's':
            return self.single_step_forward(x)
        else:
            return self.multi_step_forward(x)
        
    def decrease_tau(self):
        self.tau  *=  self.de_tau

class skip_RecurrentContainer(base.MemoryModule):
    def __init__(self, sub_module: nn.Module, step_mode='s', hid_dim=64, skip = 21):
        super().__init__()
        assert not hasattr(sub_module, 'step_mode') or sub_module.step_mode == 's'

        self.step_mode = step_mode
        self.sub_module = sub_module
        self.skip = skip    # skip span

        self.mlp = nn.Linear(hid_dim, hid_dim, bias = False)    # rnn parameter
        self.register_memory('y', None)

        torch.nn.init.orthogonal_(
            self.mlp.weight,
            gain=1.0,
        )

    def single_step_forward(self, x: torch.Tensor):
        if self.y is None:
            self.y = torch.zeros(self.skip, x.shape[0], x.shape[1]).to(x.device)

        spike = self.sub_module(self.mlp(self.y[-1]) + x)

        self.y = torch.roll(self.y, 1, 0)
        self.y[0] = spike

        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[1]
        y_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[:,t,:])
            y_seq.append(y)

        return torch.stack(y_seq).permute(1, 0, 2)

    def forward(self, x: torch.Tensor):
        if self.step_mode == 's':
            return self.single_step_forward(x)
        else:
            return self.multi_step_forward(x)
        
def generate_neuron(args):
    if args.sg == 'exp':
        from surrogate import SingleExponential as SG
    elif args.sg == 'triangle':
        from surrogate import Triangle as SG
    elif args.sg == 'rectangle':
        from surrogate import Rectangle as SG
    elif args.sg == 'sigmoid':
        from surrogate import sigmoid as SG
    elif args.sg == 'gau':
        from surrogate import ActFun_adp as SG
    else:
        raise NotImplementedError

    node = None
    if args.neuron == 'lif':
        node = LIFNODE
    elif args.neuron == 'plif':
        node = PLIFNode
    elif args.neuron == 'glif':
        node = GatedLIFNode
    else:
        raise NotImplementedError
    
    if args.neuron == 'glif':
        spiking_neuron = partial(node,
                            surrogate_function=SG.apply,
                            T = args.time_window
                            )  
    else:
        spiking_neuron = partial(node,
                                v_threshold=args.threshold,
                                surrogate_function=SG.apply,
                                hard_reset=args.hard_reset,
                                detach_reset=args.detach_reset,
                                decay_para=args.decay_para,
                                )   
    return spiking_neuron

class MODEL(nn.Module):
    def __init__(self, args):
        super().__init__()
        assert args.model_version == 'SRC' or args.model_version == 'ASRC', "no this model version"
        assert len(args.layers) == len(args.skip), "layers and skip should match"
        spiking_neuron = generate_neuron(args)
        
        self.skip_para = None
        #for parameter layer shared
        if args.network_wise and args.model_version == 'ASRC':  
            self.skip_para = torch.nn.Parameter(torch.zeros(args.skip[0], 1, 1), requires_grad=True)

        layers = []
        temp_dim = args.input_dim

        for index, hidden in enumerate(args.layers):
            layers.append(nn.Linear(temp_dim, hidden,bias=args.bias))
            temp_dim = hidden

            layers.append(layer.Dropout(args.drop, step_mode='s'))

            if args.use_layernorm:
                layers.append(torch.nn.LayerNorm(hidden))

            neuron_channel = hidden if args.neuron_channel_wise else 1
            if args.model_version == 'SRC':
                layers.append(skip_RecurrentContainer(sub_module = spiking_neuron(channel=neuron_channel), hid_dim = hidden, skip = args.skip[index]))
            elif args.model_version == 'ASRC':
                layers.append(adskip_RecurrentContainer(sub_module = spiking_neuron(channel=neuron_channel), hid_dim = hidden, skip = args.skip[index], 
                                                                        de_tau = args.de_tau, skip_para = self.skip_para))

        layers.append(nn.Linear(temp_dim, args.output_dim, bias=args.bias))

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 3, "dimension of x is not correct!"  # x: [batch size, sequenctial length, dim]

        output_current = []
        for t in range(x.size(1)):  # T loop
            output_current.append(self.features(x[:, t ,:]))
            
        res = torch.stack(output_current, 0)
        return res.sum(0)
    
    def decrease_tau(self,):
        for m in self.features:
            if isinstance(m, adskip_RecurrentContainer):
                m.decrease_tau()

class BatchNorm1d(nn.Module):
    def __init__(self, hidden):
        
        super().__init__()
        self.hidden = hidden
        self.bn = nn.BatchNorm1d(hidden)
    def forward(self, x_seq):
        s1, s2 = x_seq.shape[0], x_seq.shape[1]
        x_seq = x_seq.view(s1 * s2, -1)
        res = self.bn(x_seq).view(s1, s2, -1)
        return res
        
class MODEL_BN(nn.Module):
    def __init__(self, args):
        super().__init__()
        assert args.model_version == 'SRC' or args.model_version == 'ASRC', "no this model version"
        assert len(args.layers) == len(args.skip), "layers and skip should match"
        spiking_neuron = generate_neuron(args)
        
        self.skip_para = None
        #for parameter layer shared
        if args.network_wise and args.model_version == 'ASRC':
            self.skip_para = torch.nn.Parameter(torch.zeros(args.skip[0], 1, 1), requires_grad=True)

        layers = []
        temp_dim = args.input_dim

        for index, hidden in enumerate(args.layers):
            layers.append(nn.Linear(temp_dim, hidden, bias=args.bias))
            temp_dim = hidden

            layers.append(layer.Dropout(args.drop, step_mode='m'))

            layers.append(BatchNorm1d(hidden))
                
            neuron_channel = hidden if args.neuron_channel_wise else 1
            if args.model_version == 'SRC':
                layers.append(skip_RecurrentContainer(sub_module = spiking_neuron(channel = neuron_channel), hid_dim = hidden, 
                                                      skip = args.skip[index], step_mode = 'm'))
            elif args.model_version == 'ASRC':
                layers.append(adskip_RecurrentContainer(sub_module = spiking_neuron(channel = neuron_channel), hid_dim = hidden, 
                                    skip = args.skip[index], de_tau = args.de_tau, skip_para = self.skip_para, step_mode = 'm'))

        layers.append(nn.Linear(temp_dim, args.output_dim, bias=args.bias))

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 3, "dimension of x is not correct!"  # x: [batch size, sequenctial length, dim]
           
        res = self.features(x)

        return res.sum(1)
    
    def decrease_tau(self,):
        for m in self.features:
            if isinstance(m, adskip_RecurrentContainer):
                m.decrease_tau()
