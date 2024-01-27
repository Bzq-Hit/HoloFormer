import torch
import torch.nn as nn
import os
from collections import OrderedDict

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr



def get_arch(opt):
    from model import HoloFormer

    arch = opt.arch

    if arch == 'HoloFormer':
        model_restoration = HoloFormer(embed_dim=32,win_size=8,token_projection='linear',token_mlp=opt.token_mlp, 
                                    depths=[1, 2, 8, 8, 2, 8, 8, 2, 1]
                                    )
    elif arch == 'HoloFormer_S':
        model_restoration = HoloFormer(embed_dim=32,win_size=8,token_projection='linear',token_mlp=opt.token_mlp, 
                                    depths=[2, 2, 2, 2, 2, 2, 2, 2, 2]
                                    )
    elif arch == 'HoloFormer_T':
        model_restoration = HoloFormer(embed_dim=16,win_size=8,token_projection='linear',token_mlp=opt.token_mlp, 
                                    depths=[2, 2, 2, 2, 2, 2, 2, 2, 2]
                                    )
    else:
        raise Exception("Arch error!")

    return model_restoration

