"""python3 train.py --nepoch 500 --warmup --w_loss_contrast 5 --arch HoloFormer"""
"""python3 train.py --nepoch 500 --warmup --w_loss_contrast 5 --arch HoloFormer_S"""
"""python3 train.py --nepoch 500 --warmup --w_loss_contrast 5 --arch HoloFormer_T"""

import os
import sys

from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import make_grid

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'..'))

import argparse
import options
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='Lensless restore')).parse_args()
print(opt)

######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx

import utils
from losses import CharbonnierLoss
from losses import MSELoss
from CR import *

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

import dataloader


######### Logs dir ###########
log_dir = os.path.join(opt.save_dir, 'lensless', opt.dataset, opt.arch+opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
print("Now time is : ",datetime.datetime.now().isoformat())
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
logname = os.path.join(log_dir, now, now +'.txt') 
model_dir  = os.path.join(log_dir, now, 'models')
board_dir = os.path.join(log_dir, now, 'board')
utils.mkdir(model_dir)
utils.mkdir(board_dir)

########## Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Model ###########
model_restoration = utils.get_arch(opt)

with open(logname,'a') as f:
    f.write(str(opt)+'\n') 
    f.write(str(model_restoration)+'\n') 

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")

######### DataParallel ########### 
model_restoration = torch.nn.DataParallel (model_restoration) 
model_restoration.cuda() 

######### Scheduler ###########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    #scheduler.step()
else:
    step = opt.step_lr
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    #scheduler.step()

######### Resume ########### 
if opt.resume: 
    path_chk_rest = opt.pretrain_weights 
    print("Resume from "+path_chk_rest)
    utils.load_checkpoint(model_restoration,path_chk_rest) 
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1 
    lr = utils.load_optim(optimizer, path_chk_rest) 

    # for p in optimizer.param_groups: p['lr'] = lr 
    # warmup = False 
    # new_lr = lr 
    # print('------------------------------------------------------------------------------') 
    # print("==> Resuming Training with learning rate:",new_lr) 
    # print('------------------------------------------------------------------------------') 
    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6) 

######### Loss ###########
c_loss = CharbonnierLoss().cuda()

# add Contrast regularization
contrast_loss = ContrastLoss(ablation=opt.is_ab)


######### DataLoader ###########
print('===> Loading datasets')

# load real exp data real and img part with augmentation
opt.data_dir = '' # add your data dir

train_dataset = dataloader.get_training_data(opt.data_dir) 
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, 
        num_workers=opt.train_workers, pin_memory=False, drop_last=True)
val_dataset = dataloader.get_validation_data(opt.data_dir) 
val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False, 
        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)
with open(logname, 'a') as f:
    f.write("Sizeof training set: " + str(len_trainset) + ", sizeof validation set: " + str(len_valset))

######### Tensorboard ###########
writer = SummaryWriter(log_dir=board_dir)

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
eval_now = len(train_loader)//4
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

loss_scaler = NativeScaler()

save_index = 0 # used in tensorboard saving -- train_batch_loss
save_index_1 = 0 # used in tensorboard saving -- val_batch_loss

torch.cuda.empty_cache()
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    eval_count = 0
    psnr_train_rgb = []
    ssim_train_rgb = []

    for i, (ob_real, ob_imag, gt_real, gt_imag) in enumerate(tqdm(train_loader)): 
        # zero_grad
        optimizer.zero_grad()

        gt_real = torch.unsqueeze(gt_real, dim=1)
        gt_imag = torch.unsqueeze(gt_imag, dim=1)
        target = torch.cat((gt_real, gt_imag), dim=1)
        target = target.cuda()

        ob_real = torch.unsqueeze(ob_real, dim=1)
        ob_imag = torch.unsqueeze(ob_imag, dim=1)
        ft = torch.cat((ob_real, ob_imag), dim=1)
        input_ = ft.cuda()

        with torch.cuda.amp.autocast():
            restored = model_restoration(input_.to(torch.float32))

            # loss
            c_loss_ = c_loss(restored, target)
            contrast_loss_ = contrast_loss(restored, target.to(torch.float32), input_.to(torch.float32)) # anchor, positive, negative
            
            loss = opt.w_loss_1st * c_loss_ + opt.w_loss_contrast * contrast_loss_
        loss_scaler(
                loss, optimizer,parameters=model_restoration.parameters())
        
        writer.add_scalar('loss/train_batch_total_loss', loss.item(), save_index+1) 
        writer.add_scalar('loss/train_batch_c_loss', c_loss_.item(), save_index+1)
        writer.add_scalar('loss/train_batch_contrast_loss', contrast_loss_.item(), save_index+1)

        save_index += 1
        epoch_loss +=loss.item()

        # psnr&ssim cal in trainset
        with torch.no_grad():

            restored_norm = (restored - restored.min()) / (restored.max() - restored.min())
            target_norm = (target - target.min()) / (target.max() - target.min())

            restored_norm = torch.clamp(restored_norm, 0, 1)
            target_norm = torch.clamp(target_norm,0,1)

            psnr_train_rgb.append(utils.batch_PSNR(restored_norm, target_norm, False).item())
            ssim_train_rgb.append(utils.batch_SSIM(restored_norm, target_norm, False).item())

        #### Evaluation ####
        if (i+1)%eval_now==0 and i>0:
            eval_count += 1
            with torch.no_grad():
                model_restoration.eval()

                # psnr&ssim cal in valset
                psnr_val_rgb = []
                ssim_val_rgb = []
                for ii, (ob_real_val, ob_imag_val, gt_real_val, gt_imag_val) in enumerate((val_loader)):

                    gt_real_val = torch.unsqueeze(gt_real_val, dim=1)
                    gt_imag_val = torch.unsqueeze(gt_imag_val, dim=1)
                    target = torch.cat((gt_real_val, gt_imag_val), dim=1)
                    target = target.cuda()

                    ob_real_val = torch.unsqueeze(ob_real_val, dim=1)
                    ob_imag_val = torch.unsqueeze(ob_imag_val, dim=1)
                    ft_val = torch.cat((ob_real_val, ob_imag_val), dim=1)
                    input_ = ft_val.cuda()

                    with torch.cuda.amp.autocast():
                        restored = model_restoration(input_.to(torch.float32)) 
                    
                        # loss
                        loss_val_batch_contrast_loss = contrast_loss(restored, target.to(torch.float32), input_.to(torch.float32)) # anchor, positive, negative
                        loss_val_batch_c = c_loss(restored, target)
                        loss_val = opt.w_loss_1st * loss_val_batch_c + opt.w_loss_contrast * loss_val_batch_contrast_loss

                    writer.add_scalar('loss/val_batch_c_loss', loss_val_batch_c.item(), save_index_1+1) 
                    writer.add_scalar('loss/val_batch_contrast_loss', loss_val_batch_contrast_loss.item(), save_index_1+1)
                    writer.add_scalar('loss/val_batch_total_loss', loss_val.item(), save_index_1+1)
                    save_index_1 += 1
                    
                    restored_norm = (restored - restored.min()) / (restored.max() - restored.min())
                    target_norm = (target - target.min()) / (target.max() - target.min())

                    restored_norm = torch.clamp(restored_norm,0,1)  
                    target_norm = torch.clamp(target_norm,0,1)

                    psnr_val_rgb.append(utils.batch_PSNR(restored_norm, target_norm, False).item()) 
                    ssim_val_rgb.append(utils.batch_SSIM(restored_norm, target_norm, False).item())

                psnr_val_rgb = sum(psnr_val_rgb)/len_valset 
                ssim_val_rgb = sum(ssim_val_rgb)/len_valset

                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i 
                    torch.save({'epoch': epoch, 
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))

                print("[Ep %d it %d\t PSNR: %.4f\t] ----  [best_Ep %d best_it %d Best_PSNR %.4f] " % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr))
                print("SSIM_val: %.4f\t" % (ssim_val_rgb))
                with open(logname,'a') as f:
                    f.write("[Ep %d it %d\t PSNR: %.4f\t] ----  [best_Ep %d best_it %d Best_PSNR %.4f] " \
                        % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr)+'\n')
                    f.write("SSIM_val: %.4f\t" % (ssim_val_rgb)+'\n')
                    
                writer.add_scalar('metrics/val_psnr', psnr_val_rgb, epoch)
                writer.add_scalar('metrics/val_ssim', ssim_val_rgb, epoch)

                if eval_count == 4 and epoch%10==0:
                    input_ft_grid = make_grid(input_[:, 0, :, :].unsqueeze(1), nrow=input_.shape[0]) 
                    target_real_grid = make_grid(target[:, 0, :, :].unsqueeze(1), nrow=target.shape[0])
                    restored_real_grid = make_grid(restored[:, 0, :, :].unsqueeze(1), nrow=restored.shape[0])
                    target_imag_grid = make_grid(target[:, 1, :, :].unsqueeze(1), nrow=target.shape[0])
                    restored_imag_grid = make_grid(restored[:, 1, :, :].unsqueeze(1), nrow=restored.shape[0])

                    input_ft_grid_norm = (input_ft_grid - input_ft_grid.min()) / (input_ft_grid.max() - input_ft_grid.min())
                    target_real_grid_norm = (target_real_grid - target_real_grid.min()) / (target_real_grid.max() - target_real_grid.min())
                    restored_real_grid_norm = (restored_real_grid - restored_real_grid.min()) / (restored_real_grid.max() - restored_real_grid.min())
                    target_imag_grid_norm = (target_imag_grid - target_imag_grid.min()) / (target_imag_grid.max() - target_imag_grid.min())
                    restored_imag_grid_norm = (restored_imag_grid - restored_imag_grid.min()) / (restored_imag_grid.max() - restored_imag_grid.min())

                    writer.add_image('image/input_ft_val', input_ft_grid_norm, epoch)
                    writer.add_image('image/target_real_val', target_real_grid_norm, epoch)
                    writer.add_image('image/restored_real_val', restored_real_grid_norm, epoch)
                    writer.add_image('image/target_imag_val', target_imag_grid_norm, epoch)
                    writer.add_image('image/restored_imag_val', restored_imag_grid_norm, epoch)

                model_restoration.train()
                torch.cuda.empty_cache()
    
    psnr_train_rgb = sum(psnr_train_rgb)/len_trainset 
    ssim_train_rgb = sum(ssim_train_rgb)/len_trainset
    print("[Ep %d\t trainset_PSNR: %.4f\t] " % (epoch, psnr_train_rgb))
    print("[Ep %d\t trainset_SSIM: %.4f\t] " % (epoch, ssim_train_rgb))
    with open(logname,'a') as f:
        f.write("[Ep %d\t trainset_PSNR: %.4f\t] " % (epoch, psnr_train_rgb)+'\n')
        f.write("[Ep %d\t trainset_SSIM: %.4f\t] " % (epoch, ssim_train_rgb)+'\n')
    writer.add_scalar('metrics/train_psnr', psnr_train_rgb, epoch)
    writer.add_scalar('metrics/train_ssim', ssim_train_rgb, epoch)

    scheduler.step()

    writer.add_scalar('loss/train_epoch_loss', epoch_loss, epoch) 

    if epoch % 10 == 0:
        input_ft_grid = make_grid(input_[:, 0, :, :].unsqueeze(1), nrow=input_.shape[0]) 
        target_real_grid = make_grid(target[:, 0, :, :].unsqueeze(1), nrow=target.shape[0])
        restored_real_grid = make_grid(restored[:, 0, :, :].unsqueeze(1), nrow=restored.shape[0])
        target_imag_grid = make_grid(target[:, 1, :, :].unsqueeze(1), nrow=target.shape[0])
        restored_imag_grid = make_grid(restored[:, 1, :, :].unsqueeze(1), nrow=restored.shape[0])

        input_ft_grid_norm = (input_ft_grid - input_ft_grid.min()) / (input_ft_grid.max() - input_ft_grid.min())
        target_real_grid_norm = (target_real_grid - target_real_grid.min()) / (target_real_grid.max() - target_real_grid.min())
        restored_real_grid_norm = (restored_real_grid - restored_real_grid.min()) / (restored_real_grid.max() - restored_real_grid.min())
        target_imag_grid_norm = (target_imag_grid - target_imag_grid.min()) / (target_imag_grid.max() - target_imag_grid.min())
        restored_imag_grid_norm = (restored_imag_grid - restored_imag_grid.min()) / (restored_imag_grid.max() - restored_imag_grid.min())

        writer.add_image('image/input_ft_train', input_ft_grid_norm, epoch)
        writer.add_image('image/target_real_train', target_real_grid_norm, epoch)
        writer.add_image('image/restored_real_train', restored_real_grid_norm, epoch)
        writer.add_image('image/target_imag_train', target_imag_grid_norm, epoch)
        writer.add_image('image/restored_imag_train', restored_imag_grid_norm, epoch)

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname,'a') as f:
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0])+'\n')

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    if epoch%opt.checkpoint == 0:
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
print("Now time is : ",datetime.datetime.now().isoformat())