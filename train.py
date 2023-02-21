# basic
import os
import sys
import numpy as np
import pandas as pd
import math
from tqdm import tqdm

# vision utils
import cv2

# skit learn
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

# Pytorch
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer
from torchvision import models

# vision pretrained model & achitecture
import timm

# Albumentations
import albumentations
from albumentations.pytorch import ToTensorV2





from config import CFG
from loss import AffinityLoss, PartitionLoss
from optimizer import *
from utils import AverageMeter, Accuracy
from DANmodel import *
from datasets import get_train_transforms, get_valid_transforms, EmotionDataset



def train_fn(dataloader, model, criterion, optimizer, device, scheduler, epoch):
    model.train()
    loss_score = AverageMeter()
    total = 0
    correct_sum = 0.0
    from tqdm import tqdm
    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    
    if CFG.model_name == 'DAN':
        criterion_cls, criterion_af, criterion_pt  = criterion[0], criterion[1], criterion[2]
    
    for bi, d in tk0:
        batch_size = d[0].shape[0]

        images = d[0]
        targets = d[1]

        images = images.to(device)
        targets = targets.to(device)
        targets = targets.long()
        
        if 'efficientnet' in CFG.model_name:
            out = model(images)
            loss = criterion(out, targets)
            
        elif 'DAN' == CFG.model_name:    
            out,feat,heads = model(images)
            loss = criterion_cls(out, targets) + 1* criterion_af(feat, targets) + 1*criterion_pt(heads)
        
        optimizer.zero_grad()
        loss.backward() # 미분값
        optimizer.step() # 

        #_, predicted = output.max(1)
        total += targets.size(0)
        _, predicts = torch.max(out, 1)
        correct_num = torch.eq(predicts, targets).sum()
        correct_sum += correct_num
        acc = correct_sum.float().detach().item() / total
        loss_score.update(loss.detach().item(), batch_size) # 평균계산
        tk0.set_postfix(Train_Loss=loss_score.avg, Epoch=epoch+1, LR=optimizer.param_groups[0]['lr'],
                        Accuracy = acc,  
                        )
        #neptune.log_metric('Training Loss', loss_score.avg)
        #neptune.log_metric('Training Accuracy', acc)
        #neptune.log_metric('Learning Rate', optimizer.param_groups[0]['lr'])
    
    return loss_score.avg, acc

def eval_fn(data_loader, model, criterion, device, scheduler):
    model.eval()
    loss_score = AverageMeter()
    total = 0
    correct_sum = 0.0
    from tqdm import tqdm
    tk0 = tqdm(enumerate(data_loader), total=len(data_loader))
    
    if CFG.model_name == 'DAN':
        criterion_cls, criterion_af, criterion_pt  = criterion[0], criterion[1], criterion[2]
    
    with torch.no_grad():
        for bi, d in tk0:
            batch_size = d[0].size()[0]

            images = d[0]
            targets = d[1]

            images = images.to(device)
            targets = targets.to(device)
            targets = targets.long()

            if 'efficientnet' in CFG.model_name:
                out = model(images)
                loss = criterion(out, targets)

            elif CFG.model_name == 'DAN':    
                out,feat,heads = model(images)
                loss = criterion_cls(out, targets) + 1* criterion_af(feat, targets) + 1*criterion_pt(heads)
            total += targets.size(0)
        
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
            acc = correct_sum.float().detach().item() / total
            
            loss_score.update(loss.detach().item(), batch_size) # 평균계산
            
            tk0.set_postfix(Valid_Loss=loss_score.avg,
                        Accuracy = acc, 
                        )
            #neptune.log_metric('Validation Loss', loss_score.avg)
            #neptune.log_metric('Validation Accuracy', acc)
            
            
    if CFG.scheduler_type != None:
        scheduler.step(loss_score.avg)
            
    return loss_score.avg, acc


def running_process(tr, val):

    ## Defining Dataset
    tr_dataset = EmotionDataset(data = tr, transform = get_train_transforms())
    val_dataset = EmotionDataset(data = val, transform = get_train_transforms())

    ## Defining Dataloader
    train_loader = torch.utils.data.DataLoader(
        tr_dataset,
        batch_size=CFG.Batch_size,
        num_workers=CFG.Num_worker,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=CFG.Batch_size,
        num_workers=CFG.Num_worker,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    ## Defining Model
    if 'efficientnet' in CFG.model_name:
        model = EmotionNet(pretrained=CFG.pretrained).to(CFG.device)
        criterion = torch.nn.CrossEntropyLoss()
        criterion.to(CFG.device)
    
    elif CFG.model_name == 'DAN':
        model = DAN(num_head=4).to(CFG.device)
        ## Defining Criterion
        criterion_cls = torch.nn.CrossEntropyLoss()
        criterion_af = AffinityLoss(CFG.device)
        criterion_pt = PartitionLoss()

        criterion_cls.to(CFG.device)
        criterion_af.to(CFG.device)
        criterion_pt.to(CFG.device)
        criterion = [criterion_cls, criterion_af, criterion_pt]
        params = list(model.parameters()) + list(criterion_af.parameters())
    
    ## Defining Optimizer
    if CFG.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr_start)
    elif CFG.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=CFG.lr_start, weight_decay = 1e-4, momentum=0.9)
    elif CFG.optimizer_type == 'adamw':
        optimizer = AdamW(model.parameters(), lr=CFG.lr_start, weight_decay=CFG.weight_decay)
    
    ## Defining Scheduler
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    if CFG.scheduler_type == 'ReduceLROnPlateau': 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **CFG.ReduceLROnPlateau_PARAMS)
        
    elif CFG.scheduler_type == 'OneCycleLR':
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=steps_per_epoch, epochs=CFG.Epoch, **CFG.OneCycleLR_PARAMS)
    
    best_loss = 10000
    best_acc = 0
    stop_counts = 0
    loss_plot_tr = []
    loss_plot_val = []
    acc_plot_tr = []
    acc_plot_val = []
    for epoch in range(CFG.Epoch):
        
        if CFG.scheduler_type == None:
            train_loss, tr_acc = train_fn(train_loader, model, criterion, optimizer, CFG.device, scheduler=None, epoch=epoch)
            valid_loss, acc= eval_fn(valid_loader, model, criterion, CFG.device, scheduler=None)
        else:
            train_loss, tr_acc = train_fn(train_loader, model, criterion, optimizer, CFG.device, scheduler=scheduler, epoch=epoch)
            valid_loss, acc = eval_fn(valid_loader, model, criterion, CFG.device, scheduler=scheduler)
        loss_plot_tr.append(train_loss)
        loss_plot_val.append(valid_loss)
        acc_plot_tr.append(tr_acc)
        acc_plot_val.append(acc)
        if acc > best_acc:
            torch.save(model.state_dict(), os.path.join(CFG.save_path, f'{CFG.model_name}_best.pth'))
            best_acc = acc
            print('Accuracy :  best model found for epoch {}'.format(epoch+1))
            print("Current Best Accuracy: {}".format(best_acc))
            
            acc_1 = acc
            stop_counts = 0
        else:
            stop_counts += 1
            if stop_counts == 7:
                print(f'Early Stopping : {epoch+1} / Final Accuracy : {best_acc}')
                break
    
    return loss_plot_tr, loss_plot_val, acc_plot_tr, acc_plot_val


if __name__ == '__main__':
    torch.manual_seed(225)
    eps = sys.float_info.epsilon
    tr = pd.read_csv(os.path.join(CFG.data_path, "train.csv"), index_col = 0)
    val = pd.read_csv(os.path.join(CFG.data_path, "valid.csv"), index_col = 0)
    loss_plot_tr, loss_plot_val, acc_plot_tr, acc_plot_val = running_process(tr, val)
