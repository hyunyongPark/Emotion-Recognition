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


def test_run(data):
    ## Defining Model
    if 'efficientnet' in CFG.model_name:
        model = EmotionNet(pretrained=CFG.pretrained).to(CFG.device)
    elif CFG.model_name == 'DAN':
        model = DAN(num_head=4).to(CFG.device)
    model.load_state_dict(torch.load(os.path.join(CFG.save_path, f'{CFG.model_name}_best.pth'), map_location=CFG.device))
    model.eval()

    import tqdm

    correct_sum = 0.0
    total = 0
    non_correct_idx = []
    for i in tqdm.tqdm(range(len(data))):
        img = data.image_path[i]
        label_row = data.label[i]
        targets = torch.tensor(label_row)

        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transforms = get_valid_transforms()
        augmented = transforms(image=image)
        image = augmented['image']

        image = image.unsqueeze(0)

        image = image.to(CFG.device)
        targets = targets.to(CFG.device)
        targets = targets.long()
        
        if 'efficientnet' in CFG.model_name:
            out = model(image)
            
        elif CFG.model_name == 'DAN':    
            out,feat,heads = model(image)
            
        total += 1
        _, predicts = torch.max(out, 1)
        correct_num = torch.eq(predicts, targets).sum()
        if correct_num != 1:
            non_correct_idx.append(i)
        correct_sum += correct_num
    acc = correct_sum.float().detach().item() / total
    print(f"Accuracy : {acc}")

if __name__ == '__main__':
    torch.manual_seed(225)
    eps = sys.float_info.epsilon
    te = pd.read_csv(os.path.join(CFG.data_path, "test.csv"), index_col = 0)
    test_run(te)
    
