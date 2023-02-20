# basic
import os
import sys
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import argparse


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
from torchmetrics import Accuracy
from torch.optim.optimizer import Optimizer
from torchvision import models

# vision pretrained model & achitecture
import timm
import requests

# Albumentations
import albumentations
from albumentations.pytorch import ToTensorV2

from config import CFG
from loss import AffinityLoss, PartitionLoss
from optimizer import *
from utils import AverageMeter, Accuracy
from model import *
from datasets import get_train_transforms, get_valid_transforms, EmotionDataset



def prediction(params):
    ## Defining Model
    if 'efficientnet' in CFG.model_name:
        model = EmotionNet(pretrained=CFG.pretrained).to(CFG.device)
    elif CFG.model_name == 'DAN':
        model = DAN(num_head=4).to(CFG.device)
    model.load_state_dict(torch.load(os.path.join(CFG.save_path, f'{CFG.model_name}_best.pth'), map_location="cuda:0"))
    model.eval()

    import tqdm
    
    mappings = {'0': 'anger',
                '1': 'disgust',
                '2': 'fear',
                '3': 'happy',
                '4': 'sadness',
                '5': 'surprise'}
    
    image_nparray = np.asarray(bytearray(requests.get(params).content), dtype=np.uint8)
    image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
    #image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transforms = get_valid_transforms()
    augmented = transforms(image=image)
    image = augmented['image']

    image = image.unsqueeze(0)

    image = image.to(CFG.device)
        
    if 'efficientnet' in CFG.model_name:
        out = model(image)

    elif CFG.model_name == 'DAN':    
        out,feat,heads = model(image)

    _, predicts = torch.max(out, 1)
    pred_class = mappings[str(predicts.item())]
    return pred_class


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', type=str, default="https://img.freepik.com/free-photo/portrait-of-smiling-young-man-looking-at-camera_23-2148193854.jpg", help="image url")
    
    args = parser.parse_args()

    torch.manual_seed(225)
    eps = sys.float_info.epsilon
    pred_class = prediction(args.params)
    predict_json = {args.params : pred_class}
    print(predict_json)