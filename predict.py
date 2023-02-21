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
import requests

# Albumentations
import albumentations
from albumentations.pytorch import ToTensorV2

from config import CFG
from DANmodel import *
from ClassMapping import JsonToDict, Decoder
from LoadDataset import loader


def prediction(params):
    ## Defining Model
    if 'efficientnet' in CFG.model_name:
        model = EmotionNet(pretrained=CFG.pretrained).to(CFG.device)
    elif CFG.model_name == 'DAN':
        model = DAN(num_head=4).to(CFG.device)
    model.load_state_dict(torch.load(os.path.join(CFG.save_path, f'{CFG.model_name}_best.pth'), map_location=CFG.device))
    model.eval()

    image = loader(params)
    image = image.unsqueeze(0)
    image = image.to(CFG.device)
        
    if 'efficientnet' in CFG.model_name:
        out = model(image)

    elif CFG.model_name == 'DAN':    
        out,feat,heads = model(image)

    _, predicts = torch.max(out, 1)
    pred_class = Decoder(predicts.item())
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