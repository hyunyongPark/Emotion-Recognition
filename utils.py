import numpy as np
import pandas as pd
import math
import torch

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1): #mean , 128
        self.val = val
        self.sum += val * n # mini-batch loss sum 
        self.count += n # 
        self.avg = self.sum / self.count

def Accuracy(pred, label):
    pred = pred.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    
    pred = [np.argsort(x)[::-1][0] for x in pred]
    
    acc_1 = 0
    for i in range(len(pred)):
        if pred[i] == label[i]:
            acc_1 +=1

    return acc_1