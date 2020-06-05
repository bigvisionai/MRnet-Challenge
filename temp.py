# THis is file is only for debug purposes

import torch
from models import MRnet
from dataset import MRData
from dataset import load_data

import tqdm

print("Loading Data...")
train_loader, _, wts, _ = load_data(task = 'acl')

for x,y in train_loader:

    print(x[0].shape)
    print(y[0])
    print(wts)
    break