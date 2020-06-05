# THis is file is only for debug purposes

import torch
from models import MRnet
from dataset import load_data

import tqdm

print("Loading Data...")
train_loader, _, train_wts, _ = load_data(task = 'acl')

model = MRnet()
if torch.cuda.is_available():
    model = model.cuda()
    train_wts = train_wts.cuda()

with torch.no_grad():
    for images, label in train_loader:

        if torch.cuda.is_available():
            images = [image.cuda() for image in images]
            label = label.cuda()

        output = model(images)
        print(label)
        print(output)
        break