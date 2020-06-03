# THis is file is only for debug purposes

import torch
from models import MRnet
from dataset import MRData

print("Loading Data...")
data_train = MRData(task='acl')

train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=1, shuffle=True, num_workers=11, drop_last=False)

print("Loading Model...")
net = MRnet()

for imgs,label in train_loader:
    
    output = net(imgs)

    print(output)
    output = torch.sigmoid(output)
    print(output)

    break