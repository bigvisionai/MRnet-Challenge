# THis is file is only for debug purposes

import torch
from models import MRnet
from dataset import load_data
from dataset import MRData
import torch.utils.data as data

import tqdm

# print("Loading Data...")
# train_loader, _, train_wts, _ = load_data(task = 'acl')

# model = MRnet()
# if torch.cuda.is_available():
#     model = model.cuda()
#     train_wts = train_wts.cuda()

train_loader, val_loader, train_wts, val_wts = load_data('acl')


model = MRnet()
model = model.cuda()

if torch.cuda.is_available():
    model = model.cuda()
    train_wts = train_wts.cuda()
    val_wts = val_wts.cuda()

print(len(train_loader))
print(len(val_loader))

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train_wts).cuda()

for images, label in train_loader:
    # output = model(images)
    if torch.cuda.is_available():
            images = [image.cuda() for image in images]
            label = label.cuda()
    print(label)
    print(images[0].shape)

    out = model(images)
    print(out)
    print(label)
    print(criterion(out,label))
    break
    # print(output)