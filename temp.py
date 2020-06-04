from dataloader import MRDataset
from model import MRNet
import torch

train_dataset = MRDataset('./data/','acl',
                              'axial', train=True)


train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=11, drop_last=False)

model = MRNet()
model = model.cuda()


for image, label, _ in train_loader:
    image = image.cuda()
    out = model(image)
    print(out)
    print(image.shape)
    break