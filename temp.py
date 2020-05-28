# THis is file is only for debug purposes

import torch
from models import MRnet
from dataset import MRData

import tqdm

print("Loading Data...")
data_train = MRData(task='acl')

train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=1, shuffle=True, num_workers=11, drop_last=False)


# image,label = data_train[0]

print("Loading Model...")
net = MRnet()

criterion = torch.nn.CrossEntropyLoss()

# print(train_loader[0])

for x,y in train_loader:
#     print(x[0])
#     print(y.shape)
    output = net(x)

    print(output)
    print(torch.argmax(output).item())
    print(y[0].item())
#     print(output)
#     loss = criterion(output,y)
#     print(loss)
    break