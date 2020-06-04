import torch
import torch.nn as nn
from torchvision import models


class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        # init three backbones for three axis
        self.pretrained_model = self._generate_resnet()

        self.fc = nn.Sequential(
            nn.Linear(in_features=2048,out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024,out_features=2),
        )


    def forward(self, x):
        # squeeze the first dimension as there
        # is only one patient in each batch
        image = torch.squeeze(x, dim=0)
        image = self.pretrained_model(image).view(-1,2048)
        image = torch.max(image,dim=0,keepdim=True)[0]
        output = self.fc(image)
        # no need to take softmax here
        # as cross_entropy loss combines both softmax and NLL loss
        return output

    def _generate_resnet(self):
        """make all resnet params non-trainable, called automatically
        in `__init__` and then generate a Resnet50 model to be used as a backbone
        """
        # init resnet
        backbone = models.resnet50(pretrained=True)
        resnet_modules = list(backbone.children())

        # remove last layer of resnet
        body = nn.Sequential(*resnet_modules[:-1])
        
        # make params non trainable
        for x in body.parameters():
            x.requires_grad = False

        return body