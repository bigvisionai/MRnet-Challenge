import torch
import torch.nn as nn
from torchvision import models
import os

class MRnet(nn.Module):
    """MRnet uses pretrained resnet50 as a backbone to extract features
    """
    
    def __init__(self): # add conf file

        super(MRnet,self).__init__()

        # init three backbones for three axis
        self.axial = models.alexnet(pretrained=True).features
        self.coronal = models.alexnet(pretrained=True).features
        self.saggital = models.alexnet(pretrained=True).features

        self.fc = nn.Sequential(
            nn.Linear(in_features=3*512,out_features=1)
        )

    def forward(self,x):
        """ Input is given in the form of `[image1, image2, image3]` where
        `image1 = [1, slices, 3, 224, 224]`. Note that `1` is due to the 
        dataloader assigning it a single batch. 
        """

        # squeeze the first dimension as there
        # is only one patient in each batch
        images = [torch.squeeze(img, dim=0) for img in x]

        image1 = self.axial(images[0]).view(-1,512)
        image2 = self.coronal(images[1]).view(-1,512)
        image3 = self.saggital(images[2]).view(-1,512)

        image1 = torch.max(image1,dim=0,keepdim=True)[0]
        image2 = torch.max(image2,dim=0,keepdim=True)[0]
        image3 = torch.max(image3,dim=0,keepdim=True)[0]

        output = torch.cat([image1,image2,image3], dim=1)

        output = self.fc(output)
        return output

    def _load_wieghts(self):
        """load pretrained weights"""
        pass