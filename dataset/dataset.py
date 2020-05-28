import os
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.data as data
from torchvision import transforms
import cv2

class MRData():
    """This class used to load MRnet dataset from `./images` dir
    """

    def __init__(self,task = 'acl', train = True, transform = None, weights = None):
        """Initialize the dataset

        Args :
            plane : along which plane to load the data
            task : for which task to load the labels
            train : whether to load the train or val data
            transform : which transforms to apply

        """
        self.planes=["axial","coronal","sagittal"]
        self.records = None
        # an empty dictionary
        self.image_path={}
        
        if train:
            self.records = pd.read_csv('./images/train-{}.csv'.format(task),header=None, names=['id', 'label'])

            '''
            self.image_path[<plane>]= dictionary {<plane>: path to folder containing
                                                                image for that plane}
            '''
            for plane in self.planes:
                self.image_path[plane] = './images/train/{}/'.format(plane)
        else:
            transform = None
            self.records = pd.read_csv('./images/valid-{}.csv'.format(task),header=None, names=['id', 'label'])
            '''
            self.image_path[<plane>]= dictionary {<plane>: path to folder containing
                                                                image for that plane}
            '''
            for plane in self.planes:
                self.image_path[plane] = './images/valid/{}/'.format(plane)

        
        self.transform = transform 

        self.records['id'] = self.records['id'].map(
            lambda i: '0' * (4 - len(str(i))) + str(i))
        # empty dictionary
        self.paths={}    
        for plane in self.planes:
            self.paths[plane] = [self.image_path[plane] + filename +
                          '.npy' for filename in self.records['id'].tolist()]

        self.labels = self.records['label'].tolist()
        

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.records)

    def __getitem__(self, index):
        """
        Returns `(images,labels)` pair
        where image is a list [imgsPlane1,imgsPlane2,imgsPlane3]
        and labels is a list [gt,gt,gt]
        """
        img_raw={}
        
        for plane in self.planes:
            img_raw[plane] = np.load(self.paths[plane][index])

            # array to collect new resized images
            new = []
            
            for i in range(img_raw[plane].shape[0]):
                inter = cv2.resize(img_raw[plane][i],(224,224), interpolation=cv2.INTER_AREA)
                inter_ = np.zeros((224,224))
                inter_ = cv2.normalize(inter, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                new.append(inter_)
            
            img_raw[plane]=np.array(new)   
            
        label = self.labels[index]
        
        # apply transforms if possible, or else stack 3 images together
        # Note : if applying any transformation, use 3 to generate 3 images
        # but they should be almost similar to each other
        for plane in self.planes:
            if self.transform:
                img_raw[plane] = self.transform(img_raw[plane])
            else:
                img_raw[plane] = np.stack((img_raw[plane],)*3, axis=1)
                img_raw[plane] = torch.FloatTensor(img_raw[plane])

        return [img_raw[plane] for plane in self.planes ], label

    def pre_epoch_callback(self, epoch):
        """Callback to be called before every epoch.
        """
        pass

    def post_epoch_callback(self, epoch):
        """Callback to be called after every epoch.
        """
        pass