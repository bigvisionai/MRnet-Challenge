# MRNet
This repo contains code for the MRNet Challenge

For more details refer to https://stanfordmlgroup.github.io/competitions/mrnet/

# Folder Structure to be maintained

put `train` and `valid` folder inside `images` folder at root directory. Also put all labels inside
`images` folder.

# Understanding the Dataset

The dataset contains MRIs of different people. Each MRI consists of multiple images.
Each MRI has data in 3 perpendicular planes. And each plane as variable number of slices.

Each slice is an `256x256` image

For example:

For `MRI 1` we will have 3 planes:

Plane 1- with 35 slices

Plane 2- with 34 slices

Place 3 with 35 slices

Each MRI has to be classisifed against 3 diseases

# Approach for the model

Major challenge with while selectingt the model structure was the inconsistency in the data. Although the image size remains constant , the number of slices per plane are variable within a single MRI and varies across all MRIs.

So we are proposing a model for each plane. For each model the `batch size` will be variable and equal to `number of slices in the plane of the MRI`. So training each model, we will get features for each plane.

We also plan to have 3 separate models for each disease. --NEEDS TO BE DISCUSSED

# Internals of the Model

We will be using Resnet50 pretrained as a feature extractor. When we would have trained the 3 models on the 3 planes, we will use its feature extractor layer as an input to a `global` model for the final classification

# Another discussion

We might even use a fixed batch size model for each plane and treat our data set as randomly distributed images for a plane. Anyways we have to figure out how to come up with an structure that aloows us to extract maximun info from the `MRIs`.

