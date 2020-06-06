
# Stanford MRnet Challenge
This repo contains code for the MRNet Challenge

For more details refer to https://stanfordmlgroup.github.io/competitions/mrnet/

# Instructions to run the training
1. Clone the repository.

2. Download the dataset (~5.7 GB), and put `train` and `valid` folders along with all the the `.csv` files inside `images` folder at root directory. 

3. Make a new folder called `weights` at root directory, and inside the `weights` folder create three more folders namely `acl`, `abnormal` and `meniscus`.

4. All the hyperparameters are defined in `config.py` file. Feel free to play around those.

5. Now finally run the training using `python train.py`. All the logs for tensorboard will be stored in the `runs` directory at the root of the project.

# Our results
TODO

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

Major challenge with while selecting the model structure was the inconsistency in the data. Although the image size remains constant , the number of slices per plane are variable within a single MRI and varies across all MRIs.

So we are proposing a model for each plane. For each model the `batch size` will be variable and equal to `number of slices in the plane of the MRI`. So training each model, we will get features for each plane.

We also plan to have 3 separate models for each disease. 

# Model Specifications
We will be using Alexnet pretrained as a feature extractor. When we would have trained the 3 models on the 3 planes, we will use its feature extractor layer as an input to a `global` model for the final classification


# Link to Notebook
Here is the link to the notebook where we run our code :
https://colab.research.google.com/drive/157nwJdcUAAfu1LMSqekPYDtuDqz4ZpHA?usp=sharing

# TODOs
1. Add results to the readme
2. add code to save model when required, right now model is saved after every epoch.
3. update readme with correct model specifications.
