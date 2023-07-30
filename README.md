# Segment Tumour
An automated tumour classification pipeline, using Pytorch

## Techniques used
Pytorch, unit testing, version control, hyperparameter searching, normalisation. 

## 
Using the training and test images from ____ database.

Clone this repo to a folder of your choosing, making sure the images are place within the repo. 
Navigate to the 

If a validation set is not already created, it will create anew, using 20% of the training images. 

Afterwards the model will train itself on the images, saving itself to the folder along with the best hyperparameters. 

Script can then be run again to test the model.

# Project plan

Create a brain tumour segmentation pipeline utilising pytorch.



---
Goals 
---
- Create a tumour classifier with >70% accuracy
- Ensure test-driven development
- Deadline - 28/07/2023

### Scope 
- Tuned parameters to give highest accuracy
- Pre-process images to normalise dataset and remove noise
- Make pipeline generalisable to different datasets

## Development Roadmap
Day 1 : Set-up
- Create a git repo and conda environment
- Implement pytest and CI for TTD
- Download datasets that are going to be used
- Set up folder as if publishable

Day 2 : ML model
- Create normalisation functions to standardise data
- Build the model
- Split the dataset into test/dev/train
- Create an evaluating function, utilising dice coefficient to measure performance

Day 3 : Visualisation
- Output some visualisations of the MRI images and segmented tumour
- Implement interactive folder selection

Day 4 : Tuning
- Optimise the hyperparameters using grid search
- Write a detailed description on README
- Go public