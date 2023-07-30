# Segment Tumour
<<<<<<< HEAD
An automated tumour classification pipeline, using Pytorch
=======
An automated tumour segmentation pipeline, using Pytorch
>>>>>>> b63b917cc4c05c80a84f6261b51462fea15d4121

## 
Placeholder for project details and instructions










# Project plan

Create a brain tumour segmentation pipeline utilising pytorch.



---
Goals 
---
- Create a tumour segmenter with >80% accuracy
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