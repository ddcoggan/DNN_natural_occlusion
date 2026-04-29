## Overview

This repository contains model weights, human behavioral data, and 
experiment and analysis code for the upcoming paper:

"Exposure to naturalistic occlusion promotes generalized, human-like 
robustness in deep neural networks" by David D. Coggan and Frank Tong

This paper shows that, relative to standard training datasets, those 
augmented with simple but artificial forms of occlusion cause DNNs to diverge 
from human patterns of performance and does not generalize well when 
tested with naturalistic occlusion. Conversely, applying naturalistic occlusion 
to training images leads to more human-like patterns of performance and better 
generalization to novel forms of occlusion outside the 
training distribution. These findings suggest that human robustness to visual 
occlusion arises not because of our extensive experience with partial object 
views, which occur under any form of occlusion, but rather because of our 
experience with disentangling real objects that occlude one another in the 
visual field. They also suggest that artificial forms of occlusion similar 
to those used here (e.g., patch drop, random erasing, cutout, etc.) are 
unsuitable for both promoting and measuring robustness to real-world 
occlusion in DNNs.


## Setup

The code was written and tested using Python 3.11. To install dependencies, run:

`pip install -r requirements.txt`

DNN weights can be added by running `download_weights.py`. The download is 
a single zip file ~27GB, which is then unpacked and organized into the 
appropriate model directories.

## Occlusion training

The Visual Occluders Dataset used in this study can be obtained 
[here](https://github.com/ddcoggan/VisualOccludersDataset), along with code 
for generating datasets of occluded images or augmenting images with 
occlusion as they are loaded during DNN training or evaluation.

To train your own DNNs with occlusion, you can use my model trainer 
(https://github.com/ddcoggan/model_trainer) to perform training jobs 
specified in a config file 
([example](DNN/models/original/cornet_s_plus/natural/cfg.json)).






