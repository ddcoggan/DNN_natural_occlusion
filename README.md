## Overview

This repository contains model weights, human behavioral data, and 
experiment and analysis code for the paper:

"Exposure to naturalistic occlusion promotes generalized, human-like 
robustness in deep neural networks" by David D. Coggan and Frank Tong
([preprint](https://doi.org/10.64898/2026.04.23.720370))

This paper shows that augmenting DNN training datasets with simple but 
artificial forms of occlusion fails to elicit robustness to natural 
occlusion and causes DNNs to diverge from human patterns of performance. 
Conversely, applying naturalistic occlusion to training images leads to more 
human-like patterns of performance and better generalization to novel forms of 
occlusion outside the training distribution. These findings suggest that 
human robustness to visual occlusion arises not because of our extensive 
exposure to partial object views, which occur under any form of occlusion, 
but rather because of our experience with disentangling real objects that 
occlude one another in the visual field. They also suggest that artificial 
forms of occlusion similar to those used here (e.g., patch drop, random 
erasing, cutout, etc.) are unsuitable for both promoting and measuring 
robustness to real-world occlusion in DNNs.

All plots and statistical results used in 
the paper are already included under `DNN/evaluation/figures`, but can be 
reproduced by running `DNN/evaluation/make_figures.py`. The underlying 
performance data for each DNN in each experiment is included in the 
`DNN/models` directory. The human behavioral data is contained in 
`humans/trials.parquet`.

## Setup
The analysis code was written and tested using Python 3.11. To install 
dependencies, run:
`pip install -r requirements.txt`. This repository was developed on a Mac 
Studio (2022) with Apple M1 Max CPU and Tahoe 26.5 macOS. On this machine, 
the repository takes a few seconds to download and ~2 minutes to 
download and install dependencies. A GPU is not necessary to run any 
analysis code, as the model responses/performance is already contained in 
the repository.

## Model weights and experimental stimuli
These can be added by running `downloads.py`, although this is 
not necessary for reproducing the results. The script downloads a single zip 
file for all model weights (~27GB), which is then unpacked and organized into 
the appropriate model directories. It also downloads the stimuli used in the 
human behavioral experiments (~0.5GB) and places them under `humans/images`.

## Visual Occluders Dataset
The Visual Occluders Dataset used in this study can be obtained 
[here](https://github.com/ddcoggan/VisualOccludersDataset), along with code that applies it to image datasets.

## Training your own DNNs
To train your own DNNs with occlusion, you can use my [model trainer](https://github.com/ddcoggan/model_trainer) to perform training jobs configured in a json file (see this [example](DNN/models/original/cornet_s_plus/natural/args.json)). You will need the 
Visual Occluders Dataset or your own dataset of occluders with the same file 
structure.

## Evaluating your own DNNs
To evaluate your own DNNs on the same experiments as in the paper, you can 
use the code in `DNN/evaluation`. To evaluate DNN-human similarity, run 
`evaluate_human_likeness.py`, ensuring you adapt to code to load your model 
and weights. To evaluate robustness, run 
`imagenet_occluded.py`, ensuring you have the ImageNet-1K validation set 
and the Visual Occluders Dataset. Depending on your needs and 
computational resources, you can either pre-apply occluders to the 
validation set or apply these on the fly during evaluation. See the documentation included in this file for more information.






