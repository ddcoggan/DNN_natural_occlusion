## Overview

This repository contains the code and data used in the following paper:

"Exposure to naturalistic occlusion promotes generalized, human-like 
robustness in deep neural networks" by David D. Coggan and Frank Tong
([preprint](https://doi.org/10.64898/2026.04.23.720370)).

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
the paper are already included under   The underlying 
performance data for each DNN in each experiment is included in the 
`DNN/models` directory. The human behavioral data is contained in 
`humans/trials.parquet`.

## System requirements
All code was written and tested using Python 3.11. The code for plots and statistical results was developed on a Mac Studio (2022, 
Apple M1 Max CPU, Tahoe 26.5 macOS). The DNN training/evaluation code was developed on Ubuntu 22.04.5 LTS, and is recommended 
to be run with a cuda-compatible GPU. The code for plots and statistical results has also been tested on 
this set up. Evaluation code is contained in this repository, and the training code
is available [here](https://github.com/ddcoggan/model_trainer). 

## Installation guide
Clone the repository and create an associated Python 3.11 virtual environment. In a terminal, navigate to the top-level 
directory of the repository and install dependencies by running: `pip install -r requirements.txt`. On a typical 
machine, the repository takes a few seconds to download and dependencies take a few minutes to download and install.
All human and model responses, performance data, results figures and statistics outputs are already included under 
`DNN/evaluation/figures`. These are sufficient for reproducing the results figures and statistical tests, but further 
installation steps are necessary if you wish to reperform model training or evaluation (details below).

## Reproducing results
Reproducing the results figures and statistical outputs can be performed from the existing data without further 
installation by running `DNN/evaluation/make_figures.py`. This takes a few minutes to run in its entirety on a typical 
machine. To create examples of distorted images, you will need a local copies of 
[ImageNet-1K](https://www.image-net.org/download.php) and the
[Visual Occluders Dataset](https://github.com/ddcoggan/VisualOccludersDataset).

To reproduce the DNN-human similarity evaluation, you will need to obtain the experimental stimuli and model weights by 
running `downloads.py`. This script downloads a single zip file for all model weights (~27GB), which is then unpacked 
and organized into the appropriate model directories. It also downloads the stimuli used in the 
human behavioral experiment (~0.5GB) and places them under `humans/images`. A model can then be evaluated on the 
dataset by running `DNN/evaluation/evaluate_human_likeness.py`, after configuring the model directory at the bottom of 
the script. The processing time for each model will vary substantially depending on the architecture size and your 
hardware, but generally will take a few minutes per model on a relatively modern consumer-grade GPU. Executing the script as is
will replicate the evaluation of alexnet trained without occlusion, with expected results shown at the bottom of the 
script.

To reproduce the DNN robustness evaluation, you will also need the model weights as well as local copies of the 
[ImageNet-1K validation set](https://www.image-net.org/download.php) and 
[Visual Occluders Dataset](https://github.com/ddcoggan/VisualOccludersDataset). A model can then be evaluated by running 
`DNN/evaluation/imagenet_occluded.py`, ensuring that you point the script to correct model directory. Again, the 
processing time for each model will vary substantially depending on the model architecture, your hardware, batch 
size, and whether you opt to store the occluded objects on disk or apply occlusion on the fly during evaluation. That 
said, it will generally take a few hours to evaluate a medium-sized model on all occlusion conditions using a 
relatively modern consumer-grade GPU. Executing the script as is will replicate the evaluation of alexnet trained 
without occlusion, with expected results shown at the bottom of the script.

To reproduce the DNN training procedure, you will need the [model trainer](https://github.com/ddcoggan/model_trainer) (separate repository), local copies of 
the [ImageNet-1K training set](https://www.image-net.org/download.php) and 
[Visual Occluders Dataset](https://github.com/ddcoggan/VisualOccludersDataset), and the training configuration files 
contained in this repository (e.g., `DNN/models/original/resnet101/natural/args.json`). Further instructions and 
documentation are available at the [model trainer](https://github.com/ddcoggan/model_trainer) repo.

## Training/evaluating your own DNNs
Both the training and evaluation code natively support all model architectures in the torchvision model library plus a 
few others (e.g., CORnet-S+). The two evaluation scripts described above contain a second demo example 
showing how to add weights and evaluate pretrained AlexNet. To evaluate an unsupported architecture on these 
experiments, follow the included example of CORnet-S+, located at `DNN/cornet_s_plus.py`. Place your own architecture 
in the same directory and adapt `DNN/evaluation/utils/get_model.py` to accommodate it, then add your weights to the 
model directory and run the eval scripts. 

To train an unsupported architecture, you will need to add it to the model zoo of the [model trainer](https://github.com/ddcoggan/model_trainer) 
(separate repository) and generate a training configuration file (see this [example](DNN/models/original/cornet_s_plus/natural/args.json)). Further instructions and 
documentation are available at the [model trainer](https://github.com/ddcoggan/model_trainer) repository.






