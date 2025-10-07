This repository contains model weights, human behavioral data, and 
experiment and analysis code for the upcoming paper:

"Exposure to naturalistic occlusion promotes generalized, human-like 
robustness in deep neural networks" by David D. Coggan and Frank Tong

This paper shows that, relative to standard training datasets, those 
augmented with simple but artificial forms of occlusion cause DNNs to diverge 
from human patterns of performance and does not generalize well when 
tested with naturalistic occlusion. Conversely, applying naturalistic occlusion 
to training images leads to more human-like patterns of performance and better 
generalization to both artificial and natural occluders outside the 
training distribution. These findings suggest that human robustness to visual 
occlusion arises not because of our extensive experience with partial object 
views, which occur under any form of occlusion, but rather because of our 
experience with disentangling real objects that occlude one another in the 
visual field. They also suggest that artificial forms of occlusion similar 
to those used here (e.g., patch drop, random erasing, cutout, etc.) are 
unsuitable for both instilling and measuring robustness to real-world 
occlusion in DNNs.

The Visual Occluders Dataset used in this study can be obtained 
[here](https://github.com/ddcoggan/VisualOccludersDataset), along with code 
for generating datasets of occluded images or augmenting images with 
occlusion as they are loaded during DNN training or evaluation.

The code was written and tested using Python 3.11, and dependencies can be 
found in the requirements.txt file.

