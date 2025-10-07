"""
Train models using config file. Run this from the top-level directory of the
repository.
"""
import os
import os.path as op
from argparse import Namespace
import json

"""
# set cuda GPU visibility
gpus = input(f'Which GPU(s)? E.g., 0 or 0,1 :')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # converts to nvidia-smi order
os.environ['CUDA_VISIBLE_DEVICES'] = gpus
"""

# import and configure torch settings AFTER restricting GPU visibility
import torch
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(False)

# path to config file
config_dir = '/Users/david/PyCharmProjects/DNN_natural_occlusion/models' \
             '/training/configs'
config = op.join(config_dir, 'cornet_s_plus__classification__natural.json')

# load config, create model directory
with open(config, 'r') as f:
    args = Namespace(**json.load(f))
os.makedirs(args.model_dir, exist_ok=True)

# calculate / set defaults for any missing values in config
from utils.complete_args import complete_args
args = complete_args(args, update=False)

# train model
from utils.optimize_model import optimize_model
optimize_model(args, verbose=True)
