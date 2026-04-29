# Created by David Coggan on 2024 01 04
import pickle as pkl
import glob
import os.path as op
import json
import torchvision
from .get_model import get_model
from .load_params import load_params
from . import MODEL_BASE

def get_trained_model(model_dir, architecture,
                      return_states=False, return_blocks=None):

    # get kwargs for configurable DNN
    args_file = glob.glob(f'{model_dir}/*.json')
    if args_file:
        with open(args_file[0], 'r') as f:
            args = json.load(f)
        kwargs = args['architecture_args']
    if architecture == 'cornet_s_custom':
        with open(f'{model_dir}/config.pkl', 'rb') as f:
            M = pkl.load(f).M
        M.model_name = architecture
        M.return_states = return_states
        kwargs = {'M': M}
    elif architecture in ['cornet_rt_hw3', 'cognet_v9', 'cognet_v10']:
        kwargs = {'return_states': return_states,
                  'return_blocks': return_blocks}
    #elif architecture == 'vit_b_16':
    #    kwargs = {'image_size': 224}
    elif architecture == 'prednet':
        kwargs = dict(
            stack_sizes = (3, 48, 96, 192),
            R_stack_sizes = (3, 48, 96, 192),
            A_filter_sizes = (3, 3, 3),
            Ahat_filter_sizes = (3, 3, 3, 3),
            R_filter_sizes = (3, 3, 3, 3),
            output_mode = 'error',
            return_sequences = True)
    else:
        kwargs = {}

    # workaround for DNN that don't load downloaded pretrained weights well
    if model_dir == 'vit_b_16/pretrained_IMAGENET1K_SWAG_E2E_V1':
        model = torchvision.models.vit_b_16(
            weights='IMAGENET1K_SWAG_E2E_V1')
    elif model_dir == 'vit_h_14/pretrained_IMAGENET1K_SWAG_E2E_V1':
        model = torchvision.models.vit_h_14(
            weights='IMAGENET1K_SWAG_E2E_V1')
    elif model_dir == 'vit_l_16/pretrained_IMAGENET1K_SWAG_E2E_V1':
        model = torchvision.models.vit_l_16(
            weights='IMAGENET1K_SWAG_E2E_V1')
    else:

        # get model and load weights normallu
        model = get_model(architecture, kwargs)
        params_path = sorted(glob.glob(op.join(
            MODEL_BASE, model_dir, 'params/???.pt*')))[-1]
        model = load_params(params_path, model, 'model')

    return model


