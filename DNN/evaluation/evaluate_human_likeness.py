'''
This script tests the accuracy of CNNs on classifying the exact images
presented in the human behavioral experiment.
'''

import os
import os.path as op
import glob
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
from scipy import stats
from scipy import special
import pickle as pkl
import pandas as pd
from scipy.optimize import curve_fit
import math
import time
from types import SimpleNamespace
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from itertools import product as itp
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import gc
import torchvision
import torchvision.transforms.v2 as transforms
from torchvision.utils import save_image
from torch import float32
from utils import insert_cycle, now, MODEL_BASE
from utils.get_trained_model import get_trained_model
from utils.image_processing import tile
from utils.get_activations import get_activations
from utils.math_functions import sigmoid
from utils.plot_utils import custom_defaults
plt.rcParams.update(custom_defaults)
from humans.analysis import CFG
from humans.analysis import condwise_robustness_plot_array

np.random.seed(42)

OBJ_VARIABLES = ['object_animacy', 'object_class']
OBJ_ANIMACIES = ['animate', 'inanimate']
OBJ_CLASSES = CFG.object_classes
OBJ_CLS_IDCS = CFG.class_idxs
OBJ_CLS_DIRS = CFG.synsets
OCC_VARIABLES = ['visibility', 'occluder_class', 'occluder_color']
OCC_COLORS = ['black', 'white']
OCC_CLASSES = CFG.occluder_classes
VISIBILITIES = CFG.visibilities
RES_DIR = 'human_behavioral_exp'

def get_transform(architecture, model_dir):

    if hasattr(torchvision.models, architecture) and 'pretrained' in model_dir:
        model_attr = str([i for i in torchvision.models.__dict__ if
                             i.lower() == f'{architecture}_weights'][0])
        weights_attr = model_dir.split('pretrained_')[-1]
        transform = getattr(getattr(torchvision.models, model_attr),
                            weights_attr).transforms()
        return transform

    imsize = 256
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(float32, scale=True),
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),  # in case resize is off by a pixel
        transforms.Grayscale(num_output_channels=3),  # should be redundant
        transforms.Normalize(mean=[0.445, 0.445, 0.445],
                             std=[0.269, 0.269, 0.269]),
    ])
    return transform


def load_trials(drop_human=False, model_dir=None):
    if model_dir is None:
        trials = pd.read_parquet(f'humans/trials.parquet')
    else:
        trials = pd.read_parquet(op.join(
            MODEL_BASE, model_dir, RES_DIR, 'trials.parquet'))
        drop_human = False  # human data not present

    # drop human data
    if drop_human:  # human data
        trials.drop(columns=['prediction', 'accuracy', 'RT'], inplace=True)

    # enforce ordering of categorical variables
    trials.object_animacy = pd.Categorical(
        trials.object_animacy, OBJ_ANIMACIES, ordered=True)
    trials.object_class = pd.Categorical(
        trials.object_class, OBJ_CLASSES, ordered=True)
    trials.occluder_class = pd.Categorical(
        trials.occluder_class, OCC_CLASSES, ordered=True)
    trials.occluder_color = pd.Categorical(
        trials.occluder_color, OCC_COLORS, ordered=True)

    return trials


def list_images():
    images = sorted(glob.glob(f'humans/images/final/*.png'))
    return images


def get_responses(model_dir, architecture, layers=('output'), batch_size=64,
                  m=0, total_models=0, num_procs=1, overwrite=False):

    results_dir = op.join(MODEL_BASE, model_dir, RES_DIR)
    os.makedirs(results_dir, exist_ok=True)

    out_path = f'{results_dir}/trials.parquet'
    if not op.isfile(out_path):
        prev_trials = pd.DataFrame()
    else:
        prev_trials = pd.read_parquet(out_path)
        if overwrite:
            prev_trials = prev_trials[~prev_trials.layer.isin(layers)]
        layers = [i for i in layers if i not in prev_trials.layer.unique()]
        if not len(layers):
            return False

    non_output_layers = [l for l in layers if l != 'output']
    if len(non_output_layers):
        print(f'{now()} | Loading PCA and SVC objects...')
        transfer_dir = op.join(MODEL_BASE, model_dir, RES_DIR,
                               'transfer_learning')
        svcs = {}
        for layer in non_output_layers:
            svc_path = op.join(transfer_dir, f'{layer}.pkl')
            with open(svc_path, 'rb') as f:
                svcs[layer] = pkl.load(f)

    print(f'{now()} | Measuring responses for exp1 stimuli, '
          f'model: {m + 1}/{total_models} at {model_dir}, '
          f'layers: {layers}')

    # get responses to test images
    images = list_images()
    sampler = np.random.permutation(len(images))
    trials = load_trials(drop_human=True)
    new_trials = pd.DataFrame()

    # process in batches inside a function so we can free memory with gc
    def _process_batch(model, architecture, inputs, trials_batch):

        activations = get_activations(
            model, architecture, inputs, num_workers=num_procs,
            shuffle=False, batch_size=batch_size, layers=layers,
            transform=get_transform(architecture, model_dir))
        activations = insert_cycle(activations)
        trials_batch_lcs = pd.DataFrame()  # collates layers and cycles

        for layer, cycles in activations.items():
            for cycle, activs in cycles.items():

                # get a copy of the trial data and add relevant info
                trials_batch_lc = trials_batch.copy()
                trials_batch_lc['layer'] = layer
                trials_batch_lc['cycle'] = int(cycle[3:])

                # get pca and svc objects, if necessary
                kwargs = {}
                if layer != 'output':
                    kwargs['pca_object'] = svcs[layer][cycle]['pca']
                    kwargs['svc_object'] = svcs[layer][cycle]['svc']

                # get predictions
                print(f'{now()} | Generating predictions for {layer} {cycle}')
                trials_batch_lc = get_predictions(trials=trials_batch_lc,
                    activations=activs, readout_layer=layer, **kwargs)
                trials_batch_lcs = pd.concat(
                    [trials_batch_lcs, trials_batch_lc])

        return trials_batch_lcs

    # loop through batches
    superbatch_size = 2000  # reduces memory demand
    num_batches = np.ceil(len(images) / superbatch_size).astype(int)
    for b in range(num_batches):
        print(f'{now()} | Batch {b + 1}/{num_batches}')
        model = get_trained_model(model_dir, architecture, True, layers)
        first = b * superbatch_size
        last = min(first + superbatch_size, len(images))
        batch_ids = sampler[first:last]
        trials_batch = pd.concat([trials[trials.stimulus_id == f'{i:05}'
                                  ] for i in batch_ids])
        inputs = [images[i] for i in batch_ids]
        trials_batch = _process_batch(model, architecture, inputs,
                                      trials_batch)
        gc.collect()
        new_trials = pd.concat([new_trials, trials_batch])

    # reorder based on layer and cycle
    new_trials = new_trials \
        .sort_values(by=['layer', 'cycle', 'stimulus_id']) \
        .reset_index(drop=True)

    # print out accuracy for each layer and cycle
    new_trials.groupby(['layer', 'cycle']).apply(
        lambda df: print(
            f'{now()} | {df.name} accuracy: {df.accuracy.mean():.4}'))

    # reformat to combine all metric columns into long format
    new_trials = reshape_metrics(new_trials, 'long')

    # save trials
    all_trials = pd.concat([prev_trials, new_trials]).reset_index(drop=True)
    all_trials.to_parquet(out_path, index=False)

    return True


def get_predictions(trials, activations, readout_layer, pca_object=None,
                    svc_object=None):

    # check we have the right number of activations
    assert activations.shape[0] == len(trials), \
        'different number of images and activations'

    # predictions based on output layer or svc object
    if readout_layer == 'output':
        probs = special.softmax(activations[:, OBJ_CLS_IDCS], axis=1)
        classes_ordered = OBJ_CLASSES
    else:
        pca_weights = pca_object.transform(
            activations.reshape((len(trials), -1)))[:, :1000]
        probs = svc_object.predict_proba(pca_weights)
        classes_ordered = list(svc_object.classes_)
    assert (probs.sum(1).round(2) == 1).all(), 'probabilities do not sum to 1'

    # add predictions and other measures to trials
    trials['prediction'] = [classes_ordered[c] for c in probs.argmax(axis=1)]
    trials['accuracy'] = pd.Series(
        trials.prediction == trials.object_class, dtype=int)

    return trials


def fit_visibility_curves(trials):

    def _fit_curve(xvals, yvals, thr=.5):
        try:
            init_params = [max(yvals), np.median(xvals), 1, 0]
            popt, pcov = curve_fit(
                sigmoid, xvals, yvals, init_params, maxfev=int(10e5))
            curve = sigmoid(np.linspace(0, 1, 1000), *popt)
            threshold = sum(curve < thr) / 1000
        except:
            UserWarning('Curve fitting failed, returning NaNs')
            popt, threshold = [np.nan] * 4, np.nan
        return popt, threshold

    curves = pd.DataFrame()
    metric = trials.name[-1]

    vis = VISIBILITIES + [1]
    if metric != 'entropy':
        vis = [0] + vis

    # separate function for each occluder_class * occluder_color
    for occluder_class, occluder_color in itp(OCC_CLASSES, OCC_COLORS):

        yvals = (trials[
                     (trials['occluder_class'] == occluder_class) &
                     (trials['occluder_color'] == occluder_color)]
                 .groupby('visibility').mean(
            numeric_only=True).value.to_list())
        yval_mean = np.mean(yvals)

        unocc = trials[trials['visibility'] == 1].value.mean()
        yvals += [unocc]
        if metric != 'entropy':
            yvals = [1 / 8] + yvals

        # fit curve function
        popt, threshold = _fit_curve(vis, yvals)
        curves = pd.concat(
            [curves, pd.DataFrame({
                'subject': ['group'],
                'occluder_class': [occluder_class],
                'occluder_color': [occluder_color],
                'L': [popt[0]],
                'x0': [popt[1]],
                'k': [popt[2]],
                'b': [popt[3]],
                'threshold_50': [threshold],
                'mean': [yval_mean],
            })]).reset_index(drop=True)

    # single function across entire dataset
    yvals = (trials.groupby('visibility').mean(
        numeric_only=True).value.to_list())
    yval_mean = np.mean(yvals)
    if metric != 'entropy':
        yvals = [1 / 8] + yvals
    popt, threshold = _fit_curve(vis, yvals)
    curves = pd.concat(
        [curves, pd.DataFrame({
            'subject': ['group'],
            'occluder_class': ['all'],
            'occluder_color': ['all'],
            'L': [popt[0]],
            'x0': [popt[1]],
            'k': [popt[2]],
            'b': [popt[3]],
            'threshold_50': [threshold],
            'mean': [yval_mean],
        })]).reset_index(drop=True)

    return curves


def measure_human_likeness(trials_model, trials_human):

    human_likeness = pd.DataFrame()

    # collapse within each occluded condition for humans and model
    trials_h = (trials_human
        [trials_human.visibility < 1]
        .groupby(['subject', 'occluder_class', 'occluder_color'])
        .agg({'accuracy': 'mean'})
        .reset_index()
        .rename(columns={'accuracy': 'human_performance'}))
    trials_m = (trials_model
        [trials_model.visibility < 1]
        .groupby(['subject', 'occluder_class', 'occluder_color'])
        .agg({'value': 'mean'})
        .rename(columns={'value': 'model_performance'}))

    # compare individual subjects with DNN
    for subject in trials_h.subject.unique():

        # get performance profile for subject
        perf_h = (trials_h[trials_h.subject == subject]
            .groupby(['occluder_class', 'occluder_color'])
            .agg('mean', numeric_only=True)
            .reset_index())

        # get performance profile for model (using remaining group trials)
        perf_m = (trials_m[trials_m.subject != subject]
            .groupby(['occluder_class', 'occluder_color'])
            .agg('mean', numeric_only=True)
            .reset_index())

        # align subject and model performance in single dataframe
        perf_hm = perf_h.merge(perf_m, on=['occluder_class', 'occluder_color'])
        assert len(perf_hm) == 18, 'should be 18 conditions'

        # condition-wise accuracy correlation
        value = np.corrcoef(
            perf_hm.human_performance,
            perf_hm.model_performance)[0,1]
        human_likeness = pd.concat([human_likeness, pd.DataFrame(dict(
            subject=[subject],
            level=['condition-wise'],
            metric_sim=['cond_pearson_r'],
            value=[value]))])

    return human_likeness


def reshape_metrics(df, shape):
    """ Reshape dataframe based on different model performance metrics to help
    with grouping, plotting, etc. """

    if shape == 'long' and 'metric' not in df.columns:
        metrics = ['accuracy']
        df = df.melt(id_vars=[c for c in df.columns if c not in metrics],
            value_vars=metrics, var_name='metric')
    elif shape == 'wide' and 'metric' in df.columns:
        df = df.pivot(index=[c for c in df.columns if c not in [
            'metric', 'value']], columns='metric', values='value').reset_index()
    return df


def existing_results(path, layers, overwrite):

    if op.isfile(path):
        df = pd.read_parquet(path, columns=['layer'])
        if overwrite:
            return df[~df.layer.isin(layers)]
        elif all(layer in df.layer.unique() for layer in layers):
            return True
        return df
    return pd.DataFrame()


def analyse_performance(model_dir, m=0, total_models=0,
                        layers=('output'), overwrite=False, remake_plots=False):

    results_dir = op.join(MODEL_BASE, model_dir, RES_DIR)
    mod_str = f'model {m + 1}/{total_models} at {model_dir}'
    groupby = ['layer', 'cycle']

    # measure human likeness
    likeness_path = f'{results_dir}/human_likeness.parquet'
    existing_likeness = existing_results(likeness_path, layers, overwrite)
    if existing_likeness is not True:
        print(f'{now()} | Analysing human likeness (exp1) | {mod_str}')
        trials_human = load_trials(drop_human=False)
        trials_model = reshape_metrics(load_trials(model_dir=model_dir),
                                       shape='long')
        likeness = (trials_model
            .groupby(groupby)
            .apply(measure_human_likeness, trials_human)
            .reset_index(level=groupby[:-1]))
        likeness = pd.concat([existing_likeness, likeness]).reset_index(drop=True)
        likeness.to_parquet(likeness_path, index=False)

if __name__ == '__main__':
    get_responses(model_dir='alexnet/pretrained', architecture='alexnet')
    analyse_performance(model_dir='alexnet/pretrained', m=1)


