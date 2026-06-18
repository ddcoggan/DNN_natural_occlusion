'''
This script tests the accuracy of ANNs on public benchmarks
'''

import os
import os.path as op
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from types import SimpleNamespace
from itertools import product as itp
from tqdm import tqdm
from joblib import Parallel, delayed
import torch
import torchvision.transforms.v2 as transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import float32
from utils import now, insert_cycle
from utils.accuracy import accuracy
from utils.AverageMeter import AverageMeter
from utils.get_trained_model import get_trained_model
from utils.plot_utils import custom_defaults
from utils.Occlude import Occlude
from utils.CustomDataset import CustomDataset
from utils.get_transform import get_transform
from utils.load_benchmark_scores import load_benchmark_scores
plt.rcParams.update(custom_defaults)

np.random.seed(42)

BENCHMARK = 'ImageNet-Occluded'
DATASET_BASE = '/Users/david/Datasets/ImageNet-Occluded'
DATASETS = [i[len(DATASET_BASE) + 1:] for i in sorted(glob.glob(
            f'{DATASET_BASE}/*/*'))]


def make_dataset(overwrite=False, num_procs=1):
    """ This creates the ImageNet-Occluded dataset, i.e., for each occluder
    type and visibility level in the Visual Occluders Dataset, it creates a
    version of the ImageNet validation set with those occluders randomly
    applied. This will take up a lot of disk space, but substantially
    improves evaluation speed once generated, and so is recommended if you
    intend to evaluate many models on the entire dataset. Alternatively,
    you can choose to apply occlusion on the fly during evaluation, but this
    will be much slower. """

    vod_dir = '/Users/david/PycharmProjects/VisualOccludersDataset'
    imagenet_dir = '/Users/david/PycharmProjects/ILSVRC2012/val'
    out_dir = '/Users/david/PycharmProjects/ImageNet-Occluded'
    os.makedirs(out_dir, exist_ok=True)
    occs = [op.basename(i) for i in glob.glob(f'{vod_dir}/*')]
    viss = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def _make_dataset(occ, vis):
        vis_perc = int(vis * 100)

        transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(float32, scale=True),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            Occlude(args=SimpleNamespace(image_size=(224, 224), Occlusion=dict(
                form=occ, probability=1, visibility=vis, color='random',
                occluder_dir=vod_dir)))])

        for synset in sorted(glob.glob(f'{imagenet_dir}/*')):
            dataset = CustomDataset(synset, transform=transform)
            loader = DataLoader(dataset, batch_size=50, shuffle=False,
                num_workers=num_procs, multiprocessing_context='fork')
            synset_name = op.basename(synset)
            out_dir_synset = op.join(out_dir, occ, str(vis_perc), synset_name)
            os.makedirs(out_dir_synset, exist_ok=True)
            in_images = sorted(glob.glob(f'{synset}/*'))
            image_names = [op.basename(i) for i in in_images]
            out_images = [op.join(out_dir_synset, i) for i in image_names]
            if not all([op.isfile(i) for i in out_images]) or overwrite:
                print(f'{now()} | {occ}, {vis_perc}%, {synset_name}')
                for images in loader:
                    for i, image in enumerate(images):
                        save_image(image, out_images[i])

    Parallel(n_jobs=num_procs)(
        delayed(_make_dataset)(occ, vis) for occ, vis in itp(occs, viss)
    )


@torch.no_grad()
def score_model(model_dir, architecture, batch_size, m=0, total_models=0, num_procs=1,
                overwrite=False):

    results, out_path = load_benchmark_scores(
        model_dir, BENCHMARK, overwrite)

    if results.empty:
        subsets_to_run = DATASETS
    else:
        subsets_to_run = [i for i in DATASETS if i not in results.path.unique()]

    if not len(subsets_to_run):
        return False

    print(f'{now()} | Measuring performance for ImageNet-Occluded, '
          f'model: {m + 1}/{total_models} at {model_dir}')

    model = get_trained_model(model_dir, architecture, True, ['output'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    transform = get_transform(architecture, model_dir)

    for subset in subsets_to_run:
        occ, vis_perc = subset.split('/')
        vis = float(vis_perc)/100
        dataset = ImageFolder(op.join(DATASET_BASE, subset),
                              transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_procs)

        # loop through batches
        with tqdm(loader, unit=f"batch({batch_size})") as tepoch:

            for batch, (inputs, targets) in enumerate(tepoch):

                tepoch.set_description(f'{now()} | {subset}')

                # put inputs on device
                inputs = inputs.to(device)
                targets = targets.to(device)

                # pass through model with automatic mixed precision for speed
                with torch.autocast(device_type=device.type,
                                    dtype=torch.float16):
                    outputs = model(inputs)
                outputs = insert_cycle(outputs, batch_size=inputs.shape[0])

                # calculate accuracy
                if batch == 0:
                    performance = {k: AverageMeter() for k in outputs}
                for cycle, output in outputs.items():
                    acc = accuracy(output, targets, (1,)
                                   )[0].detach().cpu().item()
                    performance[cycle].update(acc)

                # print last and mean accuracy of final cycle
                tepoch.set_postfix_str(
                    f'acc1: {acc:.4f}({performance[cycle].avg_epoch:.4f})')

        # save results
        for cycle, perf in performance.items():
            level_1, level_2 = subset.split(f'{DATASET_BASE}/')[-1].split('/')
            new_results = pd.DataFrame({
                'benchmark': [BENCHMARK],
                'path': [subset],
                'cycle': [int(cycle[3:])],
                'level_1': [level_1],
                'level_2': [str(vis)[:3]],
                'metric': ['accuracy'],
                'score': [perf.avg_epoch]
            })
            results = pd.concat([results, new_results]).reset_index(drop=True)
            print(f'{cycle} accuracy: {perf.avg_epoch:.4f}')

        results.to_csv(out_path, index=False)

    return True

