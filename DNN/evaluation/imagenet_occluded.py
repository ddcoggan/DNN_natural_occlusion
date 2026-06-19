'''
This script tests the accuracy of ANNs on ImageNet-Occluded, which is a collection of ImageNet validation sets, each
one applied with a different occluder type and visibility level in the Visual Occluders Dataset, which can be obtained
at https://github.com/ddcoggan/VisualOccludersDataset.
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
from utils import MODEL_BASE
plt.rcParams.update(custom_defaults)

np.random.seed(42)

BENCHMARK = 'ImageNet-Occluded'
VOD_PATH = '/home/david/Datasets/VisualOccludersDataset/VisualOccludersDataset'  # required (dir containing occluders, not top-level dir of repo)
IMAGENET_PATH = '/home/david/Datasets/ILSVRC2012/val'  # required
PREAPPLIED_PATH = '/home/david/Datasets/ImageNet-Occluded'  # optional
DATASETS = [i.split('Dataset/')[-1] for i in sorted(glob.glob(
        f'{VOD_PATH}/*/*'))]

def make_dataset(overwrite=False, num_procs=1):
    """ This creates and stores the ImageNet-Occluded dataset on disk, which can speed up evaluation time significantly
    at the cost of disk space (~130GB). It is optional and only recommended if you intend to evaluate many models.
    Otherwise, you can apply occlusion on the fly during evaluation. This requires no additional disk space, but each
    evaluation run will take several times longer to complete, depending on your hardware. """

    os.makedirs(PREAPPLIED_PATH, exist_ok=True)
    occs = [op.basename(i) for i in glob.glob(f'{VOD_PATH}/*')]
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
                occluder_dir=VOD_PATH)))])

        for synset in sorted(glob.glob(f'{IMAGENET_PATH}/*')):
            dataset = CustomDataset(synset, transform=transform)
            loader = DataLoader(dataset, batch_size=50, shuffle=False,
                num_workers=num_procs, multiprocessing_context='fork')
            synset_name = op.basename(synset)
            out_dir_synset = op.join(PREAPPLIED_PATH, occ, str(vis_perc), synset_name)
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


def load_scores(model_dir, overwrite=False):

    results_dir = op.join(MODEL_BASE, model_dir)
    os.makedirs(results_dir, exist_ok=True)

    out_path = f'{results_dir}/occlusion_robustness.csv'
    if not op.isfile(out_path) or overwrite:
        results = pd.DataFrame()
    else:
        results = pd.read_csv(out_path)

    return results, out_path


@torch.no_grad()
def score_model(model_dir, architecture, batch_size, m=0, total_models=0, num_procs=1,
                overwrite=False):

    """ If the dataset is not found at the stated path, occlusion will be applied on the fly. """
    preapplied = op.isdir(PREAPPLIED_PATH)
    if preapplied:
        print(f'ImageNet-Occluded found on disk at {PREAPPLIED_PATH}')
    else:
        print(f'{PREAPPLIED_PATH} does not exist, occluders will be applied on the fly during evaluation.')

    results, out_path = load_scores(
        model_dir, overwrite)

    if results.empty:
        subsets_to_run = DATASETS
    else:
        existing_results = [f'{row.occluder_type}/{int(row.visibility*100)}' for row in results.iterrows()]
        subsets_to_run = [i for i in DATASETS if i not in existing_results]

    if not len(subsets_to_run):
        return False

    print(f'{now()} | Measuring performance for ImageNet-Occluded, '
          f'model: {m + 1}/{total_models} at {model_dir}')

    model = get_trained_model(model_dir, architecture, True, ['output'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    for subset in subsets_to_run:
        occ, vis_perc = subset.split('/')
        vis = float(vis_perc)/100
        if preapplied:
            transform = get_transform(architecture, model_dir)
            dataset = ImageFolder(op.join(PREAPPLIED_PATH, subset),
                                  transform=transform)
        else:
            transform = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(float32, scale=True),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                Occlude(args=SimpleNamespace(image_size=(224, 224), Occlusion=dict(
                    form=occ, probability=1, visibility=vis, color='random',
                    occluder_dir=VOD_PATH)))])
            dataset = ImageFolder(IMAGENET_PATH, transform=transform)
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
            level_1, level_2 = subset.split('/')[-2:]
            new_results = pd.DataFrame({
                'benchmark': [BENCHMARK],
                'cycle': [int(cycle[3:])],
                'occluder_type': [level_1],
                'visibility': [str(vis)[:3]],
                'metric': ['accuracy'],
                'score': [perf.avg_epoch]
            })
            results = pd.concat([results, new_results]).reset_index(drop=True)
            print(f'{cycle} accuracy: {perf.avg_epoch:.4f}')

        results.to_csv(out_path, index=False)

    return True

if __name__ == '__main__':

    """ Demo of an existing model for reproducing results """

    # backup existing scores
    os.rename('../models/original/alexnet/no_occlusion/occlusion_robustness.csv',
              '../models/original/alexnet/no_occlusion/occlusion_robustness_bak.csv')
    # reproduce scores
    score_model(model_dir='original/alexnet/no_occlusion', architecture='alexnet', num_procs=8, batch_size=64)
    # average score on imagenet-occluded should be 0.03824...
    # scores may vary slightly due to random aspects of object / occluder pairings, particularly if occluders were
    # applied on the fly during eval
    demo_scores = pd.read_csv('../models/original/alexnet/no_occlusion/occlusion_robustness.csv')
    print(demo_scores[demo_scores.benchmark == 'ImageNet-Occluded'].score.mean())

    """
    # demo of a pretrained model not included in this paper
    import torchvision
    #make_dataset() # uncomment this if you wish to pre-apply occluders and save dataset to disk
    weights_path = '../models/alexnet/pretrained_IMAGENET1K_V1/weights.pt'
    if not op.isfile(weights_path):
        os.makedirs(op.dirname(weights_path), exist_ok=True)
        torch.hub.download_url_to_file(torchvision.models.AlexNet_Weights.IMAGENET1K_V1.url,
                                   weights_path)
    score_model(model_dir='alexnet/pretrained_IMAGENET1K_V1', architecture='alexnet', num_procs=8, batch_size=64)
    """