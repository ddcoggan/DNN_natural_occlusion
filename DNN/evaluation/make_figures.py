import itertools
import os
import os.path as op
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
tab20 = cm.tab20.colors
from PIL import Image
import numpy as np
from itertools import product as itp
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
from types import SimpleNamespace
import torch
from torch.utils.data import default_collate
import torchvision.transforms.v2 as transforms
import sys
import shutil
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pingouin as pg
from statsmodels.stats.anova import AnovaRM
from humans.analysis import CFG
from DNN.evaluation.utils.Occlude import Occlude
from DNN.evaluation.utils.Noise import Noise
from utils.math_functions import sigmoid
from utils.plot_utils import make_legend
from utils.image_processing import tile
from utils.plot_utils import custom_defaults

plt.rcParams.update(custom_defaults)
plt.rcParams.update({
    'font.size': 8,
    'lines.linewidth': .3,
    'axes.linewidth': .3,
    'xtick.major.width': .3,
    'ytick.major.width': .3,
    'grid.linewidth': .3,
    'ytick.major.size': 2,
    'figure.dpi': 300,
})

def main():

    #image_samples()
    #robustness()
    #human_likeness()
    #performance_equated()
    #occlusion_strength()
    #other_occluders()
    other_distortions()


# useful variables
architectures = {
    'resnet101': 'ResNet101',
    'cornet_s_plus': 'CORnet-S+',
    'efficientnet_b1': 'EfficientNet-B1',
    'alexnet': 'AlexNet',
    'vit_b_16': 'ViT-B/16',
}
training_augmentations = ['no_occlusion', 'natural',
    'natural_silhouette', 'artificial_1', 'artificial_2',
]

# datasets
imagenet_dir = '/Users/david/data/datasets/images/ILSVRC2012/train'
vod = op.expanduser(f'~/PycharmProjects/VisualOccludersDataset'
                    f'/VisualOccludersDataset')
visibilities = np.arange(10, 100, 10)

occluder_sets = {
    'natural': ['natural'],
    'natural_mix': ['natural', 'natural_silhouette'],
    'natural_silhouette': ['natural_silhouette'],
    'artificial_1': [
        'horizontal_bars_04',
        'vertical_bars_04',
        'oblique_bars_04',
        'cardinal_crossed_bars',
        'oblique_crossed_bars',
        'polkadot',
        'polkasquare',
        'mud_splash'],
    'artificial_2': [
        'curved_lines',
        'straight_ines',
        'empty_rectangles',
        'filled_rectangles',
        'empty_triangles',
        'filled_triangles',
        'empty_ellipses',
        'filled_ellipses'],
}

colors_aug = {
    'humans': {
        **{'linestyle': 'dashed', 'color_light': tab20[15]},
        **{i: tab20[14] for i in ['color', 'edgecolor', 'linecolor']}},
    'no_occlusion': {
        'color': 'w', 'edgecolor': 'k', 'linecolor': 'k', 'color_light': (
            .25, .25, .25), 'linestyle': 'solid'},
    'natural': {
        **{'linestyle': 'solid', 'color_light': tab20[5]},
        **{i: 'tab:green' for i in ['color', 'edgecolor', 'linecolor']}},
    'natural_silhouette': {**{'linestyle': 'solid', 'color_light': tab20[11]},
        **{i: 'tab:brown' for i in ['color', 'edgecolor', 'linecolor']}},
    'artificial_1': {**{'linestyle': 'solid', 'color_light': tab20[1]},
        **{i: 'tab:blue' for i in ['color', 'edgecolor', 'linecolor']}},
    'artificial_2': {**{'linestyle': 'solid', 'color_light': tab20[7]},
        **{i: 'tab:red' for i in ['color', 'edgecolor', 'linecolor']}},
    'cutmix': {**{'linestyle': 'dashed'}, **{
        i: 'tab:olive' for i in ['color', 'edgecolor', 'linecolor']}},
    'mixup': {**{'linestyle': 'dashed'}, **{
        i: 'tab:cyan' for i in ['color', 'edgecolor', 'linecolor']}},
    'randomerase_0': {**{'linestyle': 'dashed'}, **{
        i: 'tab:purple' for i in ['color', 'edgecolor', 'linecolor']}},
    'randomerase_random': {**{'linestyle': 'dashed'}, **{
        i: 'tab:pink' for i in ['color', 'edgecolor', 'linecolor']}},
}

colors_arch = {
    'humans': 'tab:grey',
    'cornet_s_plus__classification': 'tab:red',
    'cornet_s_plus__simclr': 'tab:pink',
    'resnet101__classification': 'tab:blue',
    'efficientnet_b1__classification': 'tab:green',
    'vit_b_16__classification': 'tab:orange',
}

occluders = [
    'natural',
    'natural_silhouette',
    'mud_splash',
    'patch_drop',
    'cardinal_crossed_bars',
    'oblique_crossed_bars',
    'polkadot',
    'polkasquare',

    'curved_lines',
    'straight_lines',
    'filled_ellipses',
    'empty_ellipses',
    'filled_triangles',
    'empty_triangles',
    'filled_rectangles',
    'empty_rectangles',

    'coarse_noise',
    'fine_noise',
    'fine_oriented_noise',
    'pink_noise',
    'oblique_bars_02',
    'oblique_bars_04',
    'oblique_bars_08',
    'oblique_bars_16',

    'horizontal_bars_02',
    'horizontal_bars_04',
    'horizontal_bars_08',
    'horizontal_bars_16',
    'vertical_bars_02',
    'vertical_bars_04',
    'vertical_bars_08',
    'vertical_bars_16',
]


def image_samples():

    out_dir = 'figures/image_samples'
    os.makedirs(out_dir, exist_ok=True)

    # list random objects

    imagenet_classes = sorted(os.listdir(imagenet_dir))
    np.random.seed(8)
    random_classes = np.random.choice(imagenet_classes, size=6, replace=False)
    input_paths = [glob.glob(f'{imagenet_dir}/{c}/*.jpg')[0] for c in
                   random_classes]
    input_images = [Image.open(p).convert('RGB') for p in input_paths]

    # use identical base xforms to help visualize occlusion effects
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.RandomResizedCrop(224, scale=(0.8, 1), antialias=True),
        transforms.RandomHorizontalFlip()])
    xformed_images = [transform(img) for img in input_images]

    for a, aug in enumerate(training_augmentations):
        transform2 = []
        if aug != 'no_occlusion':
            transform2.append(Occlude(SimpleNamespace(
                image_size=224,
                Occlusion={
                    "color": "random",
                    "form": occluder_sets[aug],
                    "occluder_dir": vod,
                    "probability": 1,
                    "views": [0],
                    "visibility": [0.5, 0.6, 0.7, 0.8, 0.9]
                },
                num_views=1
            )))
        transform2 = transforms.Compose(transform2 + [transforms.ToPILImage()])

        keep = {'no_occlusion': 0, 'natural': 13, 'natural_silhouette': 5,
            'artificial_1': 2, 'artificial_2': 9}
        for j in range(30):
            final_images = []
            for i, img in enumerate(xformed_images):
                final_images.append(transform2(img.clone()))
            if j == keep[aug]:
                tile(final_images, num_cols=2, out_path=f'{out_dir}/{aug}.png',
                    rowgap=16, colgap=16, rowgapfreq=1, colgapfreq=1)



    # other augmentations

    cutmix = transforms.CutMix(alpha=1.0, num_classes=1000)
    labels = torch.arange(len(xformed_images))
    cutmixed_images, mixed_labels = cutmix(default_collate(xformed_images),
        labels)
    final_images = [transforms.ToPILImage()(i) for i in cutmixed_images]
    final_images[1].save(f'{out_dir}/cutmix.png')

    re = transforms.RandomErasing(value=0, inplace=True)
    re_images = [transforms.ToPILImage()(re(img.clone())) for img in xformed_images]
    re_images[1].save(f'{out_dir}/randomerase_0.png')

    re = transforms.RandomErasing(value='random', inplace=True)
    re_images = [transforms.ToPILImage()(re(img.clone())) for img in xformed_images]
    re_images[1].save(f'{out_dir}/randomerase_rand.png')

    blur = transforms.GaussianBlur(kernel_size=51, sigma=5)
    blur_images = [transforms.ToPILImage()(blur(img.clone())) for img in xformed_images]
    blur_images[1].save(f'{out_dir}/blur.png')

    noise_g = Noise('gaussian', [0.5, 0.6], [1, 1])
    noise_images = [transforms.ToPILImage()(noise_g(img.clone())) for img in xformed_images]
    noise_images[1].save(f'{out_dir}/noise_gaussian.png')

    noise_f = Noise('fourier', [0.5, 0.6], [1, 1],
        '/Users/david/data/datasets/images/ImageNet-Noise'
        '/fourier/mean_magnitude.pt')
    noise_images = [transforms.ToPILImage()(noise_f(img.clone())) for img in
                    xformed_images]
    noise_images[1].save(f'{out_dir}/noise_fourier.png')


def robustness():

    # collate data
    robustness = pd.DataFrame()
    for arch, aug in itp(architectures, training_augmentations):
        df = pd.read_csv(f'../models/original/{arch}/'
                         f'{aug}/occlusion_robustness.csv')
        df['architecture'] = arch
        df['training_augmentation'] = aug
        robustness = pd.concat((robustness, df), ignore_index=True)
    robustness.score = robustness.score.astype(float)


    out_dir = 'figures/occlusion_robustness'
    os.makedirs(out_dir, exist_ok=True)

    tests = training_augmentations[1:]

    # get name of occluder set for each test occluder
    rob = robustness[
        robustness.benchmark.isin(['ImageNet-1K', 'ImageNet-Occluded'])
    ].copy()
    test_set_list = []
    for i, row in rob.iterrows():
        if row.benchmark in ['ImageNet-1K']:
            test_set = 'no_occlusion'
        else:
            for occluder_set in tests:
                occluder_types = occluder_sets[occluder_set]
                if row.occluder_type in occluder_types:
                    test_set = occluder_set
                    break
            else:
                test_set = 'unused'
        test_set_list.append(test_set)
    rob['test_set'] = test_set_list
    rob.visibility = rob.visibility.fillna(1.)
    rob.visibility = rob.visibility.astype(float)

    # one plot per training occluder type (visibility - accuracy curves)
    rob_vis = (rob.groupby(
        ['benchmark', 'training_augmentation', 'test_set', 'visibility'],
        dropna=False).agg({'score': 'mean'}).reset_index())

    for aug in training_augmentations:

        out_path = op.join(out_dir, f'{aug}.pdf')
        fig, ax = plt.subplots(figsize=(2.5, 2.5))

        # overall accuracies inset in top left
        sub_ax = inset_axes(
            parent_axes=ax,
            width='40%',
            height='40%',
            borderpad=1,  # padding between parent and inset axes
            bbox_to_anchor=(.07, -0.02, 1, 1),
            bbox_transform=ax.transAxes,
            loc='upper left')

        for te, test_set in enumerate(tests):
            if test_set == 'artificial_3':
                color = edgecolor = linecolor = 'tab:gray'
            else:
                color, edgecolor, linecolor = [
                    colors_aug[test_set][k] for k in [
                        'color', 'edgecolor', 'linecolor']]

            # plot data points
            xvals = np.arange(0.1, 1.1, 0.1)
            yvals = (rob_vis[
                 (rob_vis.training_augmentation == aug) &
                 (rob_vis.test_set.isin([test_set, 'no_occlusion']))]
                 .sort_values(by='visibility').score.values)
            ax.scatter(xvals, yvals, s=32, clip_on=False,
                color=[color] * 9 + ['w'],  edgecolor=[edgecolor] * 9 + [
                    'k'], marker='o', zorder=13-(te * 2 + 2))

            # fit and plot curve
            init_params = [max(yvals), np.median(xvals), 1, 0]
            popt, pcov = curve_fit(
                sigmoid, xvals, yvals, init_params, maxfev=int(10e5))
            curve_x = np.linspace(0, 1, 1000)
            curve_y = sigmoid(curve_x, *popt)
            ax.plot(curve_x, curve_y, color=linecolor,
                zorder=13-(te * 2 + 3))

            # mean accuracy inset
            sub_ax.bar(te, np.mean(yvals[:-1]), color=color,
                       edgecolor=edgecolor)

        # format inset plot
        sub_ax.set_yticks((0, .1, .2, .3, .4), size=7,
                          labels=('0', '.1', '.2', '.3', '.4'))
        sub_ax.set_ylim((0, .4))
        sub_ax.set_xlim(-.8, 4.5)
        sub_ax.set_xticks([])
        sub_ax.tick_params(axis='x', which='both', length=0, pad=-2)
        sub_ax.set_title('mean accuracy', fontsize=7)
        # sub_ax.set_xlabel('training occluder strength', fontsize=7)

        # format main plot
        ax.grid(axis='both', linestyle='solid', alpha=.25, zorder=-1,
                clip_on=False)
        ax.set_xticks(np.arange(0, 1.1, .2))
        ax.set_xlim((0, 1))
        ax.set_yticks(np.arange(0, 1.1, .2))
        ax.set_ylim((0, 1))
        ax.tick_params(axis='both', which='major', labelsize=7,
                       zorder=-1)
        # ax.axhline(y=1/1000, color='k', ls='dotted')
        ax.set_xlabel('visibility')
        ax.set_ylabel('accuracy')
        fig.tight_layout()
        plt.savefig(out_path)
        plt.close()

    # alternative plot: bar plots of cross-performance

    """ test artificial occlusion-trained weights on natural occluders """
    fig, ax = plt.subplots(figsize=(3.6, 2.5))

    # pooled architectures
    for xpos, aug in enumerate(['artificial_1', 'artificial_2']):
        color = colors_aug[aug]['color']
        edgecolor = colors_aug[aug]['edgecolor']
        points = (rob[(rob.training_augmentation == aug) & (
                    rob.test_set == 'natural')].groupby(
            'architecture').score.mean())
        ax.bar(xpos+6, points.mean(), color=color, edgecolor=edgecolor,
            linewidth=1, zorder=2)
        ax.text(xpos+6, 0.005, aug.replace('_', ' '), ha='center',
            va='bottom', rotation=90, size=6, color='w', zorder=4)
        #sns.stripplot(x=xpos, y=points, zorder=3, clip_on=False,
        #    native_scale=True, dodge=True, ax=ax, color='w', size=4,
        #    linewidth=0.5, edgecolor='k')

    # upper bound
    nat_score = (rob[
        (rob.training_augmentation == 'natural') &
        (rob.test_set == 'natural')].score.mean())
    nat_col = colors_aug['natural']['edgecolor']
    ax.axhline(y=nat_score, color=nat_col, xmax=.36, zorder=1, ls='dashed')
    #ax.text(3.5, nat_score+.01, 'natural occlusion training', size=8,
    #    color=nat_col, ha='right', fontstyle='italic')

    # lower bound
    unocc_score = (rob[
        (rob.training_augmentation == 'no_occlusion') &
        (rob.test_set == 'natural')].score.mean())
    unocc_col = colors_aug['no_occlusion']['edgecolor']
    ax.axhline(y=unocc_score, xmax=.36, color=unocc_col, zorder=3, \
        ls='dashed')
    #ax.text(3.5, unocc_score + .01, 'no occlusion training', size=8,
    #    color=unocc_col, ha='right', fontstyle='italic')

    # individual architectures
    xticks = [7]
    for a, (arch, arch_label) in enumerate(architectures.items()):
        x_offset = a * 3 + 10
        xticks.append(x_offset + 1)
        rob_model = rob[rob.architecture == arch]
        for xpos, aug in enumerate(['artificial_1', 'artificial_2']):
            color = colors_aug[aug]['color']
            edgecolor = colors_aug[aug]['edgecolor']
            yval = rob_model[
                (rob_model.training_augmentation == aug) &
                (rob_model.test_set == 'natural')].score.mean()
            ax.bar(xpos + x_offset, yval, color=color, edgecolor=edgecolor,
                linewidth=1, zorder=2)

        # upper bound
        x_min = (x_offset + .3) / 24.7
        x_max = (x_offset + 2.7) / 24.7
        nat_score = (rob_model[
            (rob_model.training_augmentation == 'natural') &
            (rob_model.test_set == 'natural')].score.mean())
        nat_col = colors_aug['natural']['edgecolor']
        ax.axhline(y=nat_score, xmin=x_min, xmax=x_max,
            color=nat_col, zorder=1, ls='dashed')
        # ax.text(3.5, nat_score+.01, 'natural occlusion training', size=8,
        #    color=nat_col, ha='right', fontstyle='italic')

        # lower bound
        unocc_score = (rob_model[
            (rob_model.training_augmentation == 'no_occlusion') &
            (rob_model.test_set == 'natural')].score.mean())
        unocc_col = colors_aug['no_occlusion']['edgecolor']
        ax.axhline(y=unocc_score, xmin=x_min, xmax=x_max,
            color=unocc_col, zorder=3, ls='dashed')
        # ax.text(3.5, unocc_score + .01, 'no occlusion training', size=8,
        #    color=unocc_col, ha='right', fontstyle='italic')

    # format
    ax.set_ylabel('Top-1 accuracy')
    yticks = np.arange(0, .6, .1)
    ax.set_yticks(yticks, labels=[f'{i:.1f}' for i in yticks], size=6)
    ax.grid(axis='y', linestyle='solid', alpha=.25, zorder=0,
        clip_on=False)
    ax.set_ylim(0, .5)
    ax.set_xticks(xticks, rotation=45, ha='right', va='top',
        labels=['All DNNs'] + list(architectures.values()))
    ax.tick_params(axis='x', which='major', pad=0)
    ax.set_xlim((-1, 23.7))

    #fig.suptitle('train: artificial\ttest: natural', size=12)
    plt.tight_layout()
    #plt.subplots_adjust()
    fig.savefig(op.join(out_dir, 'gen_art_to_nat.pdf'))
    fig.savefig(op.join(out_dir, 'gen_art_to_nat.png'))
    plt.close()

    make_legend(
        outpath=op.join(out_dir, 'gen_art_to_nat_legend.png'),
        labels=['artificial 1', 'artificial 2', 'natural (upper bound)',
                'no occlusion (lower bound)'],
        markers=['s', 's', "None", "None"],
        colors=[colors_aug[aug]['edgecolor'] for aug in [
            'artificial_1', 'artificial_2', 'natural', 'no_occlusion']],
        markeredgecolors=None,
        linestyles=["None", "None", 'dashed', 'dashed'])

    """ test natural occlusion-trained weights on artificial occluders """
    for test_aug in ['artificial_1', 'artificial_2']:
        fig, ax = plt.subplots(figsize=(2.2, 2.5))

        # pooled architectures
        color = edgecolor = colors_aug['natural']['color']
        points = (rob[(rob.training_augmentation == 'natural') & (
                rob.test_set == test_aug)].groupby('architecture').score.mean())
        ax.bar(0, points.mean(), color=color, edgecolor=edgecolor,
            linewidth=1, zorder=2)
        ax.text(0, 0.005, 'natural', ha='center',
            va='bottom', rotation=90, size=6, color='w', zorder=4)

        # upper bound
        art_score = (rob[(rob.training_augmentation == test_aug) & (
                    rob.test_set == test_aug)].score.mean())
        art_col = colors_aug[test_aug]['edgecolor']
        ax.axhline(y=art_score, color=art_col, xmax=.15, zorder=1,
            ls='dashed')
        # ax.text(3.5, nat_score+.01, 'natural occlusion training', size=8,
        #    color=nat_col, ha='right', fontstyle='italic')

        # lower bound
        unocc_score = (rob[(rob.training_augmentation == 'no_occlusion') & (
                    rob.test_set == test_aug)].score.mean())
        unocc_col = colors_aug['no_occlusion']['edgecolor']
        ax.axhline(y=unocc_score, xmax=.15, color=unocc_col, zorder=3, \
            ls='dashed')
        # ax.text(3.5, unocc_score + .01, 'no occlusion training', size=8,
        #    color=unocc_col, ha='right', fontstyle='italic')

        # individual architectures
        xticks = [.5]
        for a, (arch, arch_label) in enumerate(architectures.items()):
            x_pos = a * 2 + 4
            xticks.append(x_pos+.5)
            rob_model = rob[rob.architecture == arch]
            color = edgecolor = colors_aug['natural']['color']
            yval = rob_model[(rob_model.training_augmentation ==
                              'natural') & (
                        rob_model.test_set == test_aug)].score.mean()
            ax.bar(x_pos, yval, color=color, edgecolor=edgecolor,
                linewidth=1, zorder=2)

            # upper bound
            x_min = (x_pos +.3) / 13.7
            x_max = (x_pos + 1.7) / 13.7
            art_score = (rob_model[
                             (rob_model.training_augmentation == test_aug)
                             & ( rob_model.test_set ==
                                 test_aug)].score.mean())
            art_col = colors_aug[test_aug]['edgecolor']
            ax.axhline(y=art_score, xmin=x_min, xmax=x_max, color=art_col,
                zorder=1, ls='dashed')
            # ax.text(3.5, nat_score+.01, 'natural occlusion training', size=8,
            #    color=nat_col, ha='right', fontstyle='italic')

            # lower bound
            unocc_score = (rob_model[(
                rob_model.training_augmentation == 'no_occlusion') & (
                rob_model.test_set == test_aug)].score.mean())
            unocc_col = colors_aug['no_occlusion']['edgecolor']
            ax.axhline(y=unocc_score, xmin=x_min, xmax=x_max, color=unocc_col,
                zorder=3, ls='dashed')

        # format
        ax.set_ylabel('Top-1 accuracy')
        yticks = np.arange(0, .6, .1)
        ax.set_yticks(yticks, labels=[f'{i:.1f}' for i in yticks], size=6)
        ax.grid(axis='y', linestyle='solid', alpha=.25, zorder=0, clip_on=False)
        ax.set_ylim(0, .5)
        ax.set_xticks(xticks, rotation=45, ha='right', va='top',
            labels=['All DNNs'] + list(architectures.values()))
        ax.tick_params(axis='x', which='major', pad=0, length=0)
        ax.set_xlim((-1, 12.7))

        # fig.suptitle('train: artificial\ttest: natural', size=12)
        plt.tight_layout()
        # plt.subplots_adjust()
        fig.savefig(op.join(out_dir, f'gen_nat_to_{test_aug}.pdf'))
        fig.savefig(op.join(out_dir, f'gen_nat_to_{test_aug}.png'))
        plt.close()

    """ cross-generalization between natural and natural silhouette """
    for test_aug, train_aug in zip(['natural', 'natural_silhouette'],
                ['natural_silhouette', 'natural']):
        fig, ax = plt.subplots(figsize=(2.2, 2.5))

        # pooled architectures
        color = edgecolor = colors_aug[train_aug]['color']
        points = (rob[(rob.training_augmentation == train_aug) & (
                rob.test_set == test_aug)].groupby(
            'architecture').score.mean())
        ax.bar(0, points.mean(), color=color, edgecolor=edgecolor,
            linewidth=1, zorder=2)
        aug_label = train_aug.replace('_silhouette', ' sil.')
        ax.text(0, 0.005, aug_label, ha='center', va='bottom',
            rotation=90, size=6, color='w', zorder=4)

        # upper bound
        art_score = (rob[(rob.training_augmentation == test_aug) & (
                rob.test_set == test_aug)].score.mean())
        art_col = colors_aug[test_aug]['edgecolor']
        ax.axhline(y=art_score, color=art_col, xmax=.15, zorder=1,
            ls='dashed')
        # ax.text(3.5, nat_score+.01, 'natural occlusion training', size=8,
        #    color=nat_col, ha='right', fontstyle='italic')

        # lower bound
        unocc_score = (rob[(
                                       rob.training_augmentation == 'no_occlusion') & (
                                   rob.test_set == test_aug)].score.mean())
        unocc_col = colors_aug['no_occlusion']['edgecolor']
        ax.axhline(y=unocc_score, xmax=.15, color=unocc_col, zorder=3, \
            ls='dashed')
        # ax.text(3.5, unocc_score + .01, 'no occlusion training', size=8,
        #    color=unocc_col, ha='right', fontstyle='italic')

        # individual architectures
        xticks = [.5]
        for a, (arch, arch_label) in enumerate(architectures.items()):
            x_pos = a * 2 + 4
            xticks.append(x_pos + .5)
            rob_model = rob[rob.architecture == arch]
            color = edgecolor = colors_aug[train_aug]['color']
            yval = rob_model[
                (rob_model.training_augmentation == train_aug) & (
                        rob_model.test_set == test_aug)].score.mean()
            ax.bar(x_pos, yval, color=color, edgecolor=edgecolor,
                linewidth=1, zorder=2)

            # upper bound
            x_min = (x_pos + .3) / 13.7
            x_max = (x_pos + 1.7) / 13.7
            art_score = (rob_model[
                (rob_model.training_augmentation == test_aug) &
                (rob_model.test_set == test_aug)].score.mean())
            art_col = colors_aug[test_aug]['edgecolor']
            ax.axhline(y=art_score, xmin=x_min, xmax=x_max,
                color=art_col, zorder=1, ls='dashed')
            # ax.text(3.5, nat_score+.01, 'natural occlusion training', size=8,
            #    color=nat_col, ha='right', fontstyle='italic')

            # lower bound
            unocc_score = (rob_model[(
                                                 rob_model.training_augmentation == 'no_occlusion') & (
                                                 rob_model.test_set == test_aug)].score.mean())
            unocc_col = colors_aug['no_occlusion']['edgecolor']
            ax.axhline(y=unocc_score, xmin=x_min, xmax=x_max,
                color=unocc_col, zorder=3, ls='dashed')

        # format
        ax.set_ylabel('Top-1 accuracy')
        yticks = np.arange(0, .6, .1)
        ax.set_yticks(yticks, labels=[f'{i:.1f}' for i in yticks],
            size=6)
        ax.grid(axis='y', linestyle='solid', alpha=.25, zorder=0,
            clip_on=False)
        ax.set_ylim(0, .5)
        ax.set_xticks(xticks, rotation=45, ha='right', va='top',
            labels=['All DNNs'] + list(architectures.values()))
        ax.tick_params(axis='x', which='major', pad=0, length=0)
        ax.set_xlim((-1, 12.7))

        # fig.suptitle('train: artificial\ttest: natural', size=12)
        plt.tight_layout()
        # plt.subplots_adjust()
        fig.savefig(
            op.join(out_dir, f'gen_{train_aug}_to_{test_aug}.pdf'))
        fig.savefig(
            op.join(out_dir, f'gen_{train_aug}_to_{test_aug}.png'))
        plt.close()


    """ alternative plot 2: cross-generalization matrix """

    box_color = (0,.8,0)#'tab:green'#'tab:red'
    cmap = 'inferno'
    # pooled architectures
    for score_type in ['raw', 'norm', 'base']:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))

        plot_data = (rob
            .groupby(['training_augmentation', 'test_set'], observed=True)
            .agg({'score': 'mean'}).reset_index()
            .pivot(index='test_set', columns='training_augmentation',
            values='score')
            .drop(index=['unused'])
            )
        plot_data.index = pd.Categorical(plot_data.index,
            categories=training_augmentations, ordered=True)
        plot_data = plot_data.sort_index()
        plot_data.columns = pd.Categorical(plot_data.columns,
            categories=training_augmentations, ordered=True)
        plot_data = plot_data.sort_index(axis=1)
        if score_type != 'raw':
            plot_data = plot_data.drop(index=['no_occlusion'])
            for ind in plot_data.index:
                plot_data.loc[ind, :] -= plot_data.loc[
                    ind, 'no_occlusion']
                if score_type == 'norm':
                    plot_data.loc[ind, :] /= plot_data.loc[ind, ind]
        im = ax.imshow(plot_data, cmap=cmap, vmin=0,
            vmax=plot_data.max().max())
        ax.tick_params(**{'length': 0})
        ax.set_xticks(range(len(plot_data.columns)),
            labels=[i.replace('_', ' ').capitalize() for i in plot_data.columns],
            ha='right', va='top', rotation=45, size=7)
        ax.set_xlabel('Train', size=9)
        ax.set_yticks(np.arange(len(plot_data.index)),
            labels=[i.replace('_', ' ').capitalize() for i in
                    plot_data.index], size=7)
        ax.set_ylabel('Test', size=9)
        ax.tick_params(direction='in')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        for (tr, training_set), (te, test_set) in itp(
                enumerate(plot_data.index), enumerate(plot_data.columns)):
            value = plot_data.iloc[tr, te]
            text_col = 'w' if value < .5 else 'k'
            fmt = 'bold' if value == plot_data.iloc[:, te].max() else 'normal'
            ax.text(te, tr, f'{value:.2f}'.replace('0.', '.'), weight=fmt,
                ha='center', va='center', color=text_col, size=9)
        for training_set in plot_data.index:
            box_x = plot_data.columns.get_loc(training_set) - .5
            box_y = plot_data.index.get_loc(training_set) - .5
            box = patches.Rectangle((box_x, box_y), 1, 1, linewidth=1,
                edgecolor=box_color, facecolor='none', clip_on=False, zorder=3)
            ax.add_patch(box)
        plt.tight_layout()
        for ft in ['png', 'pdf']:
            plt.savefig(op.join(
                out_dir, f'cross_generalization_matrix_{score_type}.{ft}'))
        plt.close()

    # individual architectures
    for score_type in ['raw', 'norm', 'base']:
        fig, axes = plt.subplots(ncols=len(architectures),
            figsize=(7.5, 2))
        for a, arch in enumerate(architectures):
            ax = axes[a]
            plot_data = (rob[rob.architecture == arch].groupby(
                ['training_augmentation', 'test_set'], observed=True).agg(
                {'score': 'mean'}).reset_index().pivot(
                index='test_set', columns='training_augmentation',
                values='score').drop(index=['unused']))
            plot_data.index = pd.Categorical(plot_data.index,
                categories=training_augmentations, ordered=True)
            plot_data = plot_data.sort_index()
            plot_data.columns = pd.Categorical(plot_data.columns,
                categories=training_augmentations, ordered=True)
            plot_data = plot_data.sort_index(axis=1)
            if score_type != 'raw':
                plot_data = plot_data.drop(index=['no_occlusion'])
                for ind in plot_data.index:
                    plot_data.loc[ind, :] -= plot_data.loc[
                        ind, 'no_occlusion']
                    if score_type == 'norm':
                        plot_data.loc[ind, :] /= plot_data.loc[ind, ind]
            vmax = {'raw': .72, 'norm': 1, 'base': .37}[score_type]
            im = ax.imshow(plot_data, cmap=cmap, vmin=0, vmax=vmax,
                aspect='auto')
            # plt.colorbar(im, ax=ax)#, fraction=0.046, pad=0.04)
            ax.tick_params(**{'length': 0})
            ax.set_xticks(range(len(plot_data.columns)),
                labels=[i.replace('_', ' ').capitalize() for i in
                        plot_data.columns], ha='right', va='top',
                rotation=45, size=7)
            if a == 0:
                ax.set_yticks(np.arange(len(plot_data.index)),
                    labels=[i.replace('_', ' ').capitalize() for i in
                            plot_data.index], size=7)
                ax.set_xlabel('Train', size=9)
                ax.set_ylabel('Test', size=9)
            else:
                ax.set_yticks([])
            ax.tick_params(direction='in')
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            for (tr, training_set), (te, test_set) in itp(
                    enumerate(plot_data.index),
                    enumerate(plot_data.columns)):
                value = plot_data.iloc[tr, te]
                text_col = 'w' if value < .5 else 'k'
                fmt = 'bold' if value == plot_data.iloc[:,
                                         te].max() else 'normal'
                text = f'{value:.2f}'.replace('0.', '.')
                if not text.startswith('-'):
                    text = text[:3]
                ax.text(te, tr, text, weight=fmt, ha='center', va='center',
                    color=text_col, size=9)
            for training_set in plot_data.index:
                box_x = plot_data.columns.get_loc(training_set) - .5
                box_y = plot_data.index.get_loc(training_set) - .5
                box = patches.Rectangle((box_x, box_y), 1, 1, linewidth=1,
                    edgecolor=box_color, facecolor='none', clip_on=False,
                    zorder=3)
                ax.add_patch(box)
            ax.set_title(architectures[arch])
        #plt.tight_layout()
        plt.subplots_adjust(top=.9, bottom=.45, left=.14, right=.99)
        for ft in ['png', 'pdf']:
            plt.savefig(op.join(out_dir,
                f'cross_generalization_matrix_{score_type}_ind.{ft}'))
        plt.close()


def human_likeness():

    # human data
    scores_human_all = pd.read_parquet(f'../../humans/trials.parquet')
    noise_ceiling_all = pd.read_csv(f'../../humans/noise_ceiling.csv')

    # model data
    scores_model_all = pd.DataFrame()
    for arch, aug in itp(architectures, training_augmentations):
        df = pd.read_parquet(f'../models/original/{arch}/'
                         f'{aug}/trials.parquet')
        df['architecture'] = arch
        df['training_augmentation'] = aug
        scores_model_all = pd.concat((scores_model_all, df), ignore_index=True)

    human_likeness_all = pd.DataFrame()
    for arch, aug in itp(architectures, training_augmentations):
        df = pd.read_csv(f'../models/original/{arch}/'
                         f'{aug}/human_likeness.csv')
        df['architecture'] = arch
        df['training_augmentation'] = aug
        human_likeness_all = pd.concat((human_likeness_all, df), ignore_index=True)

    hldir = f'figures/human_likeness'
    os.makedirs(hldir, exist_ok=True)


    """ raw performance, i.e., 8-way classification accuracy """
    outdir = f'{hldir}/performance'
    os.makedirs(outdir, exist_ok=True)

    # visibility-accuracy curves
    df = scores_human_all.copy()
    df = (df
        .groupby(['visibility', 'occluder_class', 'occluder_color'],
        observed=False, dropna=True)
        .agg({'accuracy': 'mean'}).reset_index())
    # df['color'] = colors_aug['humans']['linecolor']
    df['training_augmentation'] = 'humans'
    df['linestyle'] = colors_aug['humans']['linestyle']
    for aug in training_augmentations:
        df_model = (scores_model_all
            [scores_model_all.training_augmentation == aug]
            .rename(columns={'value': 'accuracy'}))
        df_model = (df_model.groupby(['visibility', 'occluder_class',
                           'occluder_color'],
            observed=True, dropna=False).agg(
            {'accuracy': 'mean'}).reset_index())
        assert len(df_model) == 91
        #df_model['color'] = colors_aug[aug]['linecolor']
        df_model['linestyle'] = colors_aug[aug]['linestyle']
        df_model['training_augmentation'] = aug
        df = pd.concat([df, df_model], ignore_index=True)
    colors = []
    for r, row in df.iterrows():
        """
        if row.occluder_color == 'black':
            colors.append(colors_aug[row.training_augmentation][
                'linecolor'])
        elif row.occluder_color == 'white':
            colors.append(colors_aug[row.training_augmentation][
                'color_light'])
        """
        if row.occluder_color in ['black', 'white']:
            colors.append(
                colors_aug[row.training_augmentation]['linecolor'])
        else:
            colors.append(np.nan)
    df['color'] = colors
    df = df.sort_values('visibility', ascending=False)

    fig, axes = plt.subplots(1, 9, figsize=(6, 1.7), sharex=True,
        sharey=True)
    xvals = CFG.visibilities + [1.0]

    for group in df.training_augmentation.unique():
        df_group = df[df.training_augmentation == group].reset_index(
            drop=True)
        ls = df_group.linestyle.values[0]
        perf_unocc = df_group[df_group.visibility == 1].accuracy.mean()
        for o, occluder_class in enumerate(CFG.occluder_classes):
            ax = axes.flatten()[o]
            # for c, occluder_color in enumerate(CFG.occluder_colors):
            #    df_line = df_group[
            #        (df_group.occluder_class == occluder_class) &
            #        (df_group.occluder_color == occluder_color)]
            df_line = df_group[df_group.occluder_class == occluder_class]
            color = df_line.color.values[0]
            yvals = (df_line.groupby('visibility',
                observed=False).accuracy.mean().to_list() + [perf_unocc])
            assert len(yvals) == len(xvals)
            ax.plot(xvals, yvals, color=color, ls=ls, clip_on=False,
                zorder=2, linewidth=.6)
            ax.set_title(CFG.occluder_labels[o].replace(' ', '\n'), size=7,
                pad=40)
            ax.set_yticks((0, 1))
            ax.set_ylim((0, 1))
            ax.set_xticks((1, .1), labels=(f'0', f'90'))
            ax.set_xlim((1.1, .1))
            ax.tick_params(axis='both', which='major', labelsize=7, length=0)
            # ax.axhline(y=acc1unalt, color=colors[0], linestyle='dashed')
            ax.axhline(y=1 / 8, color='k', linestyle='dotted', linewidth=.6)
            if o == 0:
                ax.set_xlabel('Occlusion (%)', size=8)
                ax.set_ylabel('Classification\naccuracy', size=8)
    plt.tight_layout()
    plt.savefig(op.join(outdir, f'curves.png'))
    plt.savefig(op.join(outdir, f'curves.pdf'))
    plt.close()

    # overall performance (individual and average DNN vs individual humans)
    scores_model = scores_model_all[scores_model_all.visibility < 1]
    scores_human = scores_human_all[scores_human_all.visibility < 1]
    human_accs = scores_human.groupby('subject').accuracy.mean()
    human_sem = human_accs.sem()
    human_mean = human_accs.mean()
    human_upr = human_mean + human_sem
    human_lwr = human_mean - human_sem

    fig, axes = plt.subplots(ncols=len(architectures) + 1, figsize=(6, 2.5),
        sharey=True, width_ratios=[1.5] + [1] * len(architectures))

    # pooled DNN
    ax = axes[0]
    color = colors_aug['humans']['color']
    edgecolor = colors_aug['humans']['edgecolor']
    ax.bar(0, human_mean, color=color, edgecolor=edgecolor, linewidth=1,
        zorder=2)
    ax.errorbar(0, human_mean, human_sem, color='k', capsize=3, capthick=.5,
        zorder=4)
    for xpos, aug in enumerate(training_augmentations):
        linewidth = .5 if aug == 'no_occlusion' else 0
        color = colors_aug[aug]['color']
        edgecolor = colors_aug[aug]['edgecolor']
        points = (scores_model
            [scores_model.training_augmentation == aug]
            .groupby('subject').agg({'value': 'mean'})['value'])
        ax.bar(xpos + 2, np.mean(points), color=color, edgecolor=edgecolor,
            linewidth=linewidth, zorder=2)
        ax.errorbar(xpos + 2, yerr=points.std(), y=points.mean(),  color='k',
            capsize=3, capthick=.5, zorder=4)
    ax.fill_between((-1, 20), human_lwr, human_upr, color='tab:gray', lw=0,
        zorder=1, alpha=0.5)
    ax.axhline(y=human_mean, color='k', zorder=2, lw=.5)
    ax.set_ylabel('accuracy', size=12)
    ax.set_yticks(np.arange(0, 2, .2))
    ax.set_title('humans and average DNNs', size=10)
    ax.grid(axis='y', linestyle='solid', alpha=.25, zorder=0, clip_on=False)
    ax.set_xlim((-.7, len(training_augmentations) + 1.7))
    ax.set_xticks([0] + list(np.arange(len(training_augmentations)) + 2),
        labels=['humans'] + [i.replace('_', ' ') for i in training_augmentations],
        rotation=45, ha='right', va='top', size=8)
    ax.axhline(y=1 / 8, color='k', ls='dotted')
    ax.set_ylim(0, 1)

    # individual DNN
    for m, (arch, label) in enumerate(architectures.items()):
        ax = axes[m + 1]
        for xpos, aug in enumerate(training_augmentations):
            color = colors_aug[aug]['color']
            edgecolor = colors_aug[aug]['edgecolor']
            linewidth = .5 if aug == 'no_occlusion' else 0
            points = (scores_model[
                (scores_model.architecture == arch) &
                (scores_model.training_augmentation == aug)]
                .groupby('subject').agg({'value': 'mean'})
                ['value'].values)
            if not len(points):
                continue
            ax.bar(xpos, np.mean(points), color=color,
                edgecolor=edgecolor, linewidth=linewidth, zorder=2)
            ax.errorbar(xpos, np.mean(points), stats.sem(points), color='k',
                capsize=3, capthick=.5, zorder=4)
        ax.axhline(y=human_mean, color='k', zorder=2, lw=0.5)
        ax.fill_between((-1, 20), human_lwr, human_upr, color='tab:gray',
            lw=0, zorder=1, alpha=0.5)
        ax.set_xlabel(label, size=10, rotation=0)
        if m > 0:
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', which='major', length=0)
        ax.tick_params(axis='x', which='major', length=0)
        ax.grid(axis='y', linestyle='solid', alpha=.25, zorder=0,
            clip_on=False)
        ax.set_xlim((-.7, len(training_augmentations) - 0.3))
        ax.set_xticks([])
        ax.axhline(y=1/8, color='k', ls='dotted')
        ax.set_ylim(0, 1)

    fig.text(.6, .92, 'individual DNNs', size=10)
    plt.tight_layout(pad=1)
    plt.subplots_adjust(wspace=0.1)
    fig.savefig(op.join(outdir, f'overall.pdf'))
    fig.savefig(op.join(outdir, f'overall.png'))
    plt.close()

    make_legend(outpath=op.join(outdir, 'training_augmentation_legend.pdf'),
        labels=[i.replace('_', ' ') for i in list(colors_aug.keys())],
        markers="None",
        colors=[v['linecolor'] for v in colors_aug.values()],
        markeredgecolors=None,
        linestyles=[v['linestyle'] for v in colors_aug.values()])


    """ scatterplots comparing average DNN vs average human performance 
    across 18 occluder conditions """

    outdir = f'{hldir}/alignment'
    os.makedirs(outdir, exist_ok=True)

    # aggregate relevant data
    scores_human = (
        scores_human_all[scores_human_all.visibility < 1]
        .groupby(['occluder_class', 'occluder_color'], observed=True)
        .agg({'accuracy': 'mean'})
        .reset_index()
    )
    scores_model = (scores_model_all[scores_model_all.visibility < 1]
        .groupby(['occluder_class', 'occluder_color', 'training_augmentation'],
        observed=False).agg({'value': 'mean'}).reset_index())

    fig, axes = plt.subplots(ncols=5, figsize=[6, 1.35], sharey=True,
        sharex=True)
    for a, aug in enumerate(training_augmentations):
        ax = axes[a]
        ax.set_aspect('equal')
        scores = scores_human.copy()
        scores.rename(columns={'accuracy': 'human_accuracy'}, inplace=True)
        scores_model_sub = (
            scores_model[scores_model.training_augmentation == aug]
            .rename(columns={'value': 'model_accuracy'})
        )
        scores = scores.merge(
            scores_model_sub, on=['occluder_class', 'occluder_color'])
        scores.occluder_class = pd.Categorical(scores.occluder_class,
                      categories=CFG.occluder_classes)
        scores = scores.sort_values(['occluder_class', 'occluder_color'])

        xvals, yvals = scores[['human_accuracy', 'model_accuracy']].values.T
        ax.scatter(xvals, yvals, color=CFG.plot_colors, clip_on=False, s=3)

        # run correlation and regression
        r = np.corrcoef(xvals, yvals)[0, 1]
        slope, intercept = np.polyfit(xvals, yvals, 1)
        mse = np.mean((yvals - xvals) ** 2)
        x = [0, 1]
        y = np.poly1d(np.polyfit(xvals, yvals, 1))(x)
        ax.plot(x, y, color='k')
        #ax.text(.05, .85, f'r = {r:.2f}\nslope = {slope:.2f}', size=6)
        rx, ry = (.15, .65) if aug == 'no_occlusion' else (.45, .15)
        text = f'r = {r:.2f}\nMSE = {mse:.3f}'.replace('0.', '.')
        ax.text(rx, ry, text, size=6)
        ax.plot([0, 1], [0, 1], color='k', ls='dotted')  # line of unity
        ax.set_title(f'{aug.replace("_", " ").capitalize()}', size=8)
        ax.tick_params(axis='both', which='major', labelsize=7, length=1.5)

    # format
    axes[0].set_xlabel(f'Human accuracy', size=8)
    #ax.set_xticks(np.arange(.3, .7, .1))
    axes[0].set_xticks(np.arange(0, 2, .2))
    # ax.set_xlim((.35,.6))
    axes[0].set_xlim((0.1, .8))
    axes[0].set_yticks(np.arange(0, 2, .2))
    # ax.set_ylim(ylims[aug])
    axes[0].set_ylim((0.1, .8))
    axes[0].set_ylabel(f'DNN accuracy', size=8)

    plt.tight_layout()
    plt.savefig(f'{outdir}/scatterplots.png')
    plt.savefig(f'{outdir}/scatterplots.pdf')
    plt.close()

    make_legend(outpath=op.join(outdir, 'test_occluder_legend.pdf'),
        labels=[f'{o}, {c}' for o, c in
                itp(CFG.occluder_labels, CFG.occluder_colors)],
        colors=CFG.plot_colors, markeredgecolors=None, linestyles="None")


    """ human vs DNN performance using different similarity metrics """

    outdir = f'{hldir}/alignment'
    os.makedirs(outdir, exist_ok=True)
    for metric, p in {
            'Pearson R': {
                'ylims': (-1, 1),
                'yticks': [-1, -.5, 0, .5, 1],
                'ylabel': r"Pearson's $r$"},
            'MSE': {
                'ylims': (0, .25),
                #'yticks': [.25, .20, .15, .1, .05, 0],
                'yticks': [0, .05, .1, .15, .2, .25],
                'ylabel': 'MSE'},
            }.items():

        human_likeness = human_likeness_all[
            human_likeness_all.metric_sim == metric]
        noise_ceiling = noise_ceiling_all[
            noise_ceiling_all.metric_sim == metric]
        assert len(noise_ceiling) == 30
        upper_bound = noise_ceiling.upr.mean()
        lower_bound = noise_ceiling.lwr.mean()

        fig, axes = plt.subplots(ncols=len(architectures) + 1,
            figsize=(6, 1.5), sharey=True)

        # pooled DNN
        ax = axes[0]
        for xpos, aug in enumerate(training_augmentations):
            color = colors_aug[aug]['color']
            edgecolor = colors_aug[aug]['edgecolor']
            linewidth = .5 if aug == 'no_occlusion' else 0
            points = (human_likeness[
                (human_likeness.training_augmentation == aug)]
                .groupby('subject').agg({'value': 'mean'})
                ['value'].values)
            sns.stripplot(x=xpos, y=points, zorder=3, clip_on=False,
                native_scale=True, dodge=True, ax=ax, color='tab:grey',
                size=3, alpha=.5, linewidth=0, edgecolor='None')
            if metric in ['accuracy_distance']:
                bottom, bar_height = np.mean(points), 1
            else:
                bottom, bar_height = 0, np.mean(points)
            ax.bar(xpos, bar_height, bottom=bottom, color=color,
                edgecolor=edgecolor, linewidth=linewidth, zorder=2)
            ax.errorbar(xpos, np.mean(points), stats.sem(points), color='k',
                capsize=3, capthick=.5, zorder=4)
        ax.fill_between((-1, 20), lower_bound, upper_bound, color='tab:gray',
            lw=0, zorder=1)
        ax.set_ylabel(p['ylabel'], size=8)
        ax.set_yticks(p['yticks'])
        ax.set_ylim(p['ylims'])
        ax.grid(axis='y', linestyle='solid', alpha=.25, zorder=0, clip_on=False)
        ax.set_xlim((-.7, len(training_augmentations) - 0.3))
        ax.set_xticks([])
        if metric not in ['accuracy_distance']:
            ax.axhline(y=0, color='k', ls='dotted')
        if min(p['ylims']) < 0:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='major', length=0)
        ax.set_xlabel('All DNNs', size=8)

        #ax.set_xticks(np.arange(len(training_augmentations)),
        #    labels=[i.replace('_', ' ') for i in training_augmentations],
        #    rotation=45, ha='right', va='top', size=8)

        # individual DNN
        for m, (arch, label) in enumerate(architectures.items()):
            ax = axes[m+1]
            for xpos, aug in enumerate(training_augmentations):
                color = colors_aug[aug]['color']
                edgecolor = colors_aug[aug]['edgecolor']
                linewidth = .5 if aug == 'no_occlusion' else 0
                points = (
                    human_likeness[
                        (human_likeness.architecture == arch) &
                        (human_likeness.training_augmentation == aug)]
                    ['value'].values)
                if not len(points):
                    continue
                sns.stripplot(x=xpos, y=points, zorder=3, clip_on=False,
                    native_scale=True, dodge=True, ax=ax,
                    color='tab:grey', size=2.5,
                    alpha=.5, linewidth=0, edgecolor='None')
                if metric in ['accuracy_distance']:
                    bottom, bar_height = np.mean(points), 1
                else:
                    bottom, bar_height = 0, np.mean(points)
                ax.bar(xpos, bar_height, bottom=bottom, color=color,
                    edgecolor=edgecolor, linewidth=linewidth, zorder=2)
                ax.errorbar(xpos, np.mean(points), stats.sem(points), color='k',
                    capsize=3, capthick=.5, zorder=4)
            ax.fill_between((-1, 20), lower_bound, upper_bound,
                color='tab:gray', lw=0, zorder=1)
            ax.set_xlabel(label, size=8, rotation=0)
            if m > 0:
                ax.spines['left'].set_visible(False)
                ax.tick_params(axis='y', which='major', length=0)
            ax.grid(axis='y', linestyle='solid', alpha=.25, zorder=0, clip_on=False)
            ax.set_xlim((-.7, len(training_augmentations) - 0.3))
            ax.set_xticks([])
            if min(p['ylims']) < 0:
                ax.spines['bottom'].set_visible(False)
            if metric not in ['accuracy_distance']:
                ax.axhline(y=0, color='k', ls='dotted')
            ax.set_ylim(p['ylims'])
        #fig.text(.6, .92, 'individual DNNs', size=10)
        plt.tight_layout()
        plt.subplots_adjust(wspace=.1)
        fig.savefig(f'{outdir}/{metric}.png')
        fig.savefig(f'{outdir}/{metric}.pdf')
        plt.close()

        """ summary statistics """
        df = human_likeness.copy()
        df['z'] = np.arctanh(df['value'])

        # individual architectures, pearson r and fisher z transformed
        summary_ind = (
            df[['architecture', 'training_augmentation', 'value', 'z']]
            .groupby(['architecture', 'training_augmentation'],
                observed=False)
            .agg(['mean', 'sem'], numeric_only=True).reset_index())

        # all architectures, pearson r and fisher z transformed
        summary_pooled = (
            df[['subject', 'training_augmentation', 'value', 'z']]
            .groupby(['subject', 'training_augmentation'],
                observed=False)
            .agg('mean', numeric_only=True)
            .groupby('training_augmentation', observed=False)
            .agg(['mean', 'sem'], numeric_only=True).reset_index())
        summary_pooled['architecture'] = 'all_DNNs'

        # combine and rename columns
        summary = pd.concat([summary_pooled, summary_ind],
            ignore_index=True)
        summary.columns = \
            ['training_augmentation', 'mean_r', 'sem_r',
             'mean_z', 'sem_z', 'architecture']

        # noise ceiling, pearson r and fisher z transformed
        ncs = noise_ceiling[['lwr', 'upr']]
        ncs['lwr_z'] = np.arctanh(ncs.lwr)
        ncs['upr_z'] = np.arctanh(ncs.upr)
        summary = pd.concat([summary, pd.DataFrame({
            'architecture': ['humans', 'humans'],
            'training_augmentation': ['nc_lwr', 'nc_upr'],
            'mean_r': [lower_bound, upper_bound],
            'sem_r': [ncs.lwr.sem(), ncs.lwr.sem()],
            'mean_z': [np.arctanh(lower_bound), np.arctanh(upper_bound)],
            'sem_z': [ncs.lwr_z.sem(), ncs.upr_z.sem()]
        })])
        summary.to_csv(f'{outdir}/summary_{metric}.csv')

        # 2-way ANOVA with factors architecture and training augmentation
        anova = AnovaRM(df, depvar='z', subject='subject',
            within=['architecture', 'training_augmentation'],
            aggregate_func='mean').fit().anova_table
        post_hocs = pg.pairwise_tests(dv='z',  subject='subject',
            within=['architecture', 'training_augmentation'],
            data=df, padjust='holm')
        anova.to_csv(f'{outdir}/anova-2_{metric}.csv')
        post_hocs.to_csv(f'{outdir}/posthocs-2_{metric}.csv', index=False)

        # 1-way ANOVA with factor training augmentation
        anova = AnovaRM(df, depvar='z', subject='subject',
            within=['training_augmentation'],
            aggregate_func='mean').fit().anova_table
        post_hocs = pg.pairwise_tests(dv='z', subject='subject',
            within=['training_augmentation'], data=df, padjust='holm')
        anova.to_csv(f'{outdir}/anova-1_{metric}.csv')
        post_hocs.to_csv(f'{outdir}/posthocs-1_{metric}.csv',
            index=False)

    # make barplot legend
    make_legend(
        outpath=f'{outdir}/training_augmentation_legend.png',
        labels=training_augmentations, markers="s",
        colors=[colors_aug[i]['color'] for i in training_augmentations],
        markeredgecolors=[colors_aug[i]['edgecolor'] for i in
                          training_augmentations],
        linestyles='None')


def performance_equated():
    out_dir = f'figures/performance_equated'
    os.makedirs(out_dir, exist_ok=True)

    # human data
    scores_human_all = pd.read_parquet(f'../../humans/trials.parquet')
    noise_ceiling_all = pd.read_csv(f'../../humans/noise_ceiling.csv')

    # model data

    scores_model_all = pd.DataFrame()
    human_likeness_all = pd.DataFrame()

    scores_model_all = pd.DataFrame()
    human_likeness_all = pd.DataFrame()
    model_sets = ['original', 'performance_equated']
    for model_set, arch, aug in itp(model_sets, architectures.keys(),
            training_augmentations):

        """Finetuning did not occur if overall performance across
        architectures was already human-level (i.e. natural-trained models) 
        or below (i.e. no occlusion, since finetuning wouldn't work)
        or for individual DNNs that already performed at human-level (i.e. 
        AlexNet trained with natural silhouette and artificial 2)."""

        if aug in ['no_occlusion', 'natural'] or (
                arch == 'alexnet' and aug in ['natural_silhouette',
                                              'artificial_2']):
            model_dir = f'../models/original/{arch}/{aug}'
        else:
            model_dir = f'../models/{model_set}/{arch}/{aug}'

        # accuracies
        df = pd.read_parquet(f'{model_dir}/trials.parquet')
        df['model_set'] = model_set
        df['architecture'] = arch
        df['training_augmentation'] = aug
        scores_model_all = pd.concat([scores_model_all, df])

        # human likeness
        df = pd.read_csv(f'{model_dir}/human_likeness.csv')
        df['model_set'] = model_set
        df['architecture'] = arch
        df['training_augmentation'] = aug
        human_likeness_all = pd.concat([human_likeness_all, df])

    """ raw performance, i.e., 8-way classification accuracy """
    outdir = f'{out_dir}'
    os.makedirs(outdir, exist_ok=True)

    # overall performance (individual and average DNN vs individual humans)
    scores_model = scores_model_all[scores_model_all.visibility < 1]
    scores_human = scores_human_all[scores_human_all.visibility < 1]
    human_accs = scores_human.groupby('subject').accuracy.mean()
    human_sem = human_accs.sem()
    human_mean = human_accs.mean()
    human_upr = human_mean + human_sem
    human_lwr = human_mean - human_sem

    fig, axes = plt.subplots(ncols=3, figsize=(3.7, 2.5), sharey=True,
        width_ratios=[1, 3.5, 3.5])

    # humans
    ax = axes[0]
    color = colors_aug['humans']['color']
    edgecolor = colors_aug['humans']['edgecolor']
    ax.bar(0, human_mean, color=color, edgecolor=edgecolor, linewidth=0,
        zorder=2)
    ax.errorbar(0, human_mean, human_sem, color='k', capsize=3, capthick=.5,
        zorder=4, linewidth=.25)
    ax.set_title('Humans', size=9)
    ax.set_ylabel('Classification accuracy', size=9)
    ax.set_xticks([])
    ax.set_xlim([-.7, .7])
    ax.axhline(y=1 / 8, color='k', ls='dotted')
    ax.grid(axis='y', linestyle='solid', alpha=.25, zorder=0, clip_on=False)

    # DNNs
    for m, model_set in enumerate(model_sets):
        ax = axes[m + 1]
        augs = training_augmentations
        # if training == 'finetuned':
        #    augs = augs[1:]
        for xpos, aug in enumerate(augs):
            linewidth = .5 if aug == 'no_occlusion' else 0
            color = colors_aug[aug]['color']
            edgecolor = colors_aug[aug]['edgecolor']
            points = (scores_model[(scores_model.model_set == model_set) & (
                        scores_model.training_augmentation == aug)].groupby(
                'subject').agg({'value': 'mean'})['value'])
            ax.bar(xpos, np.mean(points), color=color, edgecolor=edgecolor,
                linewidth=linewidth, zorder=2)
            ax.errorbar(xpos, yerr=points.std(), y=points.mean(), color='k',
                capsize=3, capthick=.5, zorder=4, linewidth=.25)
        ax.fill_between((-1, 20), human_lwr, human_upr, color='tab:gray', lw=0,
            zorder=1, alpha=0.5)
        ax.axhline(y=human_mean, color='k', zorder=1, lw=.5)
        ax.set_yticks(np.arange(0, 2, .2))
        ax.set_title(['Original\nDNNs', 'Performance-equated\nDNNs'][m], size=9)
        ax.grid(axis='y', linestyle='solid', alpha=.25, zorder=0, clip_on=False)
        ax.set_xlim((-.7, len(augs) - .3))
        ax.set_xticks(np.arange(len(augs)) + .3,
            labels=[i.replace('_', ' ').capitalize() for i in augs],
            rotation=45, ha='right', va='top', size=8)
        ax.axhline(y=1 / 8, color='k', ls='dotted')
        ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.subplots_adjust(left=.16, bottom=.3, right=.92, top=.87)
    fig.savefig(op.join(outdir, f'accuracy.pdf'))
    fig.savefig(op.join(outdir, f'accuracy.png'))
    plt.close()

    # human alignment

    """ scatterplots comparing average DNN vs average human performance 
            across 18 occluder conditions """

    # aggregate relevant data
    scores_human = (scores_human_all[scores_human_all.visibility < 1].groupby(
        ['occluder_class', 'occluder_color'], observed=True).agg(
        {'accuracy': 'mean'}).reset_index())
    scores_model = (scores_model_all[(scores_model_all.visibility < 1) & (
                scores_model_all.model_set == 'performance_equated')].groupby(
        ['occluder_class', 'occluder_color', 'training_augmentation'],
        observed=False).agg({'value': 'mean'}).reset_index())

    fig, axes = plt.subplots(ncols=5, figsize=[6, 1.35], sharey=True,
        sharex=True)
    for a, aug in enumerate(training_augmentations):
        ax = axes[a]
        ax.set_aspect('equal')
        scores = scores_human.copy()
        scores.rename(columns={'accuracy': 'human_accuracy'}, inplace=True)
        scores_model_sub = (
            scores_model[scores_model.training_augmentation == aug].rename(
                columns={'value': 'model_accuracy'}))
        scores = scores.merge(scores_model_sub,
            on=['occluder_class', 'occluder_color'])
        scores.occluder_class = pd.Categorical(scores.occluder_class,
            categories=CFG.occluder_classes)
        scores = scores.sort_values(['occluder_class', 'occluder_color'])

        xvals, yvals = scores[['human_accuracy', 'model_accuracy']].values.T
        ax.scatter(xvals, yvals, color=CFG.plot_colors, clip_on=False, s=3)

        # run correlation and regression
        r = np.corrcoef(xvals, yvals)[0, 1]
        slope, intercept = np.polyfit(xvals, yvals, 1)
        # mse = np.mean((yvals - xvals) ** 2)
        x = [0, 1]
        y = np.poly1d(np.polyfit(xvals, yvals, 1))(x)
        ax.plot(x, y, color='k')
        # ax.text(.05, .85, f'r = {r:.2f}\nslope = {slope:.2f}', size=6)
        rx, ry = (.15, .74)  # if aug == 'no_occlusion' else (.45, .15)
        text = f'r = {r:.2f}'.replace('0.', '.')  # \nMSE = {mse:.3f}'
        ax.text(rx, ry, text, size=6)
        ax.plot([0, 1], [0, 1], color='k', ls='dotted')  # line of unity
        ax.set_title(f'{aug.replace("_", " ").capitalize()}', size=8)
        ax.tick_params(axis='both', which='major', labelsize=7, length=1.5)

    # format
    axes[0].set_xlabel(f'Human accuracy', size=8)
    # ax.set_xticks(np.arange(.3, .7, .1))
    axes[0].set_xticks(np.arange(0, 2, .2))
    # ax.set_xlim((.35,.6))
    axes[0].set_xlim((0.1, .8))
    axes[0].set_yticks(np.arange(0, 2, .2))
    # ax.set_ylim(ylims[aug])
    axes[0].set_ylim((0.1, .8))
    axes[0].set_ylabel(f'DNN accuracy', size=8)

    plt.tight_layout()
    plt.savefig(f'{outdir}/scatterplots.png')
    plt.savefig(f'{outdir}/scatterplots.pdf')
    plt.close()

    for metric, p in {
        'Pearson R': {'ylims': (-.5, 1), 'yticks': [-1, -.5, 0, .5, 1],
            'ylabel': r"Pearson's $r$"},
        'MSE': {'ylims': (.25, 0), 'yticks': [.25, .20, .15, .1, .05, 0],
            'ylabel': 'MSE'}, }.items():
        human_likeness = human_likeness_all[
            human_likeness_all.metric_sim == metric]
        noise_ceiling = noise_ceiling_all[
            noise_ceiling_all.metric_sim == metric]
        assert len(noise_ceiling) == 30
        upper_bound = noise_ceiling.upr.mean()
        lower_bound = noise_ceiling.lwr.mean()

        fig, axes = plt.subplots(ncols=2, figsize=(3.1, 2.5), sharey=True,
            width_ratios=[1, 1])
        for m, model_set in enumerate(model_sets):
            ax = axes[m]
            augs = training_augmentations
            # if training == 'finetuned':
            #    augs = augs[1:]
            for xpos, aug in enumerate(augs):
                color = colors_aug[aug]['color']
                edgecolor = colors_aug[aug]['edgecolor']
                linewidth = .5 if aug == 'no_occlusion' else 0
                points = (human_likeness[
                              (human_likeness.model_set == model_set) & (
                                          human_likeness.training_augmentation == aug)].groupby(
                    'subject').agg({'value': 'mean'})['value'].values)
                sns.stripplot(x=xpos, y=points, zorder=3, clip_on=False,
                    native_scale=True, dodge=True, ax=ax, color='tab:grey',
                    size=3, alpha=.5, linewidth=0, edgecolor='None')
                if metric in ['MSE', 'accuracy_distance']:
                    bottom, bar_height = np.mean(points), 1
                else:
                    bottom, bar_height = 0, np.mean(points)
                ax.bar(xpos, bar_height, bottom=bottom, color=color,
                    edgecolor=edgecolor, linewidth=linewidth, zorder=2)
                ax.errorbar(xpos, np.mean(points), stats.sem(points), color='k',
                    capsize=3, capthick=.5, zorder=4)
            ax.fill_between((-1, 20), lower_bound, upper_bound,
                color='tab:gray', alpha=0.5, lw=0, zorder=1)
            ax.set_ylabel(p['ylabel'], size=9)
            ax.set_yticks(p['yticks'])
            ax.set_ylim(p['ylims'])
            ax.grid(axis='y', linestyle='solid', alpha=.25, zorder=0,
                clip_on=False)
            ax.set_xlim((-.7, len(augs) - 0.3))
            ax.set_xticks(np.arange(len(augs)) + .3,
                labels=[i.replace('_', ' ').capitalize() for i in augs],
                rotation=45, ha='right', va='top', size=8)
            if metric not in ['MSE', 'accuracy_distance']:
                ax.axhline(y=0, color='k', ls='dotted')
            if min(p['ylims']) < 0:
                ax.spines['bottom'].set_visible(False)
                ax.tick_params(axis='x', which='major', length=0)
            ax.set_title(['Original\nDNNs', 'Performance-equated\nDNNs'][m],
                size=9)
        plt.tight_layout()
        plt.subplots_adjust(left=.16, bottom=.3, right=.92, top=.87)
        fig.savefig(op.join(outdir, f'{metric}.pdf'))
        fig.savefig(op.join(outdir, f'{metric}.png'))
        plt.close()

        """ summary statistics """
        df = human_likeness[human_likeness.model_set ==
                            'performance_equated'].copy()
        df['z'] = np.arctanh(df['value'])

        # individual architectures, pearson r and fisher z transformed
        summary_ind = (
            df[['architecture', 'training_augmentation', 'value', 'z']].groupby(
                ['architecture', 'training_augmentation'], observed=False).agg(
                ['mean', 'sem'], numeric_only=True).reset_index())

        # all architectures, pearson r and fisher z transformed
        summary_pooled = (
            df[['subject', 'training_augmentation', 'value', 'z']].groupby(
                ['subject', 'training_augmentation'], observed=False).agg(
                'mean', numeric_only=True).groupby('training_augmentation',
                observed=False).agg(['mean', 'sem'],
                numeric_only=True).reset_index())
        summary_pooled['architecture'] = 'all_DNNs'

        # combine and rename columns
        summary = pd.concat([summary_pooled, summary_ind], ignore_index=True)
        summary.columns = ['training_augmentation', 'mean_r', 'sem_r', 'mean_z',
                           'sem_z', 'architecture']

        # noise ceiling, pearson r and fisher z transformed
        ncs = noise_ceiling[['lwr', 'upr']]
        ncs['lwr_z'] = np.arctanh(ncs.lwr)
        ncs['upr_z'] = np.arctanh(ncs.upr)
        summary = pd.concat([summary, pd.DataFrame(
            {'architecture': ['humans', 'humans'],
             'training_augmentation': ['nc_lwr', 'nc_upr'],
             'mean_r': [lower_bound, upper_bound],
             'sem_r': [ncs.lwr.sem(), ncs.lwr.sem()],
             'mean_z': [np.arctanh(lower_bound), np.arctanh(upper_bound)],
             'sem_z': [ncs.lwr_z.sem(), ncs.upr_z.sem()]})])
        summary.to_csv(f'{outdir}/summary_{metric}.csv')

        # 2-way ANOVA with factors architecture and training augmentation
        anova = AnovaRM(df, depvar='z', subject='subject',
            within=['architecture', 'training_augmentation'],
            aggregate_func='mean').fit().anova_table
        post_hocs = pg.pairwise_tests(dv='z', subject='subject',
            within=['architecture', 'training_augmentation'], data=df,
            padjust='holm')
        anova.to_csv(f'{outdir}/anova-2_{metric}.csv')
        post_hocs.to_csv(f'{outdir}/posthocs-2_{metric}.csv', index=False)

        # 1-way ANOVA with factor training augmentation
        anova = AnovaRM(df, depvar='z', subject='subject',
            within=['training_augmentation'],
            aggregate_func='mean').fit().anova_table
        post_hocs = pg.pairwise_tests(dv='z', subject='subject',
            within=['training_augmentation'], data=df, padjust='holm')
        anova.to_csv(f'{outdir}/anova-1_{metric}.csv')
        post_hocs.to_csv(f'{outdir}/posthocs-1_{metric}.csv', index=False)


def occlusion_strength():

    outdir = f'figures/occlusion_strength'
    os.makedirs(outdir, exist_ok=True)

    strengths = ['no', 'weak', 'moderate', 'strong']

    # human data
    scores_human_all = pd.read_parquet(f'../../humans/trials.parquet')
    noise_ceiling_all = pd.read_csv(f'../../humans/noise_ceiling.csv')

    # model data
    scores_model_all = pd.DataFrame()
    human_likeness_all = pd.DataFrame()
    
    # only resnet and efficient_net were trained for this analysis
    archs = {k: v for k, v in architectures.items() if
             k in ['resnet101', 'efficientnet_b1']}
    
    # models trained with no occlusion are from the original set
    for arch in archs:
        model_dir = f'../models/original/{arch}/no_occlusion'
        
        trials = pd.read_parquet(f'{model_dir}/trials.parquet')
        trials = (trials.groupby(
            ['subject', 'visibility', 'occluder_class', 'occluder_color'],
            observed=True, dropna=False).agg({'value': 'mean'}).reset_index())
        assert len(trials) == 91 * 30
        trials['training_augmentation'] = 'no_occlusion'
        trials['training_strength'] = 'no'
        scores_model_all = pd.concat([scores_model_all, trials])

        hl = pd.read_csv(f'{model_dir}/human_likeness.csv')
        hl['training_augmentation'] = 'no_occlusion'
        hl['training_strength'] = 'no'
        human_likeness_all = pd.concat([human_likeness_all, hl])
    
    # other models
    for strength, (arch, arch_label), aug in itp(strengths[1:],
            archs.items(), training_augmentations[1:]):
        
        # models trained with moderate occlusion are from the original set
        if strength == 'moderate':
            model_dir = f'../models/original/{arch}/{aug}'
        else:
            model_dir = f'../models/occlusion_strength/{arch}/{aug}/' \
                        f'{strength}'

        trials = pd.read_parquet(f'{model_dir}/trials.parquet')
        trials = (trials.groupby(
            ['subject', 'visibility', 'occluder_class', 'occluder_color'],
            observed=True, dropna=False).agg(
            {'value': 'mean'}).reset_index())
        assert len(trials) == 91 * 30
        trials['training_augmentation'] = aug
        trials['training_strength'] = strength
        scores_model_all = pd.concat([scores_model_all, trials])

        hl = pd.read_csv(f'{model_dir}/human_likeness.csv')
        hl['training_augmentation'] = aug
        hl['training_strength'] = strength
        human_likeness_all = pd.concat([human_likeness_all, hl])
            

    """ raw performance, i.e., 8-way classification accuracy """

    # overall performance (individual and average DNN vs individual humans)
    scores_model = scores_model_all[scores_model_all.visibility < 1]
    scores_human = scores_human_all[scores_human_all.visibility < 1]
    human_accs = scores_human.groupby('subject').accuracy.mean()
    human_sem = human_accs.sem()
    human_mean = human_accs.mean()
    human_upr = human_mean + human_sem
    human_lwr = human_mean - human_sem

    fig, axes = plt.subplots(ncols=4, figsize=(2.3, 1.5), sharey=True,
        width_ratios=[1, 3.5, 3.5, 3.5])

    # DNNs
    for s, strength in enumerate(strengths):
        ax = axes[s]
        augs = training_augmentations[1:] if s > 0 else ['no_occlusion']
        for xpos, aug in enumerate(augs):
            linewidth = .5 if aug == 'no_occlusion' else 0
            color = colors_aug[aug]['color']
            edgecolor = colors_aug[aug]['edgecolor']
            points = (scores_model[
                (scores_model.training_strength == strength) &
                (scores_model.training_augmentation == aug)]
            .groupby('subject').agg({'value': 'mean'})['value'])
            ax.bar(xpos, np.mean(points), color=color, edgecolor=edgecolor,
                linewidth=linewidth, zorder=2)
            ax.errorbar(xpos, yerr=points.std(), y=points.mean(), color='k',
                capsize=2, capthick=.5, zorder=4, linewidth=.25)
        ax.fill_between((-1, 20), human_lwr, human_upr, color='tab:gray', lw=0,
            zorder=1, alpha=0.5)
        ax.axhline(y=human_mean, color='k', zorder=1, lw=.5)
        ax.set_title(f'{strength.capitalize()}\nocclusion', size=6)
        ax.grid(axis='y', linestyle='solid', alpha=.25, zorder=0, clip_on=False)
        ax.set_xlim((-.7, len(augs) - .3))
        #if s > 0:
        #    ax.set_xticks(np.arange(len(augs)) + .3,
        #        labels=[i.replace('_', ' ').capitalize() for i in augs],
        #        rotation=45, ha='right', va='top', size=8)
        #else:
        ax.set_xticks([])
        ax.axhline(y=1 / 8, color='k', ls='dotted')
        ax.set_ylim(0, 1)
        if s == 0:
            ax.set_ylabel('Top-1 accuracy', size=6)
        yticks = np.arange(0, 1.1, .2)
        ax.set_yticks(yticks, labels=[f'{i:.1f}' for i in yticks], size=6)

    plt.tight_layout(w_pad=1.5)
    #plt.subplots_adjust(left=.16, bottom=.3, right=.92, top=.87)
    fig.savefig(op.join(outdir, f'accuracy.pdf'))
    fig.savefig(op.join(outdir, f'accuracy.png'))
    plt.close()

    # human alignment
    for metric, p in {
        'Pearson R': {
            'ylims': (-1, 1),
            'yticks': [-1, -.5, 0, .5, 1],
            'ylabel': r"Pearson's $r$"},
        'MSE': {
            'ylims': (.25, 0),
            'yticks': [.25, .20, .15, .1, .05, 0],
            'ylabel': 'MSE'}
        }.items():
        
        human_likeness = human_likeness_all[
            human_likeness_all.metric_sim == metric]
        noise_ceiling = noise_ceiling_all[
            noise_ceiling_all.metric_sim == metric]
        assert len(noise_ceiling) == 30
        upper_bound = noise_ceiling.upr.mean()
        lower_bound = noise_ceiling.lwr.mean()

        fig, axes = plt.subplots(ncols=4, figsize=(2.35, 1.5), sharey=True,
            width_ratios=[1, 3.5, 3.5, 3.5])

        for s, strength in enumerate(strengths):
            ax = axes[s]
            augs = training_augmentations[1:] if s > 0 else ['no_occlusion']
            for xpos, aug in enumerate(augs):
                color = colors_aug[aug]['color']
                edgecolor = colors_aug[aug]['edgecolor']
                linewidth = .5 if aug == 'no_occlusion' else 0
                points = (human_likeness[
                    (human_likeness.training_strength == strength) &
                    (human_likeness.training_augmentation == aug)]
                    .groupby('subject').agg({'value': 'mean'})['value'].values)
                sns.stripplot(x=xpos, y=points, zorder=3, clip_on=False,
                    native_scale=True, dodge=True, ax=ax, color='tab:grey',
                    size=1.5, alpha=.5, linewidth=0, edgecolor='None')
                if metric == 'MSE':
                    bottom, bar_height = np.mean(points), 1
                else:
                    bottom, bar_height = 0, np.mean(points)
                ax.bar(xpos, bar_height, bottom=bottom, color=color,
                    edgecolor=edgecolor, linewidth=linewidth, zorder=2)
                ax.errorbar(xpos, np.mean(points), stats.sem(points), color='k',
                    capsize=2, capthick=.5, zorder=4)
            ax.fill_between((-1, 20), lower_bound, upper_bound,
                color='tab:gray', alpha=0.5, lw=0, zorder=1)
            ax.set_ylabel(p['ylabel'], size=6)
            ax.set_yticks(p['yticks'], labels=[f'{i:.1f}' for i in p[
                'yticks']], size=6)
            ax.set_ylim(p['ylims'])
            ax.grid(axis='y', linestyle='solid', alpha=.25, zorder=0,
                clip_on=False)
            ax.set_xlim((-.7, len(augs) - 0.3))
            #if s > 0:
            #    ax.set_xticks(np.arange(len(augs)) + .3,
            #        labels=[i.replace('_', ' ').capitalize() for i in augs],
            #        rotation=45, ha='right', va='top', size=8)
            #else:
            ax.set_xticks([])
            if metric == 'Pearson R':
                ax.axhline(y=0, color='k', ls='dotted')
            if min(p['ylims']) < 0:
                ax.spines['bottom'].set_visible(False)
                ax.tick_params(axis='x', which='major', length=0)
            ax.set_title(f'{strength.capitalize()}\nocclusion', size=6)
        plt.tight_layout(w_pad=1.5)
        #plt.subplots_adjust(left=.16, bottom=.3, right=.92, top=.87)
        fig.savefig(op.join(outdir, f'{metric}.pdf'))
        fig.savefig(op.join(outdir, f'{metric}.png'))
        plt.close()



def other_occluders():

    outdir = 'figures/other_occluders'
    os.makedirs(outdir, exist_ok=True)

    # model data

    augs = {'No occlusion': 'no_occlusion',
        'Natural': 'natural',
        'Natural silhouette': 'natural_silhouette',
        'Artificial 1': 'artificial_1',
        'Artificial 2': 'artificial_2',
        'RandomErase (0)': 'randomerase_0',
        'RandomErase (rand.)': 'randomerase_random',
        'CutMix': 'cutmix'}

    scores = pd.DataFrame()
    for arch, (aug, aug_path) in itp(architectures, augs.items()):

        if aug_path in training_augmentations:
            model_dir = f'../models/original/{arch}/{aug_path}'
        else:
            model_dir = f'../models/other_occluders/{arch}/{aug_path}'

        # get DNN performance on behavioral dataset
        scores_model = pd.read_csv(f'{model_dir}/occlusion_robustness.csv')
        scores_model = scores_model[
            scores_model.benchmark == 'ImageNet-Occluded']
        scores_model['training_augmentation'] = aug
        scores_model['architecture'] = arch
        scores = pd.concat([scores, scores_model], ignore_index=True)

    scores = (scores
        .groupby(['training_augmentation', 'architecture', 'occluder_type'],
            observed=False, dropna=False)
        .agg({'score': 'mean'}).reset_index())

    fig, axes = plt.subplots(ncols=2, figsize=(4.5, 2.5), sharey=True)
    for t, test_aug in enumerate(['artificial', 'natural']):
        ax = axes[t]
        if test_aug == 'artificial':
            scores_test = scores[~scores.occluder_type.str.startswith(
                'natural')]
        else:
            scores_test = scores[scores.occluder_type == 'natural']
        for a, (aug, aug_path) in enumerate(augs.items()):
            linewidth = .5 if aug == 'No occlusion' else 0
            color = colors_aug[aug_path]['color']
            edgecolor = colors_aug[aug_path]['edgecolor']
            points = (
                scores_test[scores_test.training_augmentation == aug].groupby(
                    'architecture').agg({'score': 'mean'})['score'])
            sns.stripplot(x=a, y=points, zorder=3, clip_on=False,
                size=3, alpha=.5, color='tab:grey', ax=ax, linewidth=0)
            ax.bar(a, np.mean(points), color=color, edgecolor=edgecolor,
                linewidth=linewidth, zorder=2)
        # ax.fill_between((-1, 20), human_lwr, human_upr, color='tab:gray', lw=0,
        #    zorder=1, alpha=0.5)
        # ax.axhline(y=human_mean, color='k', zorder=1, lw=.5)

        ax.set_title(f'Test: {test_aug} occlusion', size=9)
        ax.set_ylabel('Top-1 accuracy', size=9)
        ax.set_yticks(np.arange(0, 2, .1))
        ax.set_ylim(0, .5)
        ax.grid(axis='y', linestyle='solid', alpha=.25, zorder=0, clip_on=False)
        ax.set_xlim((-.7, len(augs) - 0.3))
        ax.set_xticks(np.arange(len(augs)),
            labels=list(augs.keys()), rotation=45, ha='right', va='top', size=8)
        ax.set_xlabel('Training augmentation', size=9)
    plt.tight_layout()
    plt.subplots_adjust(wspace=.4)
    fig.savefig(op.join(outdir, f'artificial_natural.pdf'))
    fig.savefig(op.join(outdir, f'artificial_natural.png'))
    plt.close()



def other_distortions():

    outdir = f'figures/other_distortions'
    os.makedirs(outdir, exist_ok=True)

    models = {
        'No distortion': {'path': 'no_distortion', 'color': 'k'},
        'Natural occlusion': {'path': 'natural_occlusion', 'color':
            'tab:green'},
        'Gaussian blur': {'path': 'gaussian_blur', 'color': tab20[2]},
        'Fourier noise': {'path': 'fourier_noise', 'color': 'tab:orange'},
        'Gaussian noise': {'path': 'gaussian_noise', 'color': 'tab:purple'}}


    # get data
    scores = pd.DataFrame()
    for aug, info in models.items():
        scores_model = pd.read_csv(
            f'../models/other_distortions/resnet101/{info["path"]}/scores.csv')
        scores_model['training_augmentation'] = aug
        scores = pd.concat([scores, scores_model], ignore_index=True)

    scores = (scores
        .groupby(['benchmark', 'level_1', 'level_2', 'level_3',
            'training_augmentation', 'test_augmentation'], observed=False,
        dropna=False)
        .agg({'score': 'mean'}).reset_index())

    # cross-generalization matrix
    training_augmentations = list(models.keys())
    box_color = (0, .8, 0)  # 'tab:green'#'tab:red'
    cmap = 'inferno'
    for score_type in ['raw', 'norm', 'base']:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))

        plot_data = (scores.groupby(['training_augmentation',
                                     'test_augmentation'],
            observed=True).agg({'score': 'mean'}).reset_index().pivot(
            index='test_augmentation', columns='training_augmentation',
            values='score'))
        plot_data.index = pd.Categorical(plot_data.index,
            categories=training_augmentations, ordered=True)
        plot_data = plot_data.sort_index()
        #plot_data.columns = plot_data.columns.map(
        #    {v['path']: k for k, v in models.items()})
        plot_data.columns = pd.Categorical(plot_data.columns,
            categories=training_augmentations, ordered=True)
        plot_data = plot_data.sort_index(axis=1)
        if score_type != 'raw':
            plot_data = plot_data.drop(index=['No distortion'])
            for ind in plot_data.index:
                plot_data.loc[ind, :] -= plot_data.loc[ind, 'No distortion']
                if score_type == 'norm':
                    plot_data.loc[ind, :] /= plot_data.loc[ind, ind]
        im = ax.imshow(plot_data, cmap=cmap, vmin=0, vmax=plot_data.max().max())
        ax.tick_params(**{'length': 0})
        ax.set_xticks(range(len(plot_data.columns)),
            labels=[i.replace('_', ' ').capitalize() for i in
                    plot_data.columns], ha='right', va='top', rotation=45,
            size=7)
        ax.set_xlabel('Train', size=9)
        ax.set_yticks(np.arange(len(plot_data.index)),
            labels=[i.replace('_', ' ').capitalize() for i in plot_data.index],
            size=7)
        ax.set_ylabel('Test', size=9)
        ax.tick_params(direction='in')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        for (tr, training_set), (te, test_set) in itp(
                enumerate(plot_data.index), enumerate(plot_data.columns)):
            value = plot_data.iloc[tr, te]
            text_col = 'w' if value < .5 else 'k'
            fmt = 'bold' if value == plot_data.iloc[:, te].max() else 'normal'
            ax.text(te, tr, f'{value:.2f}'.replace('0.', '.'), weight=fmt,
                ha='center', va='center', color=text_col, size=9)
        for training_set in plot_data.index:
            box_x = plot_data.columns.get_loc(training_set) - .5
            box_y = plot_data.index.get_loc(training_set) - .5
            box = patches.Rectangle((box_x, box_y), 1, 1, linewidth=1,
                edgecolor=box_color, facecolor='none', clip_on=False, zorder=3)
            ax.add_patch(box)
        plt.tight_layout()
        for ft in ['png', 'pdf']:
            plt.savefig(op.join(outdir,
                f'cross_generalization_matrix_{score_type}.{ft}'))
        plt.close()



if __name__ == '__main__':
    main()
