import os
import os.path as op
import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from itertools import product as itp
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import sys
import shutil
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from humans.utils import sigmoid
from humans.utils import custom_defaults
plt.rcParams.update(custom_defaults)

def main():
    likeness_figure()
    occluder_type()

figure_dir = 'plots'
os.makedirs(figure_dir, exist_ok=True)

# useful variables
trains = ['no_occlusion', 'natural', 'natural_silhouette',
          'artificial_1', 'artificial_2']

# occluder dataset
occ_dir = op.expanduser(f'~/data/datasets/images/occluders')
visibilities = np.arange(10, 100, 10)
occluder_sets = {
    'natural': ['natural'],
    'natural_silhouette': ['natural_silhouette'],
    'artificial_1': [
        'horizontal_bars_04',
        'vertical_bars_04',
        'oblique_bars_04',
        'cardinal_crossed_bars',
        'oblique_crossed_bars',
        'mud_splash',
        'polkadot',
        'polkasquare'],
    'artificial_2': [
        'curved_lines',
        'straight_ines',
        'empty_rectangles',
        'filled_rectangles',
        'empty_triangles',
        'filled_triangles',
        'empty_ellipses',
        'filled_ellipses'],
    'artificial_3': [
        'horizontal_bars_02',
        'horizontal_bars_08',
        'horizontal_bars_16',
        'vertical_bars_02',
        'vertical_bars_08',
        'vertical_bars_16',
        'oblique_bars_02',
        'oblique_bars_08',
        'oblique_bars_16',
        'patch_drop',
        'coarse_noise',
        'fine_noise',
        'fine_oriented_noise',
        'pink_noise'],
}

colors = {
    'no_occlusion': {
        'color': 'w', 'edgecolor': 'k', 'linecolor': 'k'},
    'natural': {
        i: 'tab:green' for i in ['color', 'edgecolor', 'linecolor']},
    'natural_silhouette': {
        i: 'tab:brown' for i in ['color', 'edgecolor', 'linecolor']},
    'artificial_1': {
        i: 'tab:blue' for i in ['color', 'edgecolor', 'linecolor']},
    'artificial_2': {
        i: 'tab:red' for i in ['color', 'edgecolor', 'linecolor']},
    'artificial_3': {
        i: 'tab:purple' for i in ['color', 'edgecolor', 'linecolor']},
    }


def likeness_figure():

    from humans.analysis import CFG

    outdir = op.join(figure_dir, f'human_likeness')
    os.makedirs(outdir, exist_ok=True)

    # collate data
    scores_human = pd.read_parquet(f'../humans/trials.parquet')
    scores_human = (
        scores_human[scores_human.visibility < 1]
        #scores_human[(scores_human.visibility < 1) &
        #             (scores_human.visibility >= .4)]
        .groupby(['occluder_class', 'occluder_color'])
        .agg({'accuracy': 'mean'})
        .reset_index()
    )
    scores_model = pd.read_parquet(f'human_trials.parquet')
    scores_model = scores_model[scores_model.visibility < 1].groupby(
        ['occluder_class', 'occluder_color', 'training_occluder']).agg(
        {'value': 'mean'}).reset_index()

    # scatter plots (average models versus average human)

    ylims = {
        'no_occlusion': (0.1, 0.4),
        'natural': (0.2, 0.7),
        'natural_silhouette': (0.4, 0.7),
        'artificial_1': (0.4, 0.7),
        'artificial_2': (0.5, 0.8)
    }
    for training_occluder in scores_model.training_occluder.unique():
        scores = scores_human.copy()
        scores.rename(columns={'accuracy': 'human_accuracy'}, inplace=True)
        scores_model_sub = (
            scores_model[scores_model.training_occluder == training_occluder]
            .groupby(['occluder_class', 'occluder_color'])
            .agg({'value': 'mean'})
            .rename(columns={'value': 'model_accuracy'})
        )
        scores = scores.merge(scores_model_sub, on=['occluder_class', 'occluder_color'])
        # order occluder_class by CFG.occluder_classes
        scores.occluder_class = pd.Categorical(scores.occluder_class,
                      categories=CFG.occluder_classes)
        scores = scores.sort_values(['occluder_class', 'occluder_color'])


        xvals, yvals = scores[['human_accuracy', 'model_accuracy']].values.T
        fig, ax = plt.subplots(figsize=[3.5,3.5])
        ax.scatter(xvals, yvals, color=CFG.plot_colors, clip_on=False)
        ax.set_xlabel(f'human accuracy', size=14)
        ax.set_ylabel(f'DNN accuracy', size=14)
        ax.set_xticks(np.arange(.3, .7, .1))
        ax.set_xlim((.35,.6))
        ax.set_yticks(np.arange(.2, .8, .1))
        ax.set_ylim(ylims[training_occluder])

        # fit linear model
        slope, intercept = np.polyfit(xvals, yvals, 1)
        x = [.1, .8]
        y = np.poly1d(np.polyfit(xvals, yvals, 1))(x)
        plt.plot(x, y, color='k')

        # line of unity
        #plt.plot([.1, .8], [.1, .8], color='k', ls='dotted')

        r = np.corrcoef(xvals, yvals)[0, 1]
        plt.title(f'{training_occluder}: r = {r:.2f}')
        #ax.text(.15, .7, f'r = {r:.2f}\nslope = {slope:.2f}')
        plt.tight_layout()
        plt.savefig(f'{outdir}/{training_occluder}_scatterplot.png')
        plt.close()

    # bar plot (individual and average models versus individual humans)
    human_likeness = pd.read_parquet(f'human_likeness.parquet')
    noise_ceiling = pd.read_csv(f'../humans/noise_ceiling.csv')
    upper_bound = noise_ceiling.upr.mean()
    lower_bound = noise_ceiling.lwr.mean()

    fig, axes = plt.subplots(ncols=6, figsize=(10,3), sharey=True)

    # pooled models
    ax = axes[0]
    for xpos, training_occluder in enumerate(trains):
        color = colors[training_occluder]['color']
        edgecolor = colors[training_occluder]['edgecolor']
        points = (human_likeness
            [human_likeness.training_occluder == training_occluder]
            .groupby('subject')
            .agg({'value': 'mean'})
            ['value'].values)
        sns.stripplot(x=xpos, y=points, zorder=3, clip_on=False,
            native_scale=True, dodge=True, ax=ax, color='tab:grey',
            size=3, alpha=.5, linewidth=0, edgecolor='None')
        ax.bar(xpos, np.mean(points), color=color,
            edgecolor=edgecolor, linewidth=1, zorder=2)
        ax.errorbar(xpos, np.mean(points), stats.sem(points), color='k',
            capsize=4, zorder=4)
    ax.fill_between((-.5, 4.5), lower_bound, upper_bound, color='tab:gray',
        lw=0, zorder=1)
    ax.set_ylabel(r"Pearson's $r$", size=12)
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, -.5, 0, .5, 1])
    ax.grid(axis='y', linestyle='solid', alpha=.25, zorder=0, clip_on=False)
    ax.set_xlim((-.5, 4.5))
    ax.set_xticks([])
    ax.axhline(y=0, color='k', ls='dotted')
    ax.spines['bottom'].set_visible(False)

    # individual models
    model_order = [
        ('cornet_s_plus', 'classification'),
        ('cornet_s_plus', 'simclr'),
        ('resnet101', 'classification'),
        ('efficientnet_b1', 'classification'),
        ('vit_b_16', 'classification')]
    labels = {
        'cornet_s_plus': 'CORnet-S+',
        'resnet101': 'ResNet101',
        'efficientnet_b1': 'EfficientNet-B1',
        'vit_b_16': 'ViT-B/16',
        'classification': 'classification',
        'simclr': 'SimCLR',
    }
    for m, (arch, task) in enumerate(model_order):
        ax = axes[m+1]
        for training_occluder in trains:
            for xpos, training_occluder in enumerate(trains):
                color = colors[training_occluder]['color']
                edgecolor = colors[training_occluder]['edgecolor']
            points = (
                human_likeness[
                    (human_likeness.architecture == arch) &
                    (human_likeness.task == task) &
                    (human_likeness.training_occluder == training_occluder)]
                ['value'].values)
            sns.stripplot(x=xpos, y=points, zorder=3, clip_on=False,
                native_scale=True, dodge=True, ax=ax, color='tab:grey', size=3,
                alpha=.5, linewidth=0, edgecolor='None')
            ax.bar(xpos, np.mean(points), color=color, edgecolor=edgecolor,
                linewidth=1, zorder=2)
            ax.errorbar(xpos, np.mean(points), stats.sem(points), color='k',
                capsize=4, zorder=4)
            ax.fill_between((-.5, 4.5), lower_bound, upper_bound,
                color='tab:gray', lw=0, zorder=1)
        ax.set_xlabel(f'{labels[arch]}\n{labels[task]}', size=10, rotation=0,
            labelpad=10)
        if m > 0:
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', which='major', length=0)
        ax.grid(axis='y', linestyle='solid', alpha=.25, zorder=0, clip_on=False)
        ax.set_xlim((-.5, 4.5))
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)
        ax.axhline(y=0, color='k', ls='dotted')
    plt.tight_layout(pad=1)
    # create a gap between first plot and the rest
    plt.subplots_adjust(wspace=0.1)
    fig.savefig('plots/human_likeness/human_likeness_barplot.pdf')
    plt.close()


def occluder_type():

    # collate data
    robustness = pd.read_parquet('occlusion_robustness.parquet')
    robustness.score = robustness.score.astype(float)

    def _imagenet_occluded(robustness):

        out_dir = op.join(figure_dir, f'occlusion_robustness/ImageNet-Occluded')
        os.makedirs(out_dir, exist_ok=True)

        tests = trains[1:] + ['artificial_3']

        # get name of occluder set for each test occluder
        rob = robustness[
            robustness.benchmark.isin(['ImageNet-1K', 'ImageNet-Occluded'])
        ].copy()
        test_set_list = []
        for i, row in rob.iterrows():
            if row.benchmark in ['ImageNet-1K']:
                test_set = 'no_occlusion'
            else:
                for occluder_set, occluder_types in occluder_sets.items():
                    if row.occluder_type in occluder_types:
                        test_set = occluder_set
                        break
            test_set_list.append(test_set)
        rob['test_set'] = test_set_list
        rob.visibility = rob.visibility.fillna(1.)
        rob.visibility = rob.visibility.astype(float)

        # one plot per training occluder type (visibility - accuracy curves)
        rob_vis = (rob.groupby(
            ['benchmark', 'training_occluder', 'test_set', 'visibility'],
            dropna=False).agg({'score': 'mean'}).reset_index())

        for training_occluder in trains:

            out_path = op.join(out_dir, f'{training_occluder}.pdf')
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
                color, edgecolor, linecolor = [
                    colors[test_set][k] for k in ['color', 'edgecolor',
                                                  'linecolor']]

                # plot data points
                xvals = np.arange(0.1, 1.1, 0.1)
                yvals = (rob_vis[
                     (rob_vis.training_occluder == training_occluder) &
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
            # sub_ax.set_xticks(range(len(trains)), rotation=90, ha='center',
            #                  va='bottom',
            #                  labels=trains, size=9)
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

    def _pascal3d_occluded_objects(robustness):

        out_dir = op.join(figure_dir,
                          f'occlusion_robustness/PASCAL3Dplus_Occluded_Objects')
        os.makedirs(out_dir, exist_ok=True)

        rob = (
            robustness[robustness.benchmark == 'PASCAL3D+_Occluded_Objects']
            .groupby('training_occluder')
            .agg({'score': 'mean'})
            .reset_index())
        assert len(rob)

        # make plot
        out_path = op.join(out_dir, f'accuracy.pdf')
        fig, ax = plt.subplots(figsize=(3, 4), sharey=True)
        for tr, training_occluder in enumerate(trains):
            color, edgecolor, linecolor = [colors[training_occluder][k] for k
                in ['color', 'edgecolor', 'linecolor']]
            value = rob[rob.training_occluder == training_occluder].score.mean()
            ax.bar(tr, value, color=color, edgecolor=edgecolor, zorder=2)
            ax.text(tr, value + .0025, f'{value:.3f}'.replace('0.', '.'),
                    ha='center', va='bottom', color='k', size=9)
            #if training_occluder in ['no_occlusion',
            #                         'natural_silhouette']:
            #    occ_label = training_occluder.replace('_', '\n')
            #else:
            #    occ_label = training_occluder.replace('_', ' ')
            ax.text(tr, .005, training_occluder.replace('_', ' '), ha='center',
                va='bottom', rotation=90, size=12)

        # format plot
        ax.set_xticks([])
        ax.tick_params(**{'length': 0}, axis='x')
        #ax.set_xticks(np.arange(len(training_datasets)), ha='right',
        #              va='top', rotation=45, labels=training_datasets)
        ax.set_yticks(np.arange(0, .2, .05))
        ax.tick_params(direction='in')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel('training occluder', size=14)
        #ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='solid', alpha=.25, zorder=0,
            clip_on=False)
        ax.set_ylabel('accuracy', size=14)
        #ax.set_yticks(np.arange(0, .3, .05))
        ax.set_ylim(0, .15)
        fig.tight_layout()
        plt.savefig(out_path)
        plt.close()

    #_imagenet_occluded(robustness)
    _pascal3d_occluded_objects(robustness)


if __name__ == '__main__':
    main()
