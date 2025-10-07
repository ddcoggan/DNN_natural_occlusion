# Image Classification experiment
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import matplotlib
import os.path as op
TAB20 = matplotlib.cm.tab20.colors
from .utils import custom_defaults, make_legend
from .utils import sigmoid
from itertools import product as itp
plt.rcParams.update(custom_defaults)
from scipy.optimize import curve_fit

trials_path = 'trials.parquet'
curve_params_path = 'robustness_curves.parquet'
noise_ceiling_path = 'noise_ceiling.csv'

def main():
    #get_human_data()
    #fit_performance_curves()
    make_plots()
    #calculate_noise_ceiling()


class CFG:

    """
    This class contains useful details about the stimulus conditions and
    commonly used objects and constants, e.g. condition-color mappings, to keep
    plots consistent across human and model analysis pipelines.
    """

    # object classes
    classes_orig = [
        'brown bear, bruin, Ursus arctos',
        'bison',
        'African elephant, Loxodonta africana',
        'hare',
        'jeep, landrover',
        'table lamp',
        'sports car, sport car',
        'teapot']
    object_classes = ['bear', 'bison', 'elephant', 'hare',
               'jeep', 'lamp', 'sportsCar', 'teapot']
    class_idxs = [294, 347, 386, 331, 609, 846, 817, 849]
    synsets = ['n02132136', 'n02410509', 'n02504458', 'n02326432',
               'n03594945', 'n04380533', 'n04285008', 'n04398044']
    animate_classes = object_classes[:4]
    inanimate_classes = object_classes[4:8]

    # occluders
    occluder_classes = [
        'horizontal_bars_04', 'vertical_bars_04', 'oblique_bars_04',
        'cardinal_crossed_bars', 'oblique_crossed_bars',
        'polkadot', 'polkasquare', 'mud_splash', 'natural_silhouette']

    visibilities = [.1, .2, .4, .6, .8]
    occluder_colors = ['black', 'white']
    plot_colors = [TAB20[i] for i in [
        0, 1, 2, 3, 6, 7, 16, 17, 10, 11, 8, 9, 12, 13, 18, 19, 4, 5]]
    plot_ecolors = [TAB20[i] for i in [
        1, 0, 3, 2, 7, 6, 17, 16, 11, 10, 9, 8, 13, 12, 19, 18, 5, 4]]

    # all condition combinations
    occ_vis_combos = itp(occluder_classes, visibilities)
    occ_vis_combos = [*[('none', 1.)], *occ_vis_combos]
    occ_col_combos = itp(occluder_classes, occluder_colors)
    occ_col_labels = [', '.join([o, c]) for o, c in itp(occluder_classes,
                                                        occluder_colors)]
    cond_combos = itertools.product(occluder_classes, occluder_colors, visibilities)
    cond_combos = [*[('none', 'none', 1.)], *cond_combos]
    """
    scripts to make labels by loading imagenet label file
    label_data = pd.read_csv(open(op.expanduser(
        '~/david/datasets/images/ILSVRC2012/labels.csv'), 'r+'))
    class_idxs = [
        label_data['index'][label_data['directory'] == class_dir].item() for
        class_dir in class_dirs]
    class_labels_alt = [
        label_data['label'][label_data['directory'] == class_dir].item() for
        class_dir in class_dirs]
    """
    subjects = [f'sub-{i:02}' for i in range(30)]



def fit_performance_curves():

    all_trials = pd.read_parquet(trials_path)
    curve_params = pd.DataFrame()

    for level, subject_sample in zip(
            ['group', 'individual'], [['group'], CFG.subjects]):

        if level == 'individual':
            performance = (all_trials.groupby(
                ['subject', 'occluder_class', 'occluder_color',
                 'visibility'], dropna=False)
               .agg('mean', numeric_only=True).reset_index())
        else:  # if level == 'group'
            performance = (all_trials.groupby(
                ['occluder_class', 'occluder_color', 'visibility'], dropna=False)
                .agg('mean', numeric_only=True).reset_index())
            performance['subject'] = 'group'

        # condition-wise curve params and thresholds
        for subject, occluder_class, occluder_color in itertools.product(
                subject_sample, CFG.occluder_classes, CFG.occluder_colors):

            perf_unocc = performance[
                (performance.visibility == 1) &
                (performance.subject == subject)].accuracy.item()

            perf_occ = (performance[
                (performance.occluder_class == occluder_class) &
                (performance.occluder_color == occluder_color) &
                (performance.subject == subject)]
                .sort_values('visibility').accuracy.to_list())

            xvals = [0] + CFG.visibilities + [1]
            yvals = np.array([1/8] + list(perf_occ) + [perf_unocc])

            init_params = [max(yvals), np.median(xvals), 1, 0]
            curve_x = np.linspace(0, 1, 1000)
            try:
                popt, pcov = curve_fit(
                    sigmoid, xvals, yvals, init_params, maxfev=100000)
                curve_y = sigmoid(curve_x, *popt)
                threshold = sum(curve_y < .5) / 1000
            except:
                popt = [np.nan] * 4
                threshold = np.nan

            curve_params = pd.concat(
                [curve_params, pd.DataFrame({
                    'subject': [str(subject)],
                    'occluder_class': [occluder_class],
                    'occluder_color': [occluder_color],
                    'metric': ['accuracy'],
                    'L': [popt[0]],
                    'x0': [popt[1]],
                    'k': [popt[2]],
                    'b': [popt[3]],
                    'threshold_50': [threshold],
                    'mean': [np.mean(perf_occ)]
                })]).reset_index(drop=True)

        # mean curve across all conditions
        for dataset, subject in itp(['all', 'artificial'], subject_sample):

            perf = performance[performance.subject == subject]
            if dataset == 'artificial':
                perf = perf[perf.occluder_class != 'naturalUntexturedCropped2']
            perf = (perf
            .groupby('visibility')
            .agg('mean', numeric_only=True)
            .sort_values('visibility').accuracy.to_list())

            xvals = [0] + CFG.visibilities + [1]
            yvals = np.array([1/8] + perf)

            init_params = [max(yvals), np.median(xvals), 1, 0]
            curve_x = np.linspace(0, 1, 1000)
            try:
                popt, pcov = curve_fit(
                    sigmoid, xvals, yvals, init_params, maxfev=100000)
                curve_y = sigmoid(curve_x, *popt)
                threshold = sum(curve_y < .5) / 1000
            except:
                popt = [np.nan] * 4
                threshold = np.nan

            curve_params = pd.concat(
                [curve_params, pd.DataFrame({
                    'subject': [str(subject)],
                    'occluder_class': [dataset],
                    'occluder_color': ['all'],
                    'metric': ['accuracy'],
                    'L': [popt[0]],
                    'x0': [popt[1]],
                    'k': [popt[2]],
                    'b': [popt[3]],
                    'threshold_50': [threshold],
                    'mean': [np.mean(perf[:-1])]
                })]).reset_index(drop=True)


    curve_params.subject = curve_params.subject.astype('category')
    curve_params.to_parquet(curve_params_path, index=False)


def condwise_robustness_plot_array(df, outpath, ylabel,
    df_curves=None, yticks=(0,1), ylims=(0,1), chance=None, legend_path=None):

    """
    This function makes a 3x3 array of visibility-accuracy plots, one for
    each occluder type.
    """

    fig, axes = plt.subplots(
        3, 3, figsize=(3.5, 3.5), sharex=True, sharey=True)

    xvals = CFG.visibilities
    perf_unocc = df[df.visibility == 1].accuracy.mean()

    for o, occluder_class in enumerate(CFG.occluder_classes):
        ax = axes.flatten()[o]

        for c, occluder_color in enumerate(CFG.occluder_colors):
            face_color = CFG.plot_colors[o * 2 + c]
            edge_color = CFG.plot_colors[o * 2 + c]

            # plot curve function underneath
            if df_curves is not None:
                if 'subject' in df_curves.columns:
                    df_curves = df_curves[df_curves.subject == 'group']
                popt = (df_curves[
                    (df_curves.metric == 'accuracy') &
                    (df_curves.occluder_class == occluder_class) &
                    (df_curves.occluder_color == occluder_color)]
                    [['L', 'x0', 'k', 'b']]).values[0]
                if np.isfinite(popt).all():
                    curve_x = np.linspace(0, 1, 1000)
                    curve_y = sigmoid(curve_x, *popt)
                    ax.plot(curve_x, curve_y, color=edge_color, zorder=1)

            # plot accuracies on top
            perf_occ = df[
                (df.occluder_class == occluder_class) &
                (df.occluder_color == occluder_color)]
            yvals = perf_occ.groupby('visibility').accuracy.mean().to_list()
            ax.scatter(xvals, yvals, facecolor=face_color, clip_on=False,
                       edgecolor=edge_color, zorder=2)

        # plot unoccluded
        ax.scatter(1, perf_unocc, color='white',
                   edgecolor='k', zorder=3, clip_on=False)

        # format plot
        #ax.set_title(CFG.occluder_classes[o], size=7)
        ax.set_xticks((0, 1))
        ax.set_xlim((0, 1))
        ax.set_yticks(yticks)
        ax.set_ylim(ylims)
        ax.tick_params(axis='both', which='major', labelsize=7)
        # ax.axhline(y=acc1unalt, color=colors[0], linestyle='dashed')
        if chance is not None:
            ax.axhline(y=chance, color='k', linestyle='dotted')
        if o == 7:
            ax.set_xlabel('visibility', size=10)
        if o == 3:
            ax.set_ylabel(ylabel, size=10)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

    if legend_path:
        make_legend(
            outpath=legend_path,
            labels=['unoccluded'] + CFG.occ_col_labels,
            markers='o',
            colors=['w'] + CFG.plot_colors,
            markeredgecolors=None,
            linestyles='None')



def condwise_robustness_plot_array_long(
        df, outpath, ylabel, df_curves=None, yticks=(0,1), ylims=(0,1),
        chance=None, legend_path=None, sizes=['small']):

    """ This function makes a 1x9 array of visibility-accuracy plots, one for
    each occluder type. """

    size_settings = {
        'small': (9, 1.4),
        'large': (15, 2.2),
    }

    for size in sizes:
        if size == 'large':
            outpath = outpath.replace('.pdf', '_large.pdf')

        fig, axes = plt.subplots(1, 9, figsize=size_settings[size], sharey=True)

        xvals = CFG.visibilities
        perf_unocc = df[df.visibility == 1].accuracy.mean()

        for o, occluder_class in enumerate(CFG.occluder_classes):
            ax = axes[o]

            for c, occluder_color in enumerate(CFG.occluder_colors):
                face_color = CFG.plot_colors[o * 2 + c]
                edge_color = CFG.plot_colors[o * 2 + c]

                # plot curve function underneath
                if df_curves is not None:
                    if 'subject' in df_curves.columns:
                        df_curves = df_curves[df_curves.subject == 'group']
                    popt = (df_curves[
                        (df_curves.metric == 'accuracy') &
                        (df_curves.occluder_class == occluder_class) &
                        (df_curves.occluder_color == occluder_color)]
                        [['L', 'x0', 'k', 'b']]).values[0]
                    if np.isfinite(popt).all():
                        curve_x = np.linspace(0, 1, 1000)
                        curve_y = sigmoid(curve_x, *popt)
                        ax.plot(curve_x, curve_y, color=edge_color, zorder=1)

                # plot accuracies on top
                perf_occ = df[
                    (df.occluder_class == occluder_class) &
                    (df.occluder_color == occluder_color)]
                yvals = perf_occ.groupby('visibility').accuracy.mean().to_list()
                ax.scatter(xvals, yvals, facecolor=face_color, clip_on=False,
                           edgecolor=edge_color, zorder=2)

            # plot unoccluded
            ax.scatter(1, perf_unocc, color='white',
                       edgecolor='k', zorder=3, clip_on=False)

            # format plot
            #ax.set_title(CFG.occluder_classes[o], size=7)
            ax.set_xticks((0, 1))
            ax.set_xlim((0, 1))
            ax.set_yticks(yticks)
            ax.set_ylim(ylims)
            ax.tick_params(axis='both', which='major', labelsize=7)
            # ax.axhline(y=acc1unalt, color=colors[0], linestyle='dashed')
            if chance is not None:
                ax.axhline(y=chance, color='k', linestyle='dotted')
            if o == 4:
                ax.set_xlabel('visibility', size=10)
            if o == 0:
                ax.set_ylabel(ylabel, size=10)

        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()

    if legend_path:
        make_legend(
            outpath=legend_path,
            labels=['unoccluded'] + CFG.occ_col_labels,
            markers='o',
            colors=['w'] + CFG.plot_colors,
            markeredgecolors=None,
            linestyles='None')


def make_plots():

    os.makedirs('plots', exist_ok=True)
    trials = (
        pd.read_parquet(trials_path)
        .groupby(['occluder_class', 'occluder_color', 'visibility'])
        .agg('mean', numeric_only=True)
        .reset_index()
    )
    curve_params = pd.read_parquet(curve_params_path)

    condwise_robustness_plot_array_long(
        df=trials,
        df_curves=curve_params,
        outpath='plots/accuracy.pdf',
        ylabel='classification accuracy',
        yticks=(0, 1),
        ylims=(0, 1),
        chance=1 / 8,
        legend_path='plots/legend.pdf',
        sizes=['large'])


def calculate_noise_ceiling():

    """
    This function calculates the noise-ceiling, i.e., between-subject
    reliability of performance across conditions. This is
    calculated as the mean correlation between each participant's performance
    profile and that of the remaining group (lower bound) and entire group
    (upper bound).
    """

    trials = pd.read_parquet(trials_path)
    nc_df = pd.DataFrame()
    groupby = ['occluder_class', 'occluder_color']

    for subject in CFG.subjects:

        # individual performance profile
        trials_subject = trials[trials.subject == subject].copy(deep=True)
        trials_subject.rename(columns={'accuracy': 'subject_accuracy'},
                              inplace=True)
        trials_subject_occ = (trials_subject[trials_subject.visibility < 1]
                              .groupby(groupby + ['subject'])
                              .agg('mean', numeric_only=True)
                              .reset_index())

        # remaining group performance profile (for lower bound)
        trials_rem_grp = trials[trials.subject != subject].copy(deep=True)
        trials_rem_grp.rename(columns={'accuracy': 'group_accuracy'},
                              inplace=True)
        trials_rem_grp_occ = (trials_rem_grp[trials_rem_grp.visibility < 1]
                              .groupby(groupby)
                              .agg('mean', numeric_only=True)
                              .reset_index())

        # group performance profile (for upper bound)
        trials_grp = trials.copy(deep=True)
        trials_grp.rename(columns={'accuracy': 'group_accuracy'},
                          inplace=True)

        trials_grp_occ = (trials_grp[trials_grp.visibility < 1]
                              .groupby(groupby)
                              .agg('mean', numeric_only=True)
                              .reset_index())

        # condition-wise accuracy correlation
        subject_trials = trials_subject_occ[
            groupby + ['subject_accuracy']]
        rem_grp_trials = trials_rem_grp_occ[
            groupby + ['group_accuracy']]
        grp_trials = trials_grp_occ[
            groupby + ['group_accuracy']]
        subject_rem_grp = (subject_trials
            .merge(rem_grp_trials, on=groupby)
            .dropna()
            .groupby(groupby)
            .agg('mean', numeric_only=True))
        subject_grp = (subject_trials
            .merge(grp_trials, on=groupby)
            .dropna()
            .groupby(groupby)
            .agg('mean', numeric_only=True))

        lwr = np.corrcoef(subject_rem_grp.subject_accuracy,
                          subject_rem_grp.group_accuracy)[0, 1]
        upr = np.corrcoef(subject_grp.subject_accuracy,
                          subject_grp.group_accuracy)[0, 1]
        nc_df = pd.concat([nc_df, pd.DataFrame(dict(
            subject=[subject],
            metric=['accuracy'],
            level=['condition-wise'],
            metric_sim=['cond_pearson_r'],
            lwr=[lwr],
            upr=[upr]))])

    nc_df = nc_df.sort_values(by='subject').reset_index(drop=True)
    nc_df.to_csv(noise_ceiling_path, index=False)


if __name__ == '__main__':
    main()





    



