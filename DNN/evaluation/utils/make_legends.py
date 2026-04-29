'''
Makes legends for the model contrasts
'''

import os.path as op
import os
from itertools import product as itp
from itertools import combinations as itc
from .model_contrasts import model_contrasts, HUMAN_CONFIG
from .plot_utils import make_legend

def make_legends(overwrite=False):

    for (model_contrast, model_config) in model_contrasts.items():
        legend_dir = (f'../p022_occlusion/data/in_silico/analysis'
                      f'/{model_contrast}/legends')
        all_levels = [['level1'], ['level2'], ['level1','level2']]
        all_properties = [
            ['color'],
            ['color', 'edgecolor'],
            ['color', 'marker'],
            ['color', 'edgecolor', 'marker'],
            ['color', 'edgecolor', 'marker', 'linestyle']]
        default_properties = {
            'edgecolor': 'None',
            'linestyle': 'None',
            'marker': 's',
        }
        for humans, levels, properties in itp(
                [True, False], all_levels, all_properties):

            outpath = (f'{legend_dir}/'
                       f'{"-".join(levels)}_{"-".join(properties)}.svg')
            if humans:
                outpath = outpath.replace('.svg', '_humans.svg')
            if not op.isfile(outpath) or overwrite:
                os.makedirs(legend_dir, exist_ok=True)

                legend_properties = {i: [] for i in list(set(
                    properties + list(default_properties.keys())))}
                labels = []

                # humans
                if humans:
                    labels.append('humans')
                    [legend_properties[k].append(HUMAN_CONFIG['humans'][k])
                        for k in properties]
                    for property in legend_properties:
                        if property not in properties:
                            legend_properties[property].append(
                                default_properties[property])

                # DNN
                for level1, level2s in model_config.items():
                    for level2, config in level2s.items():
                        if len(levels) > 1:
                            label = ', '.join([level1, level2])
                        elif levels[0] == 'level1':
                            label = level1
                        elif levels[0] == 'level2':
                            label = level2
                        if label not in labels:
                            labels.append(label)
                            for property in legend_properties:
                                if property in properties:
                                    legend_properties[property].append(
                                            config[property])
                                else:
                                    legend_properties[property].append(
                                        default_properties[property])

                make_legend(
                    outpath=outpath,
                    labels=labels,
                    markers=legend_properties['marker'],
                    colors=legend_properties['color'],
                    markeredgecolors=legend_properties['edgecolor'],
                    linestyles=legend_properties['linestyle'])

                make_legend(
                    outpath=outpath.replace('.svg', '.pdf'),
                    labels=labels,
                    markers=legend_properties['marker'],
                    colors=legend_properties['color'],
                    markeredgecolors=legend_properties['edgecolor'],
                    linestyles=legend_properties['linestyle'])


