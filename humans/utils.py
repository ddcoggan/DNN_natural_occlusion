import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x, L, x0, k, b):
    """
    :param x: input values
    :param L: L is responsible for scaling the output range from [0,1] to [0,L]
    :param b: b adds bias to the output and changes its range from [0,L] to
        [b,L+b]
    :param k: k is responsible for scaling the input, which remains in
        (-inf,inf)
    :param x0: x0 is the point in the middle of the Sigmoid, i.e. the point
        where Sigmoid should originally output the value 1/2 [since if x=x0,
        we get 1/(1+exp(0)) = 1/2].
    :return: output values
    """
    y_hat = L / (1 + np.exp(-k * (x - x0))) + b
    return y_hat


def export_legend(legend, filename="legend.pdf"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi=300, bbox_inches=bbox)
    plt.close()


def make_legend(outpath, labels, markers=None, colors=None,
                markeredgecolors=None, linestyles=None):
    # put properties into dict with defaults
    p = {'m': ['o'] * len(labels), 'c': ['k'] * len(labels),
        'mec': [None] * len(labels), 'ls': ['solid'] * len(labels)}

    # update properties with user input
    for inp, (property, default) in zip(
            [markers, colors, markeredgecolors, linestyles], p.items()):
        if inp is not None:
            if type(inp) is str:  # if single item, repeat for each label
                inp = [inp] * len(labels)
            p[property] = inp

    # make legend
    handles = [plt.plot([], [], marker=m, c=c, mec=mec, ls=ls)[0] for
        m, c, mec, ls in zip(p['m'], p['c'], p['mec'], p['ls'])]
    legend = plt.legend(handles, labels, loc=3)
    export_legend(legend, filename=outpath)
    plt.close()


custom_defaults = {'font.size': 10, 'lines.linewidth': 1,
    'lines.markeredgewidth': 1, 'lines.markersize': 6, 'savefig.dpi': 300,
    'legend.frameon': False, 'ytick.direction': 'in', 'ytick.major.width': .8,
    'xtick.direction': 'in', 'xtick.major.width': .8, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': .8}
