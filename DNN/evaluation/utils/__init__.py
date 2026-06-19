# Created by David Coggan on 2023 07 10
from datetime import datetime

MODEL_BASE = '../models'
BENCHMARK_DIR = 'benchmarking'

def now():
    return datetime.now().strftime("%y/%m/%d %H:%M:%S")


def insert_cycle(activations, batch_size=None):

    output = {}

    # if activations are already in a dict, i.e., not just model outputs
    if type(activations) is dict:
        for layer, activ in activations.items():
            if type(activ) is dict:
                output[layer] = activ
            elif len(activ.shape) == 2:
                output[layer] = {'cyc-1': activ}
            elif len(activ.shape) == 3:
                assert activ.shape[1] < 32, 'you sure dim1 is the cycle dim?'
                output[layer] = {f'cyc{i:02}': activ[:, i, :] for i in
                                 range(activ.shape[1])}
        return output


    # if activations are not in a dict, i.e., just model outputs

    # non-recurrent DNN
    if len(activations.shape) == 2 and batch_size in [
            activations.shape[0], None]:
        output = {'cyc-1': activations}
        return output

    # 2D tensor where batch and cycle are mixed in zeroth dimension
    if batch_size is not None and activations.shape[0] > batch_size:
        cycles = activations.shape[0] // batch_size
        output = {f'cyc{c:02}': activations[c::cycles] for c in range(cycles)}
        return output

    # 3D tensor where cycle is first dimension
    if len(activations.shape) == 3:
        cycles = activations.shape[1]
        assert cycles < 32, 'Cycles > 32, are you sure dim1 is the cycle dim?'
        output = {f'cyc{c:02}': activations[:, c, :] for c in range(cycles)}
        return output

    Exception('Did not find a way to insert cycles into the activations. ')


 # if activations are already separated by cycle
distinct_colors_255 = {
    'red': (230, 25, 75),
    'green': (60, 180, 75),
    'yellow': (255, 225, 25),
    'blue': (0, 130, 200),
    'orange': (245, 130, 48),
    'purple': (145, 30, 180),
    'cyan': (70, 240, 240),
    'magenta': (240, 50, 230),
    'lime': (210, 245, 60),
    'pink': (250, 190, 212),
    'teal': (0, 128, 128),
    'lavender': (220, 190, 255),
    'brown': (170, 110, 40),
    'beige': (255, 250, 200),
    'maroon': (128, 0, 0),
    'mint': (170, 255, 195),
    'olive': (128, 128, 0),
    'apricot': (255, 215, 180),
    'navy': (0, 0, 128),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'gray': (127, 127, 127)}

distinct_colors = {k: tuple([x / 255. for x in v])
                   for k, v in distinct_colors_255.items()}
