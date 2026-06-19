import os
import os.path as op
import sys
from types import SimpleNamespace

import torch.nn as nn
import math
from pathlib import Path
from torchvision import models
from DNN import cornet_s_plus


def get_model(architecture, kwargs):

	if architecture in models.list_models():
		try:
			model = getattr(models, architecture)(**kwargs)
		except:
			ValueError('kwargs not accepted for this model')
	elif architecture == 'cornet_s_plus':
		try:
			model = cornet_s_plus(**kwargs)
		except:
			ValueError('kwargs not accepted for this model')
	else:
		raise ValueError('architecture not recognized')

	return model
