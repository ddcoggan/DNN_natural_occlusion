import os
import os.path as op
import sys
from types import SimpleNamespace

import torch.nn as nn
import math
from pathlib import Path
from torchvision import models

def get_model(architecture, kwargs):

	if architecture in models.list_models():
		try:
			model = getattr(models, architecture)(**kwargs)
		except:
			ValueError('kwargs not accepted for this model')
	elif architecture == 'cornet_s_plus':
		from ..cornet_s_plus import CORnet_S_custom as cornet_s_plus
		try:
			model = cornet_s_plus(**kwargs)
		except:
			try:
				model = cornet_s_plus(kwargs)
			except:
				ValueError('kwargs not accepted for this model')

	return model
