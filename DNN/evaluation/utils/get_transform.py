# Created by David Coggan on 2025 01 06
import torchvision
import torchvision.transforms.v2 as transforms
from torch import float32

def get_transform(architecture, model_dir):

    if hasattr(torchvision.models, architecture) and 'pretrained' in model_dir:
        model_attr = str([i for i in torchvision.models.__dict__ if
                             i.lower() == f'{architecture}_weights'][0])
        weights_attr = model_dir.split('pretrained_')[-1]
        transform = getattr(getattr(torchvision.models, model_attr),
                            weights_attr).transforms()
        return transform

    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(float32, scale=True),
        transforms.Resize(224),
        transforms.CenterCrop(224),  # in case resize is off by a pixel
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform