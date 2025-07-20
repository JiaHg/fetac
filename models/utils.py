import os
from . import resnet
from . import conv_4
import torch
def get_backbone_and_load_dict(model_name='resnet18',model_weight_path=None, modelpath=None, device=None):
    if model_name == 'resnet18':
        model = resnet.resnet18()
        pretrained_weights = torch.load(os.path.join(modelpath, model_weight_path), map_location=device)
        pretrained_weights.pop('fc.weight')
        pretrained_weights.pop('fc.bias')
        model.load_state_dict(pretrained_weights, strict=False)
    elif model_name == 'conv4':
        model = conv_4.Conv64F()
        pretrained_weights = torch.load(os.path.join(modelpath, model_weight_path), map_location=device)
        pretrained_weights.pop('linear_c.weight')
        pretrained_weights.pop('linear_c.bias')
        model.load_state_dict(pretrained_weights, strict=False)
    else:
        pass

    return model