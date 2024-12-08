
import json
import os
from typing import Iterable
import numpy as np

import torch


from .image_loader import default_loader
from .about_log import logger
from BH_LymphDS.core import create_model
from BH_LymphDS.core import create_standard_image_transformer

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def extract(samples, model, transformer, device=None, fp=None):
    results = []
    # Inference
    if not isinstance(samples, (list, tuple)):
        samples = [samples]
    with torch.set_grad_enabled(False):
        for sample in samples:
            fp.write(f"{os.path.basename(sample)},")
            sample_ = transformer(default_loader(sample))
            sample_ = sample_.to(device)
            # print(sample_.size())
            outputs = model(sample_.view(1, *sample_.size()))
            results.append(outputs)
    return results

def print_feature_hook(module, inp, outp, fp, post_process=None):
    features = outp.cpu().numpy()
    if post_process is not None:
        features = post_process(features)
    print(','.join(map(lambda x: f"{x:.6f}", np.reshape(features, -1))), file=fp)


def reg_hook_on_module(name, model, hook):
    handles = []
    find_ = 0
    for n, m in model.named_modules():
        if name == n:
            handle = m.register_forward_hook(hook)
            handles.append(handle)
            find_ += 1
    if find_ == 0:
        logger.warning(f'{name} not found in {model}')
    elif find_ > 1:
        logger.info(f'Found {find_} features named {name} in {model}')
    return handles


def init_from_model(config_path):
    config = json.loads(open(os.path.join(config_path, 'task.json')).read())
    model_path = os.path.join(config_path, 'BH_LymphDS.pth')
    assert 'model_name' in config and 'num_classes' in config and 'transform' in config
    # Configuration of transformer.
    transform_config = {'phase': 'valid'}
    transform_config.update(config['transform'])
    assert 'input_size' in transform_config, '`input_size` must in `transform`'
    transformer = create_standard_image_transformer(**transform_config)

    # Configuration of core
    # model_config = {'pretrained': False, 'model_name': config['model_name'], 'num_classes': config['num_classes']}
    model_config = {'model_name': config['model_name'], 'num_classes': config['num_classes']}
    model = create_model(**model_config)
    # Configuration of device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    state_dict = torch.load(model_path, map_location=device)['model_state_dict']
    for key in list(state_dict.keys()):
        if key.startswith('module.'):
            new_key = key[7:]
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)
    model.eval()
    return model, transformer, device


def remove_hooks(hook_handles):
    for handle in hook_handles:
        handle.remove()

