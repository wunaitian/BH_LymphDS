import json
import os

import yaml
from . import models

_image_backend = 'PIL'


def get_config(directory=os.getcwd(), config_file='config.txt') -> dict:
    if os.path.exists(os.path.join(directory, config_file)):
        with open(os.path.join(directory, config_file), encoding='utf8') as c:
            content = c.read()
            if '\\\\' not in content:
                content = content.replace('\\', '\\\\')
            if config_file.endswith('.txt'):
                config = json.loads(content)
            elif config_file.endswith('.yaml'):
                config = yaml.load(content, Loader=yaml.FullLoader)
            return config or {}
    else:
        return {}


def get_param_in_cwd(param: str, default=None, **kwargs):
    directory = kwargs.get('directory', os.getcwd())
    config_file = 'config.yaml' if os.path.exists(os.path.join(directory, 'config.yaml')) else 'config.txt'
    config = get_config(directory, config_file)
    ret = config.get(param, default)
    return ret

def set_image_backend(backend):
    """
    Specifies the package used to load images.

    Args:
        backend (string): Name of the image backend. one of {'PIL', 'accimage'}.
            The :mod:`accimage` package uses the Intel IPP library. It is
            generally faster than PIL, but does not support as many operations.
    """
    global _image_backend
    if backend not in ['PIL', 'accimage']:
        raise ValueError("Invalid backend '{}'. Options are 'PIL' and 'accimage'"
                         .format(backend))
    _image_backend = backend


def get_image_backend():
    """
    Gets the name of the package used to load images
    """
    return _image_backend
