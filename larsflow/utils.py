import torch
import numpy as np

import yaml
import os


def get_config(path):
    """
    Read configuration parameter form file
    :param path: Path to the yaml configuration file
    :return: Dict with parameter
    """

    with open(path, 'r') as stream:
        return yaml.load(stream, yaml.FullLoader)


def get_latest_checkpoint(dir_path, key=''):
    """
    Get path to latest checkpoint in directory
    :param dir_path: Path to directory to search for checkpoints
    :param key: Key which has to be in checkpoint name
    :return: Path to latest checkpoint
    """
    if not os.path.exists(dir_path):
        return None
    checkpoints = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if
                   os.path.isfile(os.path.join(dir_path, f)) and key in f and ".pt" in f]
    if len(checkpoints) == 0:
        return None
    checkpoints.sort()
    return checkpoints[-1]


def bitsPerDim(model, x, y=None, trans='logit', trans_param=[0.05]):
    """
    Computes the bits per dim for a batch of data, support DataParallel models
    :param model: Model to compute bits per dim for
    :param x: Batch of data
    :param y: Class labels for batch of data if base distribution is class conditional
    :param trans: Transformation to be applied to images during training
    :param trans_param: List of parameters of the transformation
    :return: Bits per dim for data batch under model
    """
    dims = torch.prod(torch.tensor(x.size()[1:]))
    if trans == 'logit':
        if y is None:
            log_q = -model(x)
        else:
            log_q = -model(x, y)
        sum_dims = list(range(1, x.dim()))
        ls = torch.nn.LogSigmoid()
        sig_ = torch.sum(ls(x) / np.log(2), sum_dims)
        sig_ += torch.sum(ls(-x) / np.log(2), sum_dims)
        b = - log_q / dims / np.log(2) - np.log2(1 - trans_param[0]) + 8
        b += sig_ / dims
    else:
        raise NotImplementedError('The transformation ' + trans + ' is not implemented.')
    return b


def bitsPerDimDataset(model, data_loader, class_cond=True, trans='logit',
                      trans_param=[0.05]):
    """
    Computes average bits per dim for an entire dataset given by a data loader
    :param model: Model to compute bits per dim for
    :param data_loader: Data loader of dataset
    :param class_cond: Flag indicating whether model is class_conditional
    :param trans: Transformation to be applied to images during training
    :param trans_param: List of parameters of the transformation
    :return: Average bits per dim for dataset
    """
    n = 0
    b_cum = 0
    with torch.no_grad():
        for x, y in iter(data_loader):
            b_ = bitsPerDim(model, x, y.to(x.device) if class_cond else None,
                            trans, trans_param)
            b_np = b_.to('cpu').numpy()
            b_cum += np.nansum(b_np)
            n += len(x) - np.sum(np.isnan(b_np))
        b = b_cum / n
    return b


class ToDouble():
    """
    Transform for dataloader casting input to double
    """
    def __init__(self):
        pass

    def __call__(self, x):
        return x.double()