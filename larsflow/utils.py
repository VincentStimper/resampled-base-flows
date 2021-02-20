import torch
import numpy as np

from matplotlib import pyplot as plt

import yaml
import os

# Try importing Boltzmann generator dependencies
try:
    import boltzgen as bg
    import mdtraj
except:
    print('Warning: Dependencies for Boltzmann generators could '
          'not be loaded. Other models can still be used.')


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


def evaluateAldp(model, test_data, n_samples=1000, n_batches=100,
                 save_path=None, data_path='.'):
    """
    Evaluate model of the Boltzmann distribution of the Alanine
    Dipeptide
    :param model: Model to be evaluated
    :param test_data: Torch array with test data
    :param n_samples: Int, number of samples to draw per batch
    :param n_batches: Int, number of batches to sample
    :param save_path: String, path where to save plots of marginals,
    if none plots are not created
    :param data_path: String, path to data used for transform init
    :return: KL divergences
    """
    # Set params for transform
    ndim = 66
    z_matrix = [
        (0, [1, 4, 6]),
        (1, [4, 6, 8]),
        (2, [1, 4, 0]),
        (3, [1, 4, 0]),
        (4, [6, 8, 14]),
        (5, [4, 6, 8]),
        (7, [6, 8, 4]),
        (11, [10, 8, 6]),
        (12, [10, 8, 11]),
        (13, [10, 8, 11]),
        (15, [14, 8, 16]),
        (16, [14, 8, 6]),
        (17, [16, 14, 15]),
        (18, [16, 14, 8]),
        (19, [18, 16, 14]),
        (20, [18, 16, 19]),
        (21, [18, 16, 19])
    ]
    cart_indices = [6, 8, 9, 10, 14]

    # Load data for transform
    # Load the alanine dipeptide trajectory
    traj = mdtraj.load(data_path)
    traj.center_coordinates()

    # superpose on the backbone
    ind = traj.top.select("backbone")

    traj.superpose(traj, 0, atom_indices=ind, ref_atom_indices=ind)

    # Gather the training data into a pytorch Tensor with the right shape
    training_data = traj.xyz
    n_atoms = training_data.shape[1]
    n_dim = n_atoms * 3
    training_data_npy = training_data.reshape(-1, n_dim)
    training_data = torch.from_numpy(training_data_npy.astype("float64"))

    # Set up transform
    transform = bg.flows.CoordinateTransform(training_data, ndim,
                                             z_matrix, cart_indices)

    z_np = np.zeros((0, 60))

    for i in range(n_batches):
        z, _ = model.sample(n_samples)
        x, _ = transform(z.cpu().double())
        z, _ = transform.inverse(x)
        z_np = np.concatenate((z_np, z.data.numpy()))

    z_d_np = test_data.cpu().data.numpy()

    # Estimate density
    nbins = 200
    hist_range = [-5, 5]
    ndims = z_np.shape[1]

    hists_train = np.zeros((nbins, ndims))
    hists_gen = np.zeros((nbins, ndims))

    for i in range(ndims):
        htrain, _ = np.histogram(z_d_np[:, i], nbins, range=hist_range, density=True);
        hgen, _ = np.histogram(z_np[:, i], nbins, range=hist_range, density=True);

        hists_train[:, i] = htrain
        hists_gen[:, i] = hgen

    # Compute KLD
    eps = 1e-10
    kld_unscaled = np.sum(hists_train * np.log((hists_train + eps) / (hists_gen + eps)), axis=0)
    kld = kld_unscaled * (hist_range[1] - hist_range[0]) / nbins

    # Split KLD into groups
    ncarts = transform.mixed_transform.len_cart_inds
    permute_inv = transform.mixed_transform.permute_inv
    bond_ind = transform.mixed_transform.ic_transform.bond_indices
    angle_ind = transform.mixed_transform.ic_transform.angle_indices
    dih_ind = transform.mixed_transform.ic_transform.dih_indices

    kld_cart = kld[:(3 * ncarts - 6)]
    kld_ = np.concatenate([kld[:(3 * ncarts - 6)], np.zeros(6), kld[(3 * ncarts - 6):]])
    kld_ = kld_[permute_inv]
    kld_bond = kld_[bond_ind]
    kld_angle = kld_[angle_ind]
    kld_dih = kld_[dih_ind]

    if save_path is not None:
        # Histograms of the groups
        hists_train_cart = hists_train[:, :(3 * ncarts - 6)]
        hists_train_ = np.concatenate([hists_train[:, :(3 * ncarts - 6)], np.zeros((nbins, 6)),
                                       hists_train[:, (3 * ncarts - 6):]], axis=1)
        hists_train_ = hists_train_[:, permute_inv]
        hists_train_bond = hists_train_[:, bond_ind]
        hists_train_angle = hists_train_[:, angle_ind]
        hists_train_dih = hists_train_[:, dih_ind]

        hists_gen_cart = hists_gen[:, :(3 * ncarts - 6)]
        hists_gen_ = np.concatenate([hists_gen[:, :(3 * ncarts - 6)], np.zeros((nbins, 6)),
                                     hists_gen[:, (3 * ncarts - 6):]], axis=1)
        hists_gen_ = hists_gen_[:, permute_inv]
        hists_gen_bond = hists_gen_[:, bond_ind]
        hists_gen_angle = hists_gen_[:, angle_ind]
        hists_gen_dih = hists_gen_[:, dih_ind]

        label = ['cart', 'bond', 'angle', 'dih']
        hists_train_list = [hists_train_cart, hists_train_bond, hists_train_angle, hists_train_dih]
        hists_gen_list = [hists_gen_cart, hists_gen_bond, hists_gen_angle, hists_gen_dih]
        x = np.linspace(*hist_range, nbins)
        for i in range(4):
            if i == 0:
                fig, ax = plt.subplots(3, 3, figsize=(10, 10))
            else:
                fig, ax = plt.subplots(6, 3, figsize=(10, 20))
                ax[5, 2].set_axis_off()
            for j in range(hists_train_list[i].shape[1]):
                ax[j // 3, j % 3].plot(x, hists_train_list[i][:, j])
                ax[j // 3, j % 3].plot(x, hists_gen_list[i][:, j])
            plt.savefig(save_path + '_' + label[i] + '.png', dpi=300)
            plt.close()

        # Remove variables
        del x, z, transform, training_data

        return (kld_cart, kld_bond, kld_angle, kld_dih)



