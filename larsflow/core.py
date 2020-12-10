import torch
import numpy as np
import normflow as nf

from . import distributions
from . import nets

# Try importing Boltzmann generator dependencies
try:
    import boltzgen as bg

    from simtk import openmm as mm
    from simtk import unit
    from simtk.openmm import app
    from openmmtools import testsystems
    import mdtraj
except:
    print('Warning: Dependencies for Boltzmann generators could '
          'not be loaded. Other models can still be used.')


class NormalizingFlow(nf.NormalizingFlow):
    """
    Normalizing Flow model to approximate target distribution
    """
    def __init__(self, q0, flows, p=None):
        """
        Constructor
        :param q0: Base distribution
        :param flows: List of flows
        :param p: Target distribution
        """
        super().__init__(q0, flows, p)

    def reverse_kld_cov(self, num_samples=1, beta=1.):
        """
        Estimates reverse KL divergence, gradients through covariance
        :param num_samples: Number of samples to draw from base distribution
        :return: Estimate of the reverse KL divergence averaged over latent samples
        """
        z0, log_q = self.q0(num_samples)
        log_det = torch.tensor(0.)
        z = z0.detach()
        for flow in self.flows:
            z, log_det_part = flow(z)
            log_det = log_det + log_det_part
        log_p = self.p.log_prob(z)
        log_p_ = log_p.detach()
        log_det_ = log_det.detach()
        log_q_ = log_q.detach()
        A = log_q_ - log_det_ - log_p_
        A_ = A - torch.mean(A)
        grad_a = torch.sum(A_ * log_q) / (num_samples - 1)
        grad_f = -(torch.mean(log_p) + torch.mean(log_det))
        rkld = torch.mean(log_q_) - torch.mean(log_p_) - torch.mean(log_det_)
        return rkld + grad_a - grad_a.detach() + beta * (grad_f - grad_f.detach())


class Glow(nf.MultiscaleFlow):
    """
    Glow model with multiscale architecture, see arXiv:1807.03039
    """
    def __init__(self, config):
        """
        Constructor
        :param config: Config dictionary, to be created from yaml file
        """
        # Get parameters
        L = config['levels']
        K = config['blocks']

        input_shape = config['input_shape']
        channels = input_shape[0]
        hidden_channels = config['hidden_channels']
        split_mode = config['split_mode']
        scale = config['scale']
        use_lu = True if not 'use_lu' in config else config['use_lu']
        class_cond = config['class_cond']
        if class_cond:
            num_classes = config['num_classes']
        else:
            num_classes = None

        # Get transform
        if 'transform' in config:
            if config['transform']['type'] == 'logit':
                transform = nf.transforms.Logit(alpha=config['transform']['param'])
            elif config['transform']['type'] == 'shift':
                transform = nf.transforms.Shift()
            else:
                raise NotImplementedError('The transform ' + config['transform']['type']
                                          + ' is not yet implemented')
        else:
            transform = None

        # Set up flows, distributions and merge operations
        q0 = []
        merges = []
        flows = []
        net_actnorm = False if not 'net_actnorm' in config else config['net_actnorm']
        for i in range(L):
            flows_ = []
            for j in range(K):
                flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1 - i), hidden_channels,
                                              split_mode=split_mode, scale=scale,
                                              use_lu=use_lu, net_actnorm=net_actnorm)]
            flows_ += [nf.flows.Squeeze()]
            flows += [flows_]
            if i > 0:
                merges += [nf.flows.Merge()]
                latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i),
                                input_shape[2] // 2 ** (L - i))
            else:
                latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L,
                                input_shape[2] // 2 ** L)
            if config['base']['type'] == 'gauss_channel':
                affine_shape = latent_shape[:1] + ((1,) * (len(latent_shape) - 1))
                q0 += [nf.distributions.AffineGaussian(latent_shape, affine_shape,
                                                       num_classes)]
            elif config['base']['type'] == 'glow_base':
                q0 += [nf.distributions.GlowBase(latent_shape, num_classes)]
            elif config['base']['type'] == 'gauss':
                q0 += [nf.distributions.AffineGaussian(latent_shape, latent_shape,
                                                       num_classes)]
            elif config['base']['type'] == 'resampled_channel':
                affine_shape = latent_shape[:1] + ((1,) * (len(latent_shape) - 1))
                layers = latent_shape[:1]
                layers += (config['base']['params']['a_hidden_units'],) \
                          * config['base']['params']['a_hidden_layers']
                same_dist = config['base']['params']['same_dist']
                num_output = 1
                if not same_dist:
                    num_output *= np.prod(latent_shape[1:])
                if class_cond:
                    num_output *= num_classes
                layers += (num_output,)
                init_zeros = True if not 'init_zeros' in config['base']['params'] \
                    else config['base']['params']['init_zeros']
                a = nf.nets.MLP(layers, output_fn='sigmoid', init_zeros=init_zeros)
                T = config['base']['params']['T']
                eps = config['base']['params']['eps']
                Z_samples = None if not 'Z_samples' in config['base']['params'] \
                    else config['base']['params']['Z_samples']
                q0 += [distributions.FactorizedResampledGaussian(latent_shape, a, T, eps,
                            affine_shape, same_dist=same_dist, num_classes=num_classes,
                            Z_samples=Z_samples)]
            elif config['base']['type'] == 'resampled_hw':
                affine_shape = latent_shape[:1] + ((1,) * (len(latent_shape) - 1))
                ds_h = latent_shape[1] if not 'downsampled_h' in config['base']['params'] \
                    else min(latent_shape[1], config['base']['params']['downsampled_h'])
                levels = int(np.round(np.log2(latent_shape[1] // ds_h)))
                a_channels_factor = config['base']['params']['a_channels']
                a_layers = config['base']['params']['a_layers']
                a_channels = [1] + a_layers * [a_channels_factor]
                a_stride = a_layers * [1]
                for l in range(levels):
                    l_ind = int((l + 1) / (levels + 1) * a_layers)
                    a_stride[l_ind] = 2
                    for c_ind in range(l_ind + 1, a_layers + 1):
                        a_channels[c_ind] = 2 * a_channels[c_ind]
                same_dist = config['base']['params']['same_dist']
                if same_dist:
                    num_output = 1
                else:
                    num_output = latent_shape[0]
                a_output_units = [ds_h ** 2 * a_channels[-1], num_output]
                init_zeros = True if not 'init_zeros' in config['base']['params'] \
                    else config['base']['params']['init_zeros']
                a = nets.ConvNet2d(a_channels, a_output_units, stride=a_stride,
                                   output_fn='sigmoid', init_zeros=init_zeros)
                T = config['base']['params']['T']
                eps = config['base']['params']['eps']
                Z_samples = None if not 'Z_samples' in config['base']['params'] \
                    else config['base']['params']['Z_samples']
                q0 += [distributions.FactorizedResampledGaussian(latent_shape, a, T, eps,
                                affine_shape, group_dim=[1, 2], same_dist=same_dist,
                                num_classes=num_classes, Z_samples=Z_samples)]
            elif config['base']['type'] == 'resampled':
                affine_shape = latent_shape[:1] + ((1,) * (len(latent_shape) - 1))
                ds_h = latent_shape[1] if not 'downsampled_h' in config['base']['params'] \
                    else min(latent_shape[1], config['base']['params']['downsampled_h'])
                levels = int(np.round(np.log2(latent_shape[1] // ds_h)))
                a_channels_factor = config['base']['params']['a_channels']
                a_layers = config['base']['params']['a_layers']
                a_channels = [latent_shape[0]] + a_layers * [input_shape[1] // latent_shape[1] * a_channels_factor]
                a_stride = a_layers * [1]
                for l in range(levels):
                    l_ind = int((l + 1) / (levels + 1) * a_layers)
                    a_stride[l_ind] = 2
                    for c_ind in range(l_ind + 1, a_layers + 1):
                        a_channels[c_ind] = 2 * a_channels[c_ind]
                same_dist = config['base']['params']['same_dist']
                if same_dist:
                    num_output = 1
                else:
                    num_output = 1
                a_output_units = [ds_h ** 2 * a_channels[-1], num_output]
                init_zeros = True if not 'init_zeros' in config['base']['params'] \
                    else config['base']['params']['init_zeros']
                a = nets.ConvNet2d(a_channels, a_output_units, stride=a_stride,
                                   output_fn='sigmoid', init_zeros=init_zeros)
                T = config['base']['params']['T']
                eps = config['base']['params']['eps']
                Z_samples = None if not 'Z_samples' in config['base']['params'] \
                    else config['base']['params']['Z_samples']
                q0 += [distributions.FactorizedResampledGaussian(latent_shape, a, T, eps,
                                                                 affine_shape, group_dim=[0, 1, 2], same_dist=same_dist,
                                                                 num_classes=num_classes, Z_samples=Z_samples)]
            else:
                raise NotImplementedError('The base distribution ' + config['base']['type']
                                          + ' is not implemented')

        # Construct flow model
        super().__init__(q0, flows, merges, transform, class_cond)

    def forward(self, x, y=None, autocast=False):
        """
        Forward pass for data parallel computation
        :param x: Input batch
        :param y: Labels of input batch
        :param autocast: Flag whether to do autocast inside forward pass
        (necessary to use mixed precision with DataParallel model)
        :return: Negative log-likelihood of batch
        """
        if autocast:
            with torch.cuda.amp.autocast():
                out = -self.log_prob(x, y)
                return out
        else:
            return -self.log_prob(x, y)


class BoltzmannGenerator(NormalizingFlow):
    """
    Boltzmann Generator with architecture inspired by arXiv:2002.06707
    """
    def __init__(self, config):
        """
        Constructor
        :param config: Dict, specified by a yaml file, see sample config file
        """

        self.config = config
        # Set up simulation object
        if config['system']['name'] == 'AlanineDipeptideVacuum':
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
            temperature = config['system']['temperature']

            if config['system']['constraints']:
                system = testsystems.AlanineDipeptideVacuum()
            else:
                system = testsystems.AlanineDipeptideVacuum(constraints=None)
            if config['system']['platform'] == 'CPU':
                sim = app.Simulation(system.topology, system.system,
                                     mm.LangevinIntegrator(temperature * unit.kelvin,
                                                           1. / unit.picosecond,
                                                           1. * unit.femtosecond),
                                          mm.Platform.getPlatformByName('CPU'))
            elif config['system']['platform'] == 'Reference':
                sim = app.Simulation(system.topology, system.system,
                                     mm.LangevinIntegrator(temperature * unit.kelvin,
                                                           1. / unit.picosecond,
                                                           1. * unit.femtosecond),
                                     mm.Platform.getPlatformByName('Reference'))
            else:
                sim = app.Simulation(system.topology, system.system,
                                     mm.LangevinIntegrator(temperature * unit.kelvin,
                                                           1. / unit.picosecond,
                                                           1. * unit.femtosecond),
                                     mm.Platform.getPlatformByName(config['system']['platform']),
                                     {'Precision': config['system']['precision']})
        else:
            raise NotImplementedError('The system ' + config['system']['name']
                                      + ' has not been implemented.')

        # Load data for transform
        # Load the alanine dipeptide trajectory
        traj = mdtraj.load(config['data_path']['transform'])
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

        # Set up model
        # Define flows
        blocks = config['model']['blocks']

        # Set target distribution
        energy_cut = config['system']['energy_cut']
        energy_max = config['system']['energy_max']
        add_transform = config['model']['transform']
        transform = bg.flows.CoordinateTransform(training_data, ndim,
                                                 z_matrix, cart_indices)

        if 'parallel_energy' in config['system'] and config['system']['parallel_energy']:
            if add_transform:
                p = bg.distributions.BoltzmannParallel(system, temperature,
                        energy_cut=energy_cut, energy_max=energy_max,
                        n_threads=config['system']['n_threads'])
            else:
                p = bg.distributions.TransformedBoltzmannParallel(system, temperature,
                        energy_cut=energy_cut, energy_max=energy_max, transform=transform,
                        n_threads=config['system']['n_threads'])
        else:
            if add_transform:
                p = bg.distributions.Boltzmann(sim.context, temperature,
                        energy_cut=energy_cut, energy_max=energy_max)
            else:
                p = bg.distributions.TransformedBoltzmann(sim.context, temperature,
                        energy_cut=energy_cut, energy_max=energy_max, transform=transform)

        # Set up parameters for flow layers
        hidden_units = config['model']['hidden_units']
        hidden_layers = config['model']['hidden_layers']
        scale = True if config['model']['coupling'] == 'affine' else False
        scale_map = 'exp' if not 'scale_map' in config['model'] \
            else config['model']['scale_map']
        init_zeros = config['model']['init_zeros']

        # Set up base distribution
        latent_size = config['model']['latent_size']
        if config['model']['base']['type'] == 'resampled':
            T = config['model']['base']['params']['T']
            eps = config['model']['base']['params']['eps']
            a_hl = config['model']['base']['params']['a_hidden_layers']
            a_hu = config['model']['base']['params']['a_hidden_units']
            init_zeros_a = True if not 'init_zeros' in config['model']['base']['params'] \
                else config['model']['base']['params']['init_zeros']
            a = nf.nets.MLP([latent_size] + a_hl * [a_hu] + [1], output_fn="sigmoid",
                            init_zeros=init_zeros_a)
            q0 = distributions.ResampledGaussian(latent_size, a, T, eps,
                                trainable=config['model']['base']['learn_mean_var'])
        elif config['model']['base']['type'] == 'gauss':
            q0 = nf.distributions.DiagGaussian(latent_size,
                                trainable=config['model']['base']['learn_mean_var'])
        else:
            raise NotImplementedError('The base distribution ' + config['model']['base']['type']
                                      + ' is not implemented')

        # Set up flow layers
        flows = []
        for i in range(blocks):
            # Coupling layer
            param_map = nf.nets.MLP([(latent_size + 1) // 2] + hidden_layers * [hidden_units]
                                    + [(latent_size // 2) * (2 if scale else 1)],
                                    init_zeros=init_zeros)
            flows += [nf.flows.AffineCouplingBlock(param_map, scale=scale,
                                                   scale_map=scale_map)]

            # Permutation
            if config['model']['permutation'] == 'affine':
                flows += [nf.flows.InvertibleAffine(latent_size)]
            else:
                flows += [nf.flows.Permute(latent_size, config['model']['permutation'])]

            # ActNorm
            if config['model']['actnorm']:
                flows += [nf.flows.ActNorm(latent_size)]
        # Coordinate transformation
        if add_transform:
            flows += [transform]

        # Construct flow model
        super().__init__(q0=q0, flows=flows, p=p)