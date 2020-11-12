# Import modules
import torch
import numpy as np
from torch.optim.swa_utils import AveragedModel

import larsflow as lf
import boltzgen as bg
import normflow as nf

import argparse
import os
from time import time


# Parse input arguments
parser = argparse.ArgumentParser(description='Train Boltzmann Generator with varying '
                                             'base distribution')

parser.add_argument('--config', type=str, default='../config/bm.yaml',
                    help='Path config file specifying model '
                         'architecture and training procedure',)
parser.add_argument("--resume", action="store_true",
                    help='Flag whether to resume training')
parser.add_argument("--tlimit", type=float, default=None,
                    help='Number of hours after which to stop training')
parser.add_argument('--mode', type=str, default='gpu',
                    help='Compute mode, can be cpu, or gpu')
parser.add_argument('--precision', type=str, default='float',
                    help='Precision to be used for computation, '
                         'can be float, double, or mixed')

args = parser.parse_args()


# Load config
config = lf.utils.get_config(args.config)


# Create model
model = lf.BoltzmannGenerator(config)

# Move model on GPU if available
use_gpu = not args.mode == 'cpu' and torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
model = model.to(device)
if args.precision == 'double':
    model = model.double()
else:
    model = model.float()


# Load data
path = config['data_path']['train']
if path[-2:] == 'h5':
    training_data = bg.utils.load_traj(path)
elif path[-2:] == 'pt':
    training_data = torch.load(path)
else:
    raise NotImplementedError('Data format ' + path[-2:] + 'not supported.')
path = config['data_path']['test']
if path[-2:] == 'h5':
    test_data = bg.utils.load_traj(path)
elif path[-2:] == 'pt':
    test_data = torch.load(path)
else:
    raise NotImplementedError('Data format ' + path[-2:] + 'not supported.')
if args.precision == 'double':
    training_data = training_data.double()
    test_data = test_data.double()
else:
    training_data = training_data.float()
    test_data = test_data.float()

# Train model
max_iter = config['training']['max_iter']
n_data = len(training_data)
log_iter = config['training']['log_iter']
checkpoint_iter = config['training']['checkpoint_iter']
root = config['training']['save_root']
cp_dir = os.path.join(root, 'checkpoints')
plot_dir = os.path.join(root, 'plots')
log_dir = os.path.join(root, 'log')
# Create dirs if not existent
for dir in [cp_dir, plot_dir, log_dir]:
    if not os.path.isdir(dir):
        os.mkdir(dir)

# Init logs
loss_hist = np.zeros((0, 2))
kld_hist = np.zeros((0, 3))
kld_cart_hist = np.zeros((0, 12))
kld_bond_hist = np.zeros((0, 20))
kld_angle_hist = np.zeros((0, 20))
kld_dih_hist = np.zeros((0, 20))

# Initialize optimizer and its parameters
lr = config['training']['learning_rate']
weight_decay = config['training']['weight_decay']
optimizer_name = 'adam' if not 'optimizer' in config['training'] \
    else config['training']['optimizer']
if optimizer_name == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
elif optimizer_name == 'adamax':
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)
else:
    raise NotImplementedError('The optimizer ' + optimizer_name + ' is not implemented.')
lr_warmup = 'warmup_iter' in config['training'] \
            and config['training']['warmup_iter'] is not None
if lr_warmup:
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                            lambda s: min(1., s / config['training']['warmup_iter']))
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                      gamma=config['training']['rate_decay'])

# Polyak (EMA) averaging
ema = True if 'ema' in config['training'] and config['training']['ema'] is not None \
    else False
if ema:
    ema_beta = config['training']['ema']
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
        ema_beta * averaged_model_parameter + (1 - ema_beta) * model_parameter
    model.p = None
    ema_model = AveragedModel(model, avg_fn=ema_avg)
    ema_kld_hist = np.zeros((0, 3))
    ema_kld_cart_hist = np.zeros((0, 12))
    ema_kld_bond_hist = np.zeros((0, 20))
    ema_kld_angle_hist = np.zeros((0, 20))
    ema_kld_dih_hist = np.zeros((0, 20))

# Resume training if needed
start_iter = 0
if args.resume:
    latest_cp = bg.utils.get_latest_checkpoint(cp_dir, 'model')
    if latest_cp is not None:
        model.load(latest_cp)
        start_iter = int(latest_cp[-10:-3])
        optimizer_path = os.path.join(cp_dir, 'optimizer.pt')
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path))
        warmup_scheduler_path = os.path.join(cp_dir, 'warmup_scheduler.pt')
        if os.path.exists(warmup_scheduler_path):
            warmup_scheduler.load_state_dict(torch.load(warmup_scheduler_path))
        # Load logs
        log_labels = ['loss', 'kld', 'kld_cart', 'kld_bond', 'kld_angle', 'kld_dih']
        log_hists = [loss_hist, kld_hist, kld_cart_hist, kld_bond_hist, kld_angle_hist,
                     kld_dih_hist]
        for log_label, log_hist in zip(log_labels, log_hists):
            log_path = os.path.join(log_dir, log_label + '.csv')
            if os.path.exists(log_path):
                log_hist_ = np.loadtxt(log_path, delimiter=',', skiprows=1)
                if log_hist_.ndim == 1:
                    log_hist_ = log_hist_[None, :]
                log_hist.resize(*log_hist_.shape, refcheck=False)
                log_hist[:, :] = log_hist_
                log_hist.resize(np.sum(kld_hist[:, 0] <= start_iter), log_hist_.shape[1],
                                refcheck=False)
        # Load ema logs if needed
        if ema:
            log_labels = ['ema_kld', 'ema_kld_cart', 'ema_kld_bond', 'ema_kld_angle',
                          'ema_kld_dih']
            log_hists = [ema_kld_hist, ema_kld_cart_hist, ema_kld_bond_hist,
                         ema_kld_angle_hist, ema_kld_dih_hist]
            for log_label, log_hist in zip(log_labels, log_hists):
                log_path = os.path.join(log_dir, log_label + '.csv')
                if os.path.exists(log_path):
                    log_hist_ = np.loadtxt(log_path, delimiter=',', skiprows=1)
                    if log_hist_.ndim == 1:
                        log_hist_ = log_hist_[None, :]
                    log_hist.resize(*log_hist_.shape, refcheck=False)
                    log_hist[:, :] = log_hist_
                    log_hist.resize(np.sum(log_hist_[:, 0] <= start_iter), log_hist_.shape[1],
                                    refcheck=False)

# Set lr scheduler towards previous state in case of resume
if start_iter > 0:
    for _ in range(start_iter // config['training']['decay_iter']):
        lr_scheduler.step()

# Get data loader
batch_size = config['training']['batch_size']
train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,
                                           shuffle=True, pin_memory=True,
                                           drop_last=True, num_workers=4)
train_iter = iter(train_loader)


# Start training
start_time = time()

for it in range(start_iter, max_iter):
    # Get batch from dataset
    try:
        x = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x = next(train_iter)
    x = x.to(device, non_blocking=True)

    # Get loss
    loss = model.forward_kld(x)

    # Make step
    if not torch.isnan(loss) and not torch.isinf(loss):
        loss.backward()
        optimizer.step()

    # Perform ema
    if ema:
        ema_model.update_parameters(model)

    # Log loss
    loss_append = np.array([[it + 1, loss.item()]])
    loss_hist = np.concatenate([loss_hist, loss_append])

    # Clear gradients
    nf.utils.clear_grad(model)

    # Do lr warmup if needed
    if lr_warmup and it <= config['training']['warmup_iter']:
        warmup_scheduler.step()

    # Update lr scheduler
    if (it + 1) % config['training']['decay_iter'] == 0:
        lr_scheduler.step()

    # Save loss
    if (it + 1) % log_iter == 0:
        np.savetxt(os.path.join(log_dir, 'loss.csv'), loss_hist,
                   delimiter=',', header='it,loss', comments='')

    if (it + 1) % checkpoint_iter == 0:
        # Save checkpoint
        model.save(os.path.join(cp_dir, 'model_%07i.pt' % (it + 1)))
        torch.save(optimizer.state_dict(),
                   os.path.join(cp_dir, 'optimizer.pt'))
        if lr_warmup:
            torch.save(warmup_scheduler.state_dict(),
                       os.path.join(cp_dir, 'warmup_scheduler.pt'))

        # Evaluate model and save plots
        model.eval()
        kld = lf.utils.evaluateAldp(model, test_data,
                                    save_path=os.path.join(plot_dir, 'marginals_%07i' % (it + 1)),
                                    data_path=config['data_path']['transform'])
        model.train()

        # Calculate and save KLD stats
        kld_ = np.concatenate(kld)
        kld_append = np.array([[it + 1, np.median(kld_), np.mean(kld_)]])
        kld_hist = np.concatenate([kld_hist, kld_append])
        np.savetxt(os.path.join(log_dir, 'kld.csv'), kld_hist, delimiter=',',
                   header='it,kld_median,kld_mean', comments='')
        kld_labels = ['cart', 'bond', 'angle', 'dih']
        kld_hists = [kld_cart_hist, kld_bond_hist, kld_angle_hist, kld_dih_hist]
        for kld_label, kld_, kld_hist_ in zip(kld_labels, kld, kld_hists):
            kld_append = np.concatenate([np.array([it + 1, np.median(kld_), np.mean(kld_)]), kld_])
            kld_hist_.resize(kld_hist_.shape[0] + 1, kld_hist_.shape[1], refcheck=False)
            kld_hist_[-1, :] = kld_append
            header = 'it,kld_median,kld_mean'
            for kld_ind in range(len(kld_)):
                header += ',kld' + str(kld_ind)
            np.savetxt(os.path.join(log_dir, 'kld_' + kld_label + '.csv'), kld_hist_,
                       delimiter=',', header=header, comments='')

        # Evaluate ema model
        if ema:
            # Set model to evaluation mode
            ema_model.eval()

            # Estimate Z if needed
            if 'resampled' in config['model']['base']['type']:
                num_s = 2 ** 17 if not 'Z_num_samples' in config['training'] \
                    else config['training']['Z_num_samples']
                num_b = 2 ** 10 if not 'Z_num_batches' in config['training'] \
                    else config['training']['Z_num_batches']
                ema_model.module.q0.estimate_Z(num_s, num_b)

            # Evaluate ema model and save plots
            kld = lf.utils.evaluateAldp(ema_model.module, test_data,
                                        save_path=os.path.join(plot_dir, 'ema_marginals_%07i' % (it + 1)),
                                        data_path=config['data_path']['transform'])

            # Calculate and save KLD stats
            kld_ = np.concatenate(kld)
            kld_append = np.array([[it + 1, np.median(kld_), np.mean(kld_)]])
            ema_kld_hist = np.concatenate([ema_kld_hist, kld_append])
            np.savetxt(os.path.join(log_dir, 'ema_kld.csv'), ema_kld_hist, delimiter=',',
                       header='it,kld_median,kld_mean', comments='')
            kld_labels = ['cart', 'bond', 'angle', 'dih']
            kld_hists = [ema_kld_cart_hist, ema_kld_bond_hist, ema_kld_angle_hist,
                         ema_kld_dih_hist]
            for kld_label, kld_, kld_hist_ in zip(kld_labels, kld, kld_hists):
                kld_append = np.concatenate([np.array([it + 1, np.median(kld_),
                                                       np.mean(kld_)]), kld_])
                kld_hist_.resize(kld_hist_.shape[0] + 1, kld_hist_.shape[1], refcheck=False)
                kld_hist_[-1, :] = kld_append
                header = 'it,kld_median,kld_mean'
                for kld_ind in range(len(kld_)):
                    header += ',kld' + str(kld_ind)
                np.savetxt(os.path.join(log_dir, 'ema_kld_' + kld_label + '.csv'), kld_hist_,
                           delimiter=',', header=header, comments='')

            # Save model
            torch.save(ema_model.state_dict(),
                       os.path.join(cp_dir, 'ema_model_%07i.pt' % (it + 1)))

            # Reset model to train mode
            ema_model.train()

        # End job if necessary
        if args.tlimit is not None and (time() - start_time) / 3600 > args.tlimit:
            break
    