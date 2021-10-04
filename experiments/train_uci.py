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
parser = argparse.ArgumentParser(description='Train flow with varying base '
                                             'distribution on UCI dataset')

parser.add_argument('--config', type=str, default='config.yaml',
                    help='Path config file specifying model '
                         'architecture and training procedure')
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

# Set seed
if 'seed' in config['training'] and config['training']['seed'] is not None:
    torch.manual_seed(config['training']['seed'])

# Create model
model = lf.UCIFlow(config)

# Move model on GPU if available
use_gpu = not args.mode == 'cpu' and torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
model = model.to(device)
if args.precision == 'double':
    model = model.double()
else:
    model = model.float()

# Load data
dataset_loader = lf.data.uci_loader[config['dataset']['name']]
dataset_path = config['dataset']['path']
training_data, test_data = dataset_loader(dataset_path)
if args.precision == 'double':
    training_data = training_data.double()
    test_data = test_data.double()
else:
    training_data = training_data.float()
    test_data = test_data.float()

# Train model
max_iter = config['training']['max_iter']
objective = 'fkld' if not 'objective' in config['training'] else 'rkld'
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
log_p_hist = np.zeros((0, 2))

# Initialize optimizer and its parameters
lr = config['training']['learning_rate']
weight_decay = config['training']['weight_decay']
optimizer_name = 'adam' if not 'optimizer' in config['training'] \
    else config['training']['optimizer']
if 'q0_iter' in config['training'] and config['training']['q0_iter'] is not None:
    q0_pretrain = True
    q0_iter = config['training']['q0_iter']
    optimizer_param = model.q0.parameters()
else:
    q0_pretrain = False
    optimizer_param = model.parameters()
if optimizer_name == 'adam':
    optimizer = torch.optim.Adam(optimizer_param, lr=lr, weight_decay=weight_decay)
elif optimizer_name == 'adamax':
    optimizer = torch.optim.Adamax(optimizer_param, lr=lr, weight_decay=weight_decay)
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
    p_tmp = model.p
    model.p = None
    ema_model = AveragedModel(model, avg_fn=ema_avg)
    model.p = p_tmp
    ema_log_p_hist = np.zeros((0, 2))

# Resume training if needed
start_iter = 0
if args.resume:
    latest_cp = bg.utils.get_latest_checkpoint(cp_dir, 'model')
    if latest_cp is not None:
        model.load(latest_cp)
        start_iter = int(latest_cp[-10:-3])
        optimizer_path = os.path.join(cp_dir, 'optimizer.pt')
        if os.path.exists(optimizer_path):
            if q0_pretrain and start_iter >= q0_iter:
                if optimizer_name == 'adam':
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                elif optimizer_name == 'adamax':
                    optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)
                else:
                    raise NotImplementedError('The optimizer ' + optimizer_name + ' is not implemented.')
            optimizer.load_state_dict(torch.load(optimizer_path))
        warmup_scheduler_path = os.path.join(cp_dir, 'warmup_scheduler.pt')
        if os.path.exists(warmup_scheduler_path):
            warmup_scheduler.load_state_dict(torch.load(warmup_scheduler_path))
        # Load logs
        log_labels = ['loss', 'log_p_test']
        log_hists = [loss_hist, log_p_hist]
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
        # Load ema model and logs if needed
        if ema:
            ema_path = os.path.join(cp_dir, 'ema_model_%07i.pt' % start_iter)
            if os.path.exists(ema_path):
                ema_model.load_state_dict(torch.load(ema_path))
            log_labels = ['ema_log_p_test']
            log_hists = [ema_log_p_hist]
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
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          shuffle=False, pin_memory=True,
                                          drop_last=False, num_workers=4)

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

    loss = model.forward_kld(x)

    # Make step
    if not torch.isnan(loss) and not torch.isinf(loss):
        loss.backward()
        optimizer.step()

    # Update Lipschitz constant if flows are residual
    if model.flow_type == 'residual':
        nf.utils.update_lipschitz(model, 5)

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

    # Delete variables to prevent out of memory errors
    del loss
    del x

    # End q0 pretraining
    if q0_pretrain and it == q0_iter:
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamax':
            optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise NotImplementedError('The optimizer ' + optimizer_name + ' is not implemented.')

    # Save loss
    if (it + 1) % log_iter == 0:
        np.savetxt(os.path.join(log_dir, 'loss.csv'), loss_hist,
                   delimiter=',', header='it,loss', comments='')
        if use_gpu:
            torch.cuda.empty_cache()

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
        log_p_sum = 0
        for x in iter(test_loader):
            log_p = model.log_prob(x)
            log_p_sum += torch.sum(log_p.detach())
        model.train()
        log_p_avg = log_p_sum / len(test_data)

        # Save log_p
        log_p_hist = np.concatenate([log_p_hist, np.array([[it + 1, log_p_avg.item()]])])
        np.savetxt(os.path.join(log_dir, 'log_p_test.csv'), log_p_hist,
                   delimiter=',', header='it,log_p', comments='')

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

            log_p_sum = 0
            for x in iter(test_loader):
                log_p = model.log_prob(x)
                log_p_sum += torch.sum(log_p.detach())
            log_p_avg = log_p_sum / len(test_data)

            # Save log_p
            ema_log_p_hist = np.concatenate([ema_log_p_hist, np.array([[it + 1, log_p_avg.item()]])])
            np.savetxt(os.path.join(log_dir, 'ema_log_p_test.csv'), ema_log_p_hist,
                       delimiter=',', header='it,log_p', comments='')

            # Save model
            torch.save(ema_model.state_dict(),
                       os.path.join(cp_dir, 'ema_model_%07i.pt' % (it + 1)))

            # Reset model to train mode
            ema_model.train()

    # End job if necessary
    if it % checkpoint_iter == 0 and args.tlimit is not None:
        time_past = (time() - start_time) / 3600
        num_cp = (it + 1 - start_iter) / checkpoint_iter
        if num_cp > .5 and time_past * (1 + 1 / num_cp) > args.tlimit:
            break
