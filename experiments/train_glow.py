# Import required packages
import torch
from torch.utils import data
import torchvision as tv
import torch_optimizer as optim

import numpy as np
import normflow as nf

from matplotlib import pyplot as plt

from larsflow import utils
from larsflow import Glow

import os
import argparse

from time import time


# Log start time
start_time = time()

# Parse input arguments
parser = argparse.ArgumentParser(description='Train Glow model on image dataset.')

parser.add_argument('--config', type=str, default='config/glow.yaml',
                    help='Path config file specifying model architecture and training procedure')
parser.add_argument('--resume', action='store_true', help='Flag whether to resume training')
parser.add_argument('--mode', type=str, default='distributed',
                    help='Compute mode, can be cpu, gpu, or distributed')
parser.add_argument('--precision', type=str, default='float',
                    help='Precision to be used for computation, can be float, double or mixed')
parser.add_argument('--tlimit', type=float, default=None,
                    help='Number of hours after which to stop training')
parser.add_argument('--worldsize', type=int, default=1,
                    help='Number of workers for distributed training')
parser.add_argument('--rank', type=int, default=0,
                    help='Rank within distributed worker group')\


args = parser.parse_args()


# Load config
config = utils.get_config(args.config)


# Get computing device
use_gpu = not args.mode == 'cpu' and torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
if use_gpu:
    torch.backends.cudnn.benchmark = True
if use_gpu and args.worldsize > 1 and args.mode == 'distributed':
    distributed = True
    torch.distributed.init_process_group(backend='nccl', init_method='env://',
                                         world_size=args.worldsize, rank=args.rank)
else:
    distributed = False



# Set seed if needed
seed = None
if 'seed' in config['training'] and config['training']['seed'] is not None:
    seed = config['training']['seed']
elif distributed:
    seed = 0
if seed is not None:
    torch.manual_seed(seed)
    if distributed:
        np.random.seed(seed)
        ns = config['training']['max_iter'] // config['training']['cp_iter']
        reset_seeds = np.random.randint(0, 1000000, ns)



# Prepare training data
# Batch size per processing unit
batch_size = config['training']['batch_size'] // args.worldsize
class_cond = config['model']['class_cond']

# Load dataset
if config['dataset']['name'] == 'cifar10':
    # Model parameter
    input_shape = (3, 32, 32)
    config['model']['input_shape'] = input_shape
    if class_cond:
        num_classes = 10
        config['model']['num_classes'] = 10

    # Transform for data loader
    test_trans = [tv.transforms.ToTensor()]
    if args.precision == 'double':
        test_trans += [utils.ToDouble()]
    train_trans = [tv.transforms.RandomHorizontalFlip()] + test_trans
    # Init data loader
    train_data = tv.datasets.CIFAR10(config['dataset']['path'], train=True, download=True,
                                     transform=tv.transforms.Compose(train_trans))
    test_data = tv.datasets.CIFAR10(config['dataset']['path'], train=False, download=True,
                                    transform=tv.transforms.Compose(test_trans))
    if distributed:
        # Need to define sampler to ensure every worker gets a different batch
        train_sampler = data.distributed.DistributedSampler(train_data,
                                                            num_replicas=args.worldsize,
                                                            rank=args.rank)
        train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size,
                                       shuffle=False, num_workers=4, pin_memory=True,
                                       sampler=train_sampler)
    else:
        train_loader = data.DataLoader(train_data, batch_size=batch_size,
                                       shuffle=True, num_workers=4, pin_memory=True)

    test_loader = data.DataLoader(test_data, batch_size=batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
else:
    raise NotImplementedError('The dataset ' + config['dataset']['name']
                              + 'is not implemented.')
# Get number of dims as they are needed to compute bits per dim later
n_dims = np.prod(input_shape)

train_iter = iter(train_loader)


# Create model
model = Glow(config['model'])

if args.precision == 'double':
    model = model.double()


# Prepare folders for results
root = config['training']['save_root']
cp_dir = os.path.join(root, 'checkpoints')
sam_dir = os.path.join(root, 'samples')
log_dir = os.path.join(root, 'log')
# Create dirs if not existent
if args.rank == 0:
    for dir in [cp_dir, sam_dir, log_dir]:
        if not os.path.isdir(dir):
            os.mkdir(dir)

# Resume training if needed, otherwise initialize ActNorm layers
start_iter = 0
latest_cp = utils.get_latest_checkpoint(cp_dir, 'model')
if args.resume and latest_cp is not None:
    model.load(latest_cp)
    start_iter = int(latest_cp[-10:-3])
else:
    # Initialize ActNorm Layers
    init_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        x, y = next(iter(init_loader))
        _ = model(x, y if class_cond else None)
        del (x, y, init_loader)


# Move model on GPU if available
model = model.to(device)


# Prepare training utilities
max_iter = config['training']['max_iter']
cp_iter = config['training']['cp_iter']
log_iter = config['training']['log_iter']
num_samples = config['training']['num_samples']
sample_temperature = [None]
if 'sample_temperature' in config['training']:
    if hasattr(config['training']['sample_temperature'], '__iter__'):
        sample_temperature += list(config['training']['sample_temperature'])
    else:
        sample_temperature += [config['training']['sample_temperature']]

loss_hist = np.zeros((0, 2))
bpd_hist = np.zeros((0, 4))

# Initialize optimizer
# Get parameters
lr = config['training']['lr']
weight_decay = config['training']['weight_decay']
if 'q0_weight_decay' in config['training'] and\
        config['training']['q0_weight_decay'] is not None:
    q0_weight_decay = config['training']['q0_weight_decay']
else:
    q0_weight_decay = weight_decay
#params = [{'params': model.q0.parameters(), 'weight_decay': q0_weight_decay},
#          {'params': model.flows.parameters()}]
momentum = 0.9 if not 'momentum' in config['training'] else config['training']['momentum']
beta = 0.999 if not 'beta' in config['training'] else config['training']['beta']
amsgrad = False if not 'amsgrad' in config['training'] else config['training']['amsgrad']
params = model.parameters()
# Get optimizer
optimizer_name = 'adam' if not 'optimizer' in config['training'] \
    else config['training']['optimizer']
if optimizer_name == 'adam':
    optimizer = torch.optim.Adam(params, lr=lr, betas=(momentum, beta),
                                 weight_decay=weight_decay, amsgrad=amsgrad)
elif optimizer_name == 'adamax':
    optimizer = torch.optim.Adamax(params, lr=lr, betas=(momentum, beta),
                                   weight_decay=weight_decay)
elif optimizer_name == 'adabound':
    if 'gamma' in config['training'] and \
            config['training']['gamma'] is not None:
        gamma = config['training']['gamma']
    else:
        gamma = 1e-3
    optimizer = optim.AdaBound(params, lr=lr, betas=(momentum, beta), gamma=gamma,
                               weight_decay=weight_decay, amsgrad=amsgrad)
elif optimizer_name == 'lamb':
    optimizer = optim.Lamb(params, lr=lr, betas=(momentum, beta),
                           weight_decay=weight_decay)
elif optimizer_name == 'sgd':
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
if args.precision == 'mixed':
    scaler = torch.cuda.amp.GradScaler()
lr_warmup = 'warmup_iter' in config['training'] \
            and config['training']['warmup_iter'] is not None
if lr_warmup:
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                            lambda s: min(1., s / config['training']['warmup_iter']))
lr_decay = 'lr_decay' in config['training'] and config['training']['lr_decay'] is not None
if lr_decay:
    lr_decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                                config['training']['lr_decay'])

# Polyak (EMA) averaging
ema = True if 'ema' in config['training'] and config['training']['ema'] is not None \
    else False
if args.rank == 0 and ema:
    ema_beta = config['training']['ema']
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
        ema_beta * averaged_model_parameter + (1 - ema_beta) * model_parameter
    ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
    ema_bpd_hist = np.zeros((0, 4))



# Load optimizer, etc. if needed
if args.resume:
    optimizer_path = os.path.join(cp_dir, 'optimizer.pt')
    if os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path))
    if args.precision == 'mixed':
        scaler_path = os.path.join(cp_dir, 'scaler.pt')
        if os.path.exists(scaler_path):
            scaler.load_state_dict(torch.load(scaler_path))
    if lr_warmup:
        warmup_scheduler_path = os.path.join(cp_dir, 'warmup_scheduler.pt')
        if os.path.exists(warmup_scheduler_path):
            warmup_scheduler.load_state_dict(torch.load(warmup_scheduler_path))
    if lr_decay and start_iter > 0:
        for _ in range(start_iter // config['training']['lr_decay_iter']):
            lr_decay_scheduler.step()
    if args.rank == 0:
        loss_path = os.path.join(log_dir, 'loss.csv')
        if os.path.exists(loss_path):
            loss_hist = np.loadtxt(loss_path, delimiter=',', skiprows=1)
            loss_hist = loss_hist[:, :2]
            loss_hist = loss_hist[loss_hist[:, 0] <= start_iter, :]
        bpd_path = os.path.join(log_dir, 'bits_per_dim.csv')
        if os.path.exists(bpd_path):
            bpd_hist = np.loadtxt(bpd_path, delimiter=',', skiprows=1)
            if bpd_hist.ndim == 1:
                bpd_hist = bpd_hist[None, :]
            bpd_hist = bpd_hist[bpd_hist[:, 0] <= start_iter, :]
        if ema:
            ema_model_path = os.path.join(cp_dir, 'ema_model_' + str(start_iter) + '.pt')
            if os.path.exists(ema_model_path):
                ema_model.load_state_dict(torch.load(ema_model_path))
            ema_bpd_path = os.path.join(log_dir, 'ema_bits_per_dim.csv')
            if os.path.exists(bpd_path):
                ema_bpd_hist = np.loadtxt(bpd_path, delimiter=',', skiprows=1)
                if ema_bpd_hist.ndim == 1:
                    ema_bpd_hist = bpd_hist[None, :]
                ema_bpd_hist = ema_bpd_hist[ema_bpd_hist[:, 0] <= start_iter, :]

# Make model a distributed one if needed
if distributed:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0])


# Train model
for it in range(start_iter, max_iter):
    # Training step
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)
    # Move data to device
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True) if class_cond else None
    if args.precision == 'mixed':
        with torch.cuda.amp.autocast():
            nll = model(x, y, autocast=True if distributed else False)
            loss = torch.mean(nll)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        nll = model(x, y)
        loss = torch.mean(nll)
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()

    if args.rank == 0:
        # Log loss
        loss_append = np.array([[it + 1, loss.item()]])
        loss_hist = np.concatenate([loss_hist, loss_append])

        # Perform ema
        if ema:
            ema_model.update_parameter(model)

    # Clear gradients
    nf.utils.clear_grad(model)

    # Do lr warmup if needed
    if lr_warmup:
        warmup_scheduler.step()

    # Do lr decay if needed
    if lr_decay and (it + 1) % config['training']['lr_decay_iter'] == 0:
        lr_decay_scheduler.step()

    # Delete variables to prevent out of memory errors
    del x, y, nll, loss

    # Evaluation
    if args.rank == 0 and (it + 1) % log_iter == 0:
        bpd_train = loss_hist[:, 1:] / np.log(2) / n_dims + 8
        np.savetxt(os.path.join(log_dir, 'loss.csv'),
                   np.concatenate([loss_hist, bpd_train], 1),
                   delimiter=',', header='it,loss,bpd', comments='')

    # Checkpoint, i.e. save model, generate samples, and get bits per dim on test set
    if (it + 1) % cp_iter == 0:
        if args.rank == 0:
            # Save checkpoint
            if distributed:
                model.module.save(os.path.join(cp_dir, 'model_%07i.pt' % (it + 1)))
            else:
                model.save(os.path.join(cp_dir, 'model_%07i.pt' % (it + 1)))
            torch.save(optimizer.state_dict(), os.path.join(cp_dir, 'optimizer.pt'))
            if args.precision == 'mixed':
                torch.save(scaler.state_dict(), os.path.join(cp_dir, 'scaler.pt'))
            if lr_warmup:
                torch.save(warmup_scheduler.state_dict(), os.path.join(cp_dir, 'warmup_scheduler.pt'))

            # Set model to evaluation mode
            model.eval()
            with torch.no_grad():
                # Generate samples
                for st in sample_temperature:
                    if class_cond:
                        y = torch.arange(num_classes).repeat(num_samples).to(device)
                        nrow = num_classes
                    else:
                        y = None
                        nrow = 8
                    if distributed:
                        x, _ = model.module.sample(num_samples, y=y, temperature=st)
                    else:
                        x, _ = model.sample(num_samples, y=y, temperature=st)
                    x_ = torch.clamp(x.cpu(), 0, 1)
                    img = np.transpose(tv.utils.make_grid(x_, nrow=nrow).numpy(), (1, 2, 0))
                    plt.imsave(os.path.join(sam_dir, 'samples_T_%.2f_%07i.png'
                                            % (1. if st is None else st, it + 1)), img)

                # Get bits per dim on test set
                bpd_test = np.array([])
                for x, y in iter(test_loader):
                    # Move data to device
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True) if class_cond else None
                    nll = model(x, y)
                    bpd_test = np.concatenate([bpd_test,
                                               nll.cpu().numpy() / np.log(2) / n_dims + 8])
                n_not_nan = np.sum(np.logical_not(np.isnan(bpd_test)))
                bpd_append = np.array([[it + 1, np.nanmean(bpd_test), np.nanstd(bpd_test),
                                        np.nanstd(bpd_test) / np.sqrt(n_not_nan)]])
                bpd_hist = np.concatenate([bpd_hist, bpd_append])
                np.savetxt(os.path.join(log_dir, 'bits_per_dim.csv'), bpd_hist, delimiter=',',
                           header='it,test_mean,test_std,test_err_mean', comments='')

                # Clean up to make sure enough GPU memory is available for training
                del x, y, x_, nll
                if use_gpu:
                    torch.cuda.empty_cache()

            # Reset model to train mode
            model.train()

            # Evaluate ema model
            if ema:
                # Set model to evaluation mode
                ema_model.eval()
                with torch.no_grad():
                    # Generate samples
                    for st in sample_temperature:
                        if class_cond:
                            y = torch.arange(num_classes).repeat(num_samples).to(device)
                            nrow = num_classes
                        else:
                            y = None
                            nrow = 8
                        x, _ = ema_model.sample(num_samples, y=y, temperature=st)
                        x_ = torch.clamp(x.cpu(), 0, 1)
                        img = np.transpose(tv.utils.make_grid(x_, nrow=nrow).numpy(), (1, 2, 0))
                        plt.imsave(os.path.join(sam_dir, 'ema_samples_T_%.2f_%07i.png'
                                                % (1. if st is None else st, it + 1)), img)

                    # Get bits per dim on test set
                    bpd_test = np.array([])
                    for x, y in iter(test_loader):
                        # Move data to device
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True) if class_cond else None
                        nll = ema_model(x, y)
                        bpd_test = np.concatenate([bpd_test,
                                                   nll.cpu().numpy() / np.log(2) / n_dims + 8])
                    n_not_nan = np.sum(np.logical_not(np.isnan(bpd_test)))
                    bpd_append = np.array([[it + 1, np.nanmean(bpd_test), np.nanstd(bpd_test),
                                            np.nanstd(bpd_test) / np.sqrt(n_not_nan)]])
                    ema_bpd_hist = np.concatenate([ema_bpd_hist, bpd_append])
                    np.savetxt(os.path.join(log_dir, 'ema_bits_per_dim.csv'), ema_bpd_hist,
                               delimiter=',', header='it,test_mean,test_std,test_err_mean',
                               comments='')

                    # Clean up to make sure enough GPU memory is available for training
                    del x, y, x_, nll
                    if use_gpu:
                        torch.cuda.empty_cache()

                # Reset model to train mode
                ema_model.train()

        # Reset seeds in distributed setting to ensure samples drawn in model during
        # training are the same
        if distributed:
            torch.manual_seed(reset_seeds[it // cp_iter])

    # Check whether time limit will be hit
    if it % cp_iter == 0:
        if args.tlimit is not None:
            time_past = (time() - start_time) / 3600
            num_cp = (it + 1 - start_iter) / cp_iter
            if num_cp > .5 and time_past * (1 + 1 / num_cp) > args.tlimit:
                break
