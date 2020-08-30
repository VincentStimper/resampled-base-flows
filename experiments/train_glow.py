# Import required packages
import torch
import torchvision as tv
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
parser.add_argument('--mode', type=str, default='mgpu',
                    help='Compute mode, can be cpu, gpu, or mgpu for multiple gpu')
parser.add_argument('--precision', type=str, default='float',
                    help='Precision to be used for computation, can be float, double or mixed')
parser.add_argument('--tlimit', type=float, default=None,
                    help='Number of hours after which to stop training')

args = parser.parse_args()


# Load config
config = utils.get_config(args.config)


# Get computing device
use_gpu = not args.mode == 'cpu' and torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')


# Set seed if needed
if 'seed' in config['training'] and config['training']['seed'] is not None:
    torch.manual_seed(config['training']['seed'])


# Prepare training data
batch_size = config['training']['batch_size']
class_cond = config['model']['class_cond']

# Load dataset
if config['dataset']['name'] == 'cifar10':
    config['model']['input_shape'] = (3, 32, 32)
    if class_cond:
        num_classes = 10
        config['model']['num_classes'] = 10

    if config['dataset']['transform']['type'] == 'logit':
        alpha = config['dataset']['transform']['param']
        logit = nf.utils.Logit(alpha=alpha)
        test_trans = [tv.transforms.ToTensor(), nf.utils.Jitter(),
                      logit, nf.utils.ToDevice(device, args.precision)]
        train_trans = [tv.transforms.RandomHorizontalFlip()] + test_trans
        # Set parameters for bits per dim evaluation
        bpd_trans = 'logit'
        bpd_param = [alpha]
    else:
        raise NotImplementedError('The transform ' + config['dataset']['transform']['type']
                                  + 'is not implemented for ' + config['dataset']['name'])
    train_data = tv.datasets.CIFAR10(config['dataset']['path'], train=True, download=True,
                                     transform=tv.transforms.Compose(train_trans))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=True)

    test_data = tv.datasets.CIFAR10(config['dataset']['path'], train=False, download=True,
                                    transform=tv.transforms.Compose(test_trans))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=True)
else:
    raise NotImplementedError('The dataset ' + config['dataset']['name']
                              + 'is not implemented.')

train_iter = iter(train_loader)
test_iter = iter(test_loader)


# Create model
model = Glow(config['model'])

# Move model on GPU if available
model = model.to(device)
if args.precision == 'double':
    model = model.double()

if use_gpu and args.mode == 'mgpu' and torch.cuda.device_count() > 1:
    # Initialize ActNorm Layers
    with torch.no_grad():
        nsplit = batch_size // torch.cuda.device_count()
        if nsplit == 0:
            nsplit = 1
        x, y = next(train_iter)
        _ = model(x[:nsplit, ...], y[:nsplit, ...].to(device) if class_cond else None)
        del(x, y)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    data_parallel = True
else:
    data_parallel = False


# Prepare folders for results
root = config['training']['save_root']
cp_dir = os.path.join(root, 'checkpoints')
sam_dir = os.path.join(root, 'samples')
log_dir = os.path.join(root, 'log')
# Create dirs if not existent
for dir in [cp_dir, sam_dir, log_dir]:
    if not os.path.isdir(dir):
        os.mkdir(dir)


# Prepare training utilities
max_iter = config['training']['max_iter']
cp_iter = config['training']['cp_iter']
log_iter = config['training']['log_iter']
num_samples = config['training']['num_samples']
sample_temperature = [1.0]
if 'sample_temperature' in config['training']:
    if hasattr(config['training']['sample_temperature'], '__iter__'):
        sample_temperature += list(config['training']['sample_temperature'])
    else:
        sample_temperature += [config['training']['sample_temperature']]

loss_hist = np.zeros((0, 2))
bpd_hist = np.zeros((0, 5))

optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'],
                             weight_decay=config['training']['weight_decay'])
if args.precision == 'mixed':
    scaler = torch.cuda.amp.GradScaler()


# Resume training if needed
start_iter = 0
if args.resume:
    latest_cp = utils.get_latest_checkpoint(cp_dir, 'model')
    if latest_cp is not None:
        if data_parallel:
            model.module.load(latest_cp)
        else:
            model.load(latest_cp)
        start_iter = int(latest_cp[-10:-3])
        optimizer_path = os.path.join(cp_dir, 'optimizer.pt')
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path))
        if args.precision == 'mixed':
            scaler_path = os.path.join(cp_dir, 'scaler.pt')
            if os.path.exists(scaler_path):
                scaler.load_state_dict(torch.load(scaler_path))
        loss_path = os.path.join(log_dir, 'loss.csv')
        if os.path.exists(loss_path):
            loss_hist = np.loadtxt(loss_path, delimiter=',', skiprows=1)
            loss_hist = loss_hist[loss_hist[:, 0] <= start_iter, :]
        bpd_path = os.path.join(log_dir, 'bits_per_dim.csv')
        if os.path.exists(bpd_path):
            bpd_hist = np.loadtxt(bpd_path, delimiter=',', skiprows=1)
            bpd_hist = bpd_hist[bpd_hist[:, 0] <= start_iter, :]


# Train model
for it in range(start_iter, max_iter):
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)
    optimizer.zero_grad()
    if args.precision == 'mixed':
        with torch.cuda.amp.autocast():
            nll = model(x, y.to(device) if class_cond else None,
                        autocast=True if data_parallel else False)
            loss = torch.mean(nll)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        nll = model(x, y.to(device) if class_cond else None)
        loss = torch.mean(nll)
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()

    loss_append = np.array([[it + 1, loss.detach().to('cpu').numpy()]])
    loss_hist = np.concatenate([loss_hist, loss_append])
    del (x, y, loss, nll)

    if (it + 1) % log_iter == 0:
        with torch.no_grad():
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            b = utils.bitsPerDim(model, x, y.to(device) if class_cond else None,
                                 trans=bpd_trans, trans_param=bpd_param)
            bpd_train = b.to('cpu').numpy()
            try:
                x, y = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                x, y = next(test_iter)
            b = utils.bitsPerDim(model, x, y.to(device) if class_cond else None,
                                 trans=bpd_trans, trans_param=bpd_param)
            bpd_test = b.to('cpu').numpy()
            del(x, y, b)
            if use_gpu:
                torch.cuda.empty_cache()
        bpd_append = np.array([[it + 1, np.nanmean(bpd_train), np.nanstd(bpd_train),
                                np.nanmean(bpd_test), np.nanstd(bpd_test)]])
        bpd_hist = np.concatenate([bpd_hist, bpd_append])
        np.savetxt(os.path.join(log_dir, 'bits_per_dim.csv'), bpd_hist, delimiter=',',
                   header='it,train_mean,train_std,test_mean,test_std', comments='')
        np.savetxt(os.path.join(log_dir, 'loss.csv'), loss_hist, delimiter=',',
                   header='it,loss', comments='')

    if (it + 1) % cp_iter == 0:
        # Save checkpoint
        if data_parallel:
            model.module.save(os.path.join(cp_dir, 'model_%07i.pt' % (it + 1)))
        else:
            model.save(os.path.join(cp_dir, 'model_%07i.pt' % (it + 1)))
        torch.save(optimizer.state_dict(), os.path.join(cp_dir, 'optimizer.pt'))
        if args.precision == 'mixed':
            torch.save(scaler.state_dict(), os.path.join(cp_dir, 'scaler.pt'))

        # Generate samples
        with torch.no_grad():
            for st in sample_temperature:
                if class_cond:
                    y = torch.arange(num_classes).repeat(num_samples).to(device)
                    nrow = num_classes
                else:
                    y = None
                    nrow = 8
                if data_parallel:
                    x, _ = model.module.sample(num_samples, y=y, temperature=st)
                else:
                    x, _ = model.sample(num_samples, y=y, temperature=st)
                if config['dataset']['transform']['type'] == 'logit':
                    x = logit.inverse(x)
                x_ = torch.clamp(x.cpu(), 0, 1)
                img = np.transpose(tv.utils.make_grid(x_, nrow=nrow).numpy(), (1, 2, 0))
                plt.imsave(os.path.join(sam_dir, 'samples_T_%.2f_%07i.png' % (st, it + 1)), img)
                del(x, y, x_)
                if use_gpu:
                    torch.cuda.empty_cache()

        if args.tlimit is not None:
            time_past = (time() - start_time) / 3600
            num_cp = (it + 1 - start_iter) / cp_iter
            if time_past * (1 + 1 / num_cp) > args.tlimit:
                break