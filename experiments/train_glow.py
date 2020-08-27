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
parser.add_argument('--cpu', action='store_true',
                    help='Flag whether to use cpu even if gpu is available')
parser.add_argument('--tlimit', type=float, default=None,
                    help='Number of hours after which to stop training')

args = parser.parse_args()


# Load config
config = utils.get_config(args.config)


# Get computing device
device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')


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
                      logit, nf.utils.ToDevice(device)]
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


# Create model
model = Glow(config['model'])

# Move model on GPU if available
model = model.to(device)
model = model.double()


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

loss_hist = np.zeros((0, 2))
bpd_hist = np.zeros((0, 5))

optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'],
                             weight_decay=config['training']['weight_decay'])

train_iter = iter(train_loader)
test_iter = iter(test_loader)


# Resume training if needed
start_iter = 0
if args.resume:
    latest_cp = utils.get_latest_checkpoint(os.path.join(cp_dir, 'checkpoints'),
                                            'model')
    if latest_cp is not None:
        model.load(latest_cp)
        optimizer_path = os.path.join(cp_dir, 'optimizer.pt')
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path))
        loss_path = os.path.join(log_dir, 'loss.csv')
        if os.path.exists(loss_path):
            loss_hist = np.loadtxt(loss_path, delimiter=',', skiprows=1)
        bpd_path = os.path.join(log_dir, 'bits_per_dim.csv')
        if os.path.exists(loss_path):
            bpd_hist = np.loadtxt(loss_path, delimiter=',', skiprows=1)
        start_iter = int(latest_cp[-10:-3])


# Train model
for it in range(max_iter):
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)
    optimizer.zero_grad()
    loss = model.forward_kld(x, y.to(device) if class_cond else None)

    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()

    loss_append = np.array([[it + 1, loss.detach().to('cpu').numpy()]])
    loss_hist = np.concatenate([loss_hist, loss_append])
    del (x, y, loss)

    if (it + 1) % log_iter == 0:
        with torch.no_grad():
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            b = nf.utils.bitsPerDim(model, x, y.to(device) if class_cond else None,
                                    trans=bpd_trans, trans_param=bpd_param)
            bpd_train = b.to('cpu').numpy()
            try:
                x, y = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                x, y = next(test_iter)
            b = nf.utils.bitsPerDim(model, x, y.to(device) if class_cond else None,
                                    trans=bpd_trans, trans_param=bpd_param)
            bpd_test = b.to('cpu').numpy()
            del(x, y, b)
            torch.cuda.empty_cache()
        bpd_append = np.array([[it + 1, np.nanmean(bpd_train), np.nanstd(bpd_train),
                                np.nanmean(bpd_test), np.nanstd(bpd_test)]])
        bpd_hist = np.concatenate([bpd_hist, bpd_append])
        np.savetxt(os.path.join(cp_dir, 'bits_per_dim.csv'), bpd_hist, delimiter=',',
                   header='it,train_mean,train_std,test_mean,test_std', comments='')
        np.savetxt(os.path.join(cp_dir, 'loss.csv'), loss_hist, delimiter=',',
                   header='it,loss', comments='')

    if (it + 1) % cp_iter == 0:
        # Save checkpoint
        model.save(os.path.join(cp_dir, 'model_%07i.pt' % (it + 1)))
        torch.save(optimizer.state_dict(), os.path.join(cp_dir, 'optimizer.pt'))

        # Generate samples
        with torch.no_grad():
            if class_cond:
                y = torch.arange(num_classes).repeat(num_samples).to(device)
                nrow = num_classes
            else:
                y = None
                nrow = 8
            x, _ = model.sample(num_samples, y=y)
            if config['dataset']['transform']['type'] == 'logit':
                x = logit.inverse(x)
            x_ = torch.clamp(x.cpu(), 0, 1)
            img = np.transpose(tv.utils.make_grid(x_, nrow=nrow).numpy(), (1, 2, 0))
            plt.imsave(os.path.join(sam_dir, 'samples_%07i.png' % (it + 1)), img)

    if args.tlimit is not None and (time() - start_time) / 3600 > args.tlimit:
        break