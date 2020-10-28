# Import modules
import torch
import numpy as np

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
max_iter = config['train']['max_iter']
n_data = len(training_data)
log_iter = config['train']['log_iter']
checkpoint_iter = config['train']['checkpoint_iter']
root = config['training']['save_root']
cp_dir = os.path.join(root, 'checkpoints')
plot_dir = os.path.join(root, 'plots')
log_dir = os.path.join(root, 'log')
# Create dirs if not existent
for dir in [cp_dir, plot_dir, log_dir]:
    if not os.path.isdir(dir):
        os.mkdir(dir)

loss_hist = np.zeros((0, 2))

optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'],
                             weight_decay=config['train']['weight_decay'])
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                      gamma=config['train']['rate_decay'])

# Resume training if needed
start_iter = 0
if args.resume:
    latest_cp = bg.utils.get_latest_checkpoint(cp_dir, 'model')
    if latest_cp is not None:
        model.load(latest_cp)
        optimizer_path = os.path.join(cp_dir, 'optimizer.pt')
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path))
        loss_path = os.path.join(log_dir, 'loss.csv')
        if os.path.exists(loss_path):
            loss_hist = np.loadtxt(loss_path)
            loss_hist = loss_hist[loss_hist[:, 0] <= start_iter, :]
        start_iter = int(latest_cp[-10:-3])
if start_iter > 0:
    for _ in range(start_iter // config['train']['decay_iter']):
        lr_scheduler.step()

# Start training
start_time = time()

batch_size = config['train']['batch_size']

for it in range(start_iter, max_iter):
    # Get batch from dataset
    ind = torch.randint(n_data, (batch_size, ))
    x = training_data[ind, :].to(device)

    # Get loss
    loss = model.forward_kld(x)

    # Make step
    if not torch.isnan(loss) and not torch.isinf(loss):
        loss.backward()
        optimizer.step()

    # Log loss
    loss_append = np.array([[it + 1, loss.item()]])
    loss_hist = np.concatenate([loss_hist, loss_append])

    # Clear gradients
    nf.utils.clear_grad(model)

    # Save loss
    if (it + 1) % log_iter == 0:
        np.savetxt(os.path.join(log_dir, 'loss.csv'), loss_hist)

    # Save checkpoint
    if (it + 1) % checkpoint_iter == 0:
        model.save(os.path.join(cp_dir, 'model_%07i.pt' % (it + 1)))
        torch.save(optimizer.state_dict(),
                   os.path.join(cp_dir, 'optimizer.pt'))
        if args.tlimit is not None and (time() - start_time) / 3600 > args.tlimit:
            break

    # Update lr scheduler
    if (it + 1) % config['train']['decay_iter'] == 0:
        lr_scheduler.step()
    