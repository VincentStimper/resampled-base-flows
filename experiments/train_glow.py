# Import required packages
import torch
import torchvision as tv
import numpy as np
import normflow as nf

import argparse


# Parse input arguments
parser = argparse.ArgumentParser(description='Train Glow model on image dataset.')

parser.add_argument('--config', type=str, default='config/glow.yaml',
                    help='Path config file specifying model architecture and training procedure')
parser.add_argument('--resume', action='store_true', help='Flag whether to resume training')
#parser.add_argument('--tlimit', type=float, default=None,
#                    help='Number of hours after which to stop training')

args = parser.parse_args()

# Load config
config = bg.utils.get_config(args.config)


# Create model
model = bg.BoltzmannGenerator(config)

# Move model on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = model.double()