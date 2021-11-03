# Resampling Base Distributions of Normalizing Flows

Normalizing flows are a popular class of models for approximating probability 
distributions. However, their invertible nature limits their ability to model 
target distributions with a complex topological structure, such as Boltzmann 
distributions. Several procedures have been proposed to solve this problem 
but many of them sacrifice invertibility and, thereby, tractability of the 
log-likelihood as well as other desirable properties. To address these limitations, 
we introduce a base distribution for normalizing flows based on learned rejection 
sampling in our article
[Resampling Base Distributions of Normalizing Flows](https://arxiv.org/abs/2110.15828),
allowing the resulting normalizing flow to model complex topologies without giving
up bijectivity. In this repository, we implemented this class of base distributions
and provide the script for various experiments comparing them to other commonly used
base distributions. Some results of applying our method to 2D distributions are shown
below.

![Normalizing flows with different base distributions](https://github.com/VincentStimper/resampled-base-flows/blob/master/images/2d_distributions.png "Normalizing flows with different base distributions")

## Methods of Installation

The latest version of the package can be installed via pip

```
pip install --upgrade git+https://github.com/VincentStimper/resampled-base-flows.git
```

If you want to use a GPU, make sure that PyTorch is set up correctly by
by following the instructions at the
[PyTorch website](https://pytorch.org/get-started/locally/).

To run the Boltzmann generator experiments it is necessary to install OpenMM.
Instructions on how to do this can be found 
[here](http://docs.openmm.org/7.0.0/userguide/application.html#installing-openmm).
