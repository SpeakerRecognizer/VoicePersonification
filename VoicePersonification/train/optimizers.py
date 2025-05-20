# The script is borrowed from the following repository: https://github.com/clovaai/voxceleb_trainer
# The script creates different optimizers


# Import of modules
import torch


def AdamOptimizer(parameters, lr, weight_decay, **kwargs):
    # Function to create Adam optimizer

    print('Initialised Adam optimizer.')

    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay);

def SGDOptimizer(parameters, lr, weight_decay, **kwargs):
    # Function to create SGD optimizer

    print('Initialised SGD optimizer.')

    return torch.optim.SGD(parameters, lr = lr, momentum = 0.9, weight_decay=weight_decay)