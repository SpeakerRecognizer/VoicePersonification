# The script is borrowed from the following repository: https://github.com/clovaai/voxceleb_trainer
# The script creates different schedulers


# Import of modules
import torch


def StepLRScheduler(optimizer, test_interval, lr_decay, **kwargs):
    # Function to create scheduler
    
    sche_fn = torch.optim.lr_scheduler.StepLR(optimizer, step_size=test_interval, gamma=lr_decay)

    lr_step = 'epoch'

    print('Initialised step LR scheduler.')

    return sche_fn, lr_step

def OneCycleLRScheduler(optimizer, pct_start, cycle_momentum, max_lr, div_factor, final_div_factor, total_steps, **kwargs):
    # Function to create scheduler
    
    sche_fn = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                  pct_start=pct_start, 
                                                  cycle_momentum=cycle_momentum, 
                                                  max_lr=max_lr, 
                                                  div_factor=div_factor, 
                                                  final_div_factor=final_div_factor, 
                                                  total_steps=total_steps)
    
    lr_step = 'iteration'

    print('Initialised OneCycle LR scheduler.')

    return sche_fn, lr_step