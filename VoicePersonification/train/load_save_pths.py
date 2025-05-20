import os
import torch


def saveParameters(model, optimizer, scheduler, num_epoch, path):
    # Save parameters
    
    checkpoint = {}
    checkpoint['model']     = model.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['scheduler'] = scheduler[0].state_dict()
    checkpoint['num_epoch'] = num_epoch
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    torch.save(checkpoint, os.path.join(path, ''.join(['model_', str(num_epoch).zfill(4), '.pth'])))
    
def loadParameters(model, optimizer, scheduler, path):
    # Load parameters

    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler[0].load_state_dict(checkpoint['scheduler'])
    
    return checkpoint['num_epoch']