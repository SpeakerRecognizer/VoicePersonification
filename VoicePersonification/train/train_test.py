import torch 
import os
import pandas as pd
from metrics.eer import EERMetric
from tqdm import tqdm as tqdm 
from math import ceil

def train_network(train_loader, main_model, optimizer, scheduler, num_epoch, verbose=False):

    assert scheduler[1] in ['epoch', 'iteration']
    
    main_model.train()
    
    stepsize = train_loader.batch_size

    loss    = 0
    top1    = 0
    counter = 0
    index   = 0

    for data, data_label in train_loader:
        data = data.transpose(1, 0)
        
        main_model.zero_grad()

        label = torch.LongTensor(data_label).cuda()

        nloss, prec1 = main_model(data, label)
        nloss.backward()
        optimizer.step()

        loss    += nloss.detach().cpu().item()
        top1    += prec1.detach().cpu().item()
        counter += 1
        index   += stepsize
        
        if verbose:
            print("Epoch {:1.0f}, Batch {:1.0f}, LR {:f} Loss {:f}, Accuracy {:2.3f}%".format(num_epoch, counter, optimizer.param_groups[0]['lr'], loss/counter, top1/counter))

        if scheduler[1] == 'iteration': scheduler[0].step()

    if scheduler[1] == 'epoch': scheduler[0].step()

    return (loss/counter, top1/counter)

def prepare_pandas_protocol(protocol_path: str,
                            imposter_fname: str = "imp-enroll-test.txt",
                            targets_fname: str = "tar-enroll-test.txt",):
    names = ["enroll", "test"]
    imposters_pairs = pd.read_csv(os.path.join(protocol_path, imposter_fname), sep=" ", names=names)
    targets_pairs = pd.read_csv(os.path.join(protocol_path, targets_fname), sep=" ", names=names)
    imposters_pairs["is_target"] = 0
    targets_pairs["is_target"] = 1
    protocol = pd.concat([imposters_pairs, targets_pairs])
    return protocol


def test_network(test_loader, main_model, protocol_path, chunk_size=1000, min_chunk_size=100):
    # Function to test model    
    
    protocol = prepare_pandas_protocol(protocol_path)
    main_model.eval()
    eer = EERMetric()
    print(f"Start validation on {protocol_path}...")
    with torch.no_grad():
        for data_label, data in tqdm(test_loader, total=len(test_loader.dataset)):
            steps = max(1, ceil((data.shape[-1] - min_chunk_size) \
                        / chunk_size))
            weights = torch.zeros([steps, 1])
            embs = []
            for step in range(steps):
                cur_feat = data[..., (step *  chunk_size):((step+1) *  chunk_size)]
                weights[step] = cur_feat.shape[-1]
                embs.append(main_model(cur_feat.transpose(0,1)))
            embedding = torch.cat(embs)
            embedding = embedding * weights.to(embedding.device)
            embedding = embedding.sum(dim=0)
            eer.update(data_label[0], embedding)
    print(eer.compute(protocol))    
