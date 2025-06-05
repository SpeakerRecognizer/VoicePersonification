import sys, os
dpath = '/mnt/cs/voice/korenevskaya-a/VoicePersonification/VoicePersonification'
sys.path.append(dpath)
dpath = '/mnt/cs/voice/korenevskaya-a/VoicePersonification/'
sys.path.append(dpath)

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

from models.wav2vec_bert import Wav2VecBERTModel, Wav2Vec2BertFeatures
from data.verification_train import VerificationTrainDataset
from data.verification_test import VerificationTestDataset
from train.loss_functions import AAMSoftmaxLoss, CELoss
from train.optimizers import AdamOptimizer, SGDOptimizer
from train.schedulers import StepLRScheduler, OneCycleLRScheduler
from train.main_model import MainModel
from train.train_test import train_network, test_network
from train.load_save_pths import loadParameters, saveParameters
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from metrics.eer import EERMetric

# Select hyperparameters

nOut              = 512                                  # embedding size

# Loss function for angular losses
margin            = 0.35                                 # margin parameter
scale             = 32.0                                 # scale parameter

# Train dataloader
max_frames_train  = 200                                  # number of frame to train
train_path        = 'data/raw_data/voxceleb1_dev/wav'    # path to train wav files
batch_size_train  = 5                                  # batch size to train
pin_memory        = False                                # pin memory
num_workers_train = 5                                    # number of workers to train
shuffle           = True                                 # shuffling of training examples

# Test dataloader
max_frames_test   = 1000                                 # number of frame to test
test_path         = 'data/raw_data/voxceleb1_test/wav'   # path to val wav files
batch_size_test   = 1                                  # batch size to test
num_workers_test  = 5                                    # number of workers to test

# Optimizer
lr                = 0.01                                  # learning rate value
weight_decay      = 0                                    # weight decay value

# Scheduler
val_interval      = 1                                    # frequency of validation step
max_epoch         = 2                                    # number of epoches

def train():
    # Initialize train dataloader (without augmentation)
    data_path = '/mnt/cs/voice/korenevskaya-a/VoicePersonification/'
    feature_extractor = Wav2Vec2BertFeatures()
    train_dataset = VerificationTrainDataset('/mnt/cs/voice/korenevskaya-a/data/vc2_subsample/wav_subsample.scp', #data_path + 'data/scp/voxceleb1/dev/wav.scp', 
                                            '/mnt/cs/voice/korenevskaya-a/data/vc2_subsample/utt2spk_subsample', #data_path + 'data/scp/voxceleb1/dev/utt2spk',
                                            16000,
                                            max_frames_train, 
                                            feature_extractor=feature_extractor)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size_train, pin_memory=pin_memory, num_workers=num_workers_train, shuffle=shuffle)

    # Initialize validation dataloader
    test_dataset = VerificationTestDataset(data_path + 'data/scp/voxceleb1/test/wav.scp', 
                                        '',
                                        data_path + 'data/protocols',
                                        16000,
                                        feature_extractor=feature_extractor)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size_test, num_workers=num_workers_test)

    # Initialize model
    model      = Wav2VecBERTModel()
    trainfunc  = AAMSoftmaxLoss(nOut, 3, margin=margin, scale=scale)
    main_model = MainModel(model, trainfunc).cuda()

    # Initialize optimizer and scheduler
    optimizer = SGDOptimizer(main_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = OneCycleLRScheduler(optimizer, 
                                    pct_start=0.05, 
                                    cycle_momentum=False, 
                                    max_lr=lr, 
                                    div_factor=100, 
                                    final_div_factor=10000, 
                                    total_steps=max_epoch*len(train_loader))

    # training                
    start_epoch = 0
    checkpoint_flag = False

    if checkpoint_flag:
        start_epoch = loadParameters(main_model, optimizer, scheduler, path= data_path + 'data/w2v_bert_train/1.ckpt')
        start_epoch = start_epoch + 1

    # Train model
    for num_epoch in range(start_epoch, max_epoch):
        train_loss, train_top1 = train_network(train_loader, main_model, optimizer, scheduler, num_epoch, verbose=True)
        
        print("Epoch {:1.0f}, Loss (train set) {:f}, Accuracy (train set) {:2.3f}%".format(num_epoch, train_loss, train_top1))

        if (num_epoch + 1)%val_interval == 0:
            test_network(test_loader, main_model, 'data/protocols/')
            saveParameters(main_model, optimizer, scheduler, num_epoch, path=data_path + 'data/w2v_bert_train')                                


if __name__ == '__main__':
    train()