import os
import torch
import torchaudio as ta
import numpy as np
import pandas as pd
import soundfile as sf
import kaldiio

from torch.utils.data import Dataset

from VoicePersonification.utils.data import read_scp, apply_vad, parse_utt2spk


import os
import glob

import numpy
import random
import soundfile
from scipy import signal

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def loadWAV(filename, max_frames, evalmode=True, num_eval=10):
    # Load wav file
    
    max_audio = max_frames*160 + 240

    audio, sample_rate = soundfile.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage  = max_audio - audiosize + 1 
        audio     = numpy.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    if evalmode:
        startframe = numpy.linspace(0, audiosize - max_audio, num=num_eval)
    
    else:
        startframe = numpy.array([numpy.int64(random.random()*(audiosize - max_audio))])
    
    feats = []
    
    if evalmode and max_frames == 0:
        feats.append(audio)
    
    else:
        
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])

    feat = numpy.stack(feats, axis=0).astype(float)

    return feat, sample_rate

class VerificationTrainDataset(Dataset):

    def __init__(self, wav_scp: str, 
                 utt2spk: str,  
                 sample_rate: int,
                 max_frames: int,
                 vad_scp: str = '',
                 feature_extractor=None):
        self.wav_dct = read_scp(wav_scp)
        if vad_scp != '':
            self.vad_dct = read_scp(vad_scp)
        else:
            self.vad_dct = None    
        self.wav_lst = sorted(self.wav_dct.keys())
        self.max_frames = max_frames
        self.sample_rate = sample_rate
        self.utt2id= parse_utt2spk(utt2spk)
        self.fe = feature_extractor 
        self.id_list = list(set(self.utt2id.values()))
        self.id_dict = {self.id_list[i]: i for i in range(len(self.id_list))}
            
    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def classes(self):
        return list(set(self.utt2id.values()))
    
    def __getitem__(self, index):
        key = self.wav_lst[index]
        wav, sr = loadWAV(self.wav_dct[key], self.max_frames, evalmode=False)
        
        if sr != self.sample_rate:
            wav = ta.functional.resample(torch.from_numpy(wav), sr, self.sample_rate).numpy()

        if self.vad_dct != None:
            vad = kaldiio.load_mat(self.vad_dct[key]).astype(np.bool_)
            wav = apply_vad(wav, vad)
        
        spk = self.utt2id[key]
        out = torch.from_numpy(wav.astype(np.float32)).squeeze()
        if self.fe != None:
            out = self.fe(out.unsqueeze(0))

        return out, self.id_dict[spk]

    def __len__(self):
        return len(self.wav_lst)