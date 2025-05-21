import os
import torch
import torchaudio as ta
import numpy as np
import pandas as pd
import soundfile as sf
import kaldiio

from torch.utils.data import Dataset

from VoicePersonification.utils.data import read_scp, apply_vad


class VerificationTestDataset(Dataset):

    def __init__(self, wav_scp: str, 
                 vad_scp: str, 
                 protocol_path: str,
                 sample_rate: int,
                 imposter_fname: str = "imp-enroll-test.txt",
                 targets_fname: str = "tar-enroll-test.txt",
                 dtype='float64'):
        self.wav_dct = read_scp(wav_scp)
        self.vad_dct = read_scp(vad_scp)
        self.wav_lst = sorted(self.wav_dct.keys())

        self.dtype = dtype
        self.sample_rate = sample_rate

        names = ["enroll", "test"]
        imposters_pairs = pd.read_csv(os.path.join(protocol_path, imposter_fname), sep="\t", names=names)
        targets_pairs = pd.read_csv(os.path.join(protocol_path, targets_fname), sep="\t", names=names)
        imposters_pairs["is_target"] = 0
        targets_pairs["is_target"] = 1
        
        self.protocol = pd.concat([imposters_pairs, targets_pairs])

    def __getitem__(self, index):
        key = self.wav_lst[index]
        wav, sr = sf.read(self.wav_dct[key], dtype=self.dtype)
        vad = kaldiio.load_mat(self.vad_dct[key]).astype(np.bool_)

        if sr != self.sample_rate:
            wav = ta.functional.resample(torch.from_numpy(wav), sr, self.sample_rate).numpy()

        wav = apply_vad(wav, vad)

        return key, torch.from_numpy(wav.astype(np.float32)).squeeze()

    def __len__(self):
        return len(self.wav_lst)
    
    def get_protocol(self):
        return self.protocol
