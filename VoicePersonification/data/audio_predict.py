import torch
import torchaudio as ta
import numpy as np
import soundfile as sf

from torch.utils.data import Dataset

from VoicePersonification.utils.data import read_scp


class AudioPredictDataset(Dataset):

    def __init__(self, wav_scp: str, 
                 sample_rate: int,
                 feature_extractor=None):
        self.wav_dct = read_scp(wav_scp)
        self.wav_lst = sorted(self.wav_dct.keys())
        self.sample_rate = sample_rate
        self.fe = feature_extractor

    def __getitem__(self, index):
        key = self.wav_lst[index]
        wav, sr = sf.read(self.wav_dct[key])
        if sr != self.sample_rate:
            wav = ta.functional.resample(torch.from_numpy(wav), sr, self.sample_rate).numpy()
        
        out = torch.from_numpy(wav.astype(np.float32)).squeeze()
        if self.fe != None:
           out = self.fe(out.unsqueeze(0))
        
        return key, out

    def __len__(self):
        return len(self.wav_lst)
