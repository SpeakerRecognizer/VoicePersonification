import torch

import numpy as np
import pandas as pd

import torch.distributed
from itertools import chain
from typing import Any
from torchmetrics import Metric
from collections import defaultdict

def cdf(scores, x):
    yp = np.arange(1, 1 + scores.shape[0]) / scores.shape[0]
    xp = np.sort(scores, axis=0)
    
    return np.interp(x, xp, yp)

def compute_eer(tar_scores: np.array, imp_scores: np.array):
    x = np.sort(np.concatenate([tar_scores, imp_scores], axis=0))

    far = 1 - cdf(imp_scores, x)
    frr = cdf(tar_scores, x)

    i = np.argmin(np.abs(far - frr))

    return frr[i], x[i]


class EERMetric(Metric):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.embs = defaultdict(list)

    def update(self, key: str, emb: torch.Tensor):
        self.embs[key].append(emb)

    def reset(self) -> None:
        self.embs = defaultdict(list)
        return super().reset()
    
    def compute(self, protocol: pd.DataFrame, require_snc: bool = True):
        if require_snc:
            self.sync_embs()

        def get_cossim(s):
            enroll = torch.mean(torch.stack(self.embs[s["enroll"]]), dim=0, keepdim=True)
            test = torch.mean(torch.stack(self.embs[s["test"]]), dim=0, keepdim=True)
            return torch.nn.functional.cosine_similarity(enroll, test).item()
        
        scores = protocol.apply(get_cossim, axis=1)
        eer_val, thr_val = compute_eer(
            scores[protocol["is_target"]==1],
            scores[protocol["is_target"]==0]
        )

        return {
            "EER": eer_val,
            "Thr": thr_val
        }
    
    def sync_embs(self):
        if not torch.distributed.is_initialized():
            return
        out = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(out, self.embs)
        
        keys = set(chain.from_iterable(out))
        cat_lists = lambda lsts: list(chain.from_iterable(lsts))
        embs = defaultdict(list)

        for key in keys:
            embs[key] = cat_lists(map(lambda e: e[key], out))
        
        self.embs = embs