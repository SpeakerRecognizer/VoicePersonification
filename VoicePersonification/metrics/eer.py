import torch

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from typing import Any
from torchmetrics import Metric


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
        self.embs = {}

    def update(self, key: str, emb: torch.Tensor):
        self.embs[key] = emb

    def reset(self) -> None:
        self.embs = None
        return super().reset()
    
    def compute(self, protocol: pd.DataFrame):
        def get_cossim(s):
            enroll = self.embs[s["enroll"]].reshape(1, -1)
            test = self.embs[s["test"]].reshape(1, -1)
            return cosine_similarity(enroll, test)[0, 0]
        
        scores = protocol.apply(get_cossim, axis=1)
        eer_val, thr_val = compute_eer(
            scores[protocol["is_target"]==1],
            scores[protocol["is_target"]==0]
        )

        return {
            "EER": eer_val,
            "Thr": thr_val
        }