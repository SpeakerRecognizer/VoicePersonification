from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import Dict, Any
from dataclasses import dataclass, field
from lightning import Trainer, LightningModule
from torch.utils.data import Dataset, DataLoader

class VerificationTrainTask:

    def __init__(self, cfg: DictConfig) -> None:

        self.trainer: Trainer = instantiate(cfg.trainer)
        self.train_dataset: Dataset = instantiate(cfg.train_dataset)
        self.train_dataloader: DataLoader = instantiate(cfg.train_dataloader, dataset=self.train_dataset)
        self.val_dataset: Dataset = instantiate(cfg.val_dataset)
        self.val_dataloader: DataLoader = instantiate(cfg.val_dataloader, dataset=self.val_dataset)
        self.model: LightningModule = instantiate(cfg.model, cfg=cfg.train, _recursive_=False)

    def run(self):
        return self.trainer.fit(
            model=self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader
        )