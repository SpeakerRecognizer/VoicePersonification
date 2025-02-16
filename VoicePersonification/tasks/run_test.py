from omegaconf import DictConfig
from hydra.utils import instantiate
from lightning import Trainer, LightningModule
from torch.utils.data import Dataset, DataLoader

class VerificationTestTask:

    def __init__(self, cfg: DictConfig) -> None:

        self.trainer: Trainer = instantiate(cfg.trainer)
        self.dataset: Dataset = instantiate(cfg.dataset)
        self.model: LightningModule = instantiate(cfg.model)
        self.dataloader: DataLoader = instantiate(cfg.dataloader, dataset=self.dataset)

    def run(self):
        return self.trainer.test(
            model=self.model,
            dataloaders=[self.dataloader]
        )