from omegaconf import DictConfig
from hydra.utils import instantiate
from lightning import Trainer, LightningModule
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger()

class PredictTask:

    def __init__(self, cfg: DictConfig) -> None:

        self.trainer: Trainer = instantiate(cfg.trainer)
        self.dataset: Dataset = instantiate(cfg.dataset)
        self.model: LightningModule = instantiate(cfg.model)
        self.dataloader: DataLoader = instantiate(cfg.dataloader, dataset=self.dataset)
        self.output_handler = cfg.output_handler

    def run(self):
        results =  self.trainer.predict(
            model=self.model,
            dataloaders=[self.dataloader]
        )
        msg = instantiate(self.output_handler, results=results)
        logger.info(msg)
        return results
