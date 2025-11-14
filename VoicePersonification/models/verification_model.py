import torch

from lightning import LightningModule
from typing import Tuple
from hydra.utils import instantiate
from torchmetrics import Accuracy
from VoicePersonification.metrics.eer import EERMetric
import logging

class VerificationModel(LightningModule):
    def __init__(self, cfg) -> None:
        super(VerificationModel, self).__init__()
        self.train_cfg = cfg
        self.criterion = instantiate(self.train_cfg.criterion)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.num_train_classes)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optim_conf = self.train_cfg.optimizer
        optimizer = instantiate(optim_conf, params=self.parameters())
        scheduler_conf = getattr(self.train_cfg, "scheduler", None)
        if scheduler_conf is not None:
            scheduler = instantiate(scheduler_conf, optimizer=optimizer, total_steps=self.trainer.estimated_stepping_batches)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return optimizer

    def on_train_epoch_start(self) -> None:
        self.total_loss = 0
        self.total_acc = 0 
        self.num_steps_last_epoch = 0
        
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        data, labels = batch
        output, logits = self.forward(data, labels)
        loss = self.criterion(output, labels)
        acc = self.train_accuracy(logits, labels)
        self.total_loss += loss
        self.total_acc += acc
        self.num_steps_last_epoch += 1
        self.log_dict({'Train loss': loss, 'Train acc': acc}, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        """Compute and log accuracy over the whole train set for the current epoch."""
        print(f'Epoch completed. Train loss: {self.total_loss/self.num_steps_last_epoch},\
                Train accuracy: {(self.total_acc/self.num_steps_last_epoch)[0]}')        

    def on_test_epoch_start(self) -> None:
        self.metrics = [EERMetric() for _ in range(len(self.trainer.test_dataloaders))]

    def test_step(self, batch: tuple, batch_id: int, ):
        key, feats = batch
        return key, self(feats)

    def on_test_batch_end(self, outputs: tuple, batch: torch.Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        key, emb = outputs
        self.metrics[dataloader_idx].update(
            key=key[0],
            emb=emb.cpu().squeeze()
        )

    def on_test_epoch_end(self) -> None:
        """
        The function `on_test_epoch_end` computes metrics for each test dataloader and logs the results.
        """
        outs = []

        for i, metric in enumerate(self.metrics):
            outs.append(metric.compute(
                self.trainer.test_dataloaders[i].dataset.get_protocol()
            ))
            metric.reset()
        self.metrics = None
        
        for out in outs:
            self.log_dict(out)

    def on_validation_epoch_start(self) -> None:
        self.metrics = EERMetric() 

    def validation_step(self, batch: tuple, batch_id: int, ):
        key, feats = batch
        return key, self(feats)

    def on_validation_batch_end(self, outputs: tuple, batch: torch.Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        key, emb = outputs
        self.metrics.update(
            key=key[0],
            emb=emb.cpu().squeeze()
        )

    def on_validation_epoch_end(self) -> None:
        outs = []
        outs.append(self.metrics.compute(
                self.trainer.val_dataloaders.dataset.get_protocol()
            ))
        self.metrics.reset()
        self.metrics = None
        
        for out in outs:
            self.log_dict(out)        

    def verify(self, enroll: torch.Tensor, test: torch.Tensor):
        device = next(iter(self.parameters())).device
        dtype = next(iter(self.parameters())).dtype

        enroll_emb = self(enroll.to(device, dtype)).view(1, -1)
        test_emb = self(test.to(device, dtype)).view(1, -1)

        return torch.cosine_similarity(enroll_emb, test_emb).item()