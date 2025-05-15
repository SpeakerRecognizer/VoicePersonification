import torch

from lightning import LightningModule

from VoicePersonification.metrics.eer import EERMetric

class VerificationModel(LightningModule):

    def __init__(self, *args: torch.Any, **kwargs: torch.Any) -> None:
        super().__init__(*args, **kwargs)

    def on_test_epoch_start(self) -> None:
        self.metrics = [EERMetric() for _ in range(len(self.trainer.test_dataloaders))]

    def test_step(self, batch: tuple, btach_id: int, ):
        key, feats = batch
        return key, self(feats)

    def on_test_batch_end(self, outputs: tuple, batch: torch.Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        key, emb = outputs
        self.metrics[dataloader_idx].update(
            key=key[0],
            emb=emb.cpu().squeeze()
        )

    def on_test_epoch_end(self) -> None:
        outs = []

        for i, metric in enumerate(self.metrics):
            outs.append(metric.compute(
                self.trainer.test_dataloaders[i].dataset.get_protocol()
            ))
            metric.reset()
        self.metrics = None
        
        for out in outs:
            self.log_dict(out)