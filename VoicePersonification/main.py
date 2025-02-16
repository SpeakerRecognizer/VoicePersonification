import hydra

from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base="1.3.2",config_name="", config_path="")
def main(cfg: DictConfig):
    task_cls = instantiate(cfg.task, _partial_=True)
    task = task_cls(cfg=cfg)
    task.run()

if __name__ == "__main__":
    main()