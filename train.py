import pdb
import yaml
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import os

from pytorch_lightning import Trainer, seed_everything


@hydra.main(config_path="./config", config_name="base", version_base="1.1")
def main(config: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    seed_everything(config.seed, workers=True)

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer: Trainer = hydra.utils.instantiate(config.trainer)

    training_loop = hydra.utils.instantiate(config.training_loop)
    data = hydra.utils.instantiate(config.data)

    with open(Path(config.save_dir) / "config.yaml", 'w') as f:
        yaml.dump(OmegaConf.to_container(config, resolve=True), f)

    trainer.fit(model=training_loop, datamodule=data)


if __name__ == "__main__":
    main()
