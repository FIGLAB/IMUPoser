# %%
# %load_ext autoreload
# %autoreload 2

# %%
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

from imuposer.config import Config, amass_combos
from imuposer.models.utils import get_model
from imuposer.datasets.utils import get_datamodule
from imuposer.utils import get_parser
from imuposer.get_device import get_device


def main():
    # set the random seed
    seed_everything(42, workers=True)

    parser = get_parser()
    args = parser.parse_args()
    combo_id = args.combo_id
    fast_dev_run = args.fast_dev_run
    _experiment = args.experiment

    # %%
    device = get_device()
    print(f"Using device: {device}")

    config = Config(
        experiment=f"{_experiment}_{combo_id}",
        model="GlobalModelIMUPoser",
        project_root_dir="../../",
        joints_set=amass_combos[combo_id],
        normalize="no_translation",
        r6d=True,
        loss_type="mse",
        use_joint_loss=True,
        device=device,
    )

    # %%
    # instantiate model and data
    model = get_model(config)
    datamodule = get_datamodule(config)
    checkpoint_path = config.checkpoint_path

    # %%
    wandb_logger = WandbLogger(project=config.experiment, save_dir=checkpoint_path)

    early_stopping_callback = EarlyStopping(
        monitor="validation_step_loss",
        mode="min",
        verbose=False,
        min_delta=0.00001,
        patience=5,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_step_loss",
        mode="min",
        verbose=False,
        save_top_k=5,
        dirpath=checkpoint_path,
        save_weights_only=True,
        filename="epoch={epoch}-val_loss={validation_step_loss:.5f}",
    )

    # trainer = pl.Trainer(fast_dev_run=fast_dev_run, logger=wandb_logger, max_epochs=1000, accelerator="gpu", devices=[0],
    #                      callbacks=[early_stopping_callback, checkpoint_callback], deterministic=True)

    if device.type == "mps":
        trainer = pl.Trainer(
            fast_dev_run=fast_dev_run,
            logger=wandb_logger,
            max_epochs=1000,
            accelerator="mps",
            devices=[0],
            callbacks=[early_stopping_callback, checkpoint_callback],
            deterministic=True,
        )
    elif device.type == "cuda":
        trainer = pl.Trainer(
            fast_dev_run=fast_dev_run,
            logger=wandb_logger,
            max_epochs=1000,
            accelerator="gpu",
            devices=[0],
            callbacks=[early_stopping_callback, checkpoint_callback],
            deterministic=True,
        )
    else:
        trainer = pl.Trainer(
            fast_dev_run=fast_dev_run,
            logger=wandb_logger,
            max_epochs=1000,
            accelerator="cpu",
            callbacks=[early_stopping_callback, checkpoint_callback],
            deterministic=True,
        )
    # %%
    trainer.fit(model, datamodule=datamodule)

    # %%
    with open(checkpoint_path / "best_model.txt", "w") as f:
        f.write(
            f"{checkpoint_callback.best_model_path}\n\n{checkpoint_callback.best_k_models}"
        )


if __name__ == "__main__":
    main()
