"""Script to train and validate the naive kmer count logitic regressor model."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from composite.util.config import BaseConfig
from models.nt50_gene_family.arch.nt50m_model import DNASequenceDataset, build_ia3_model

warnings.filterwarnings("ignore", category=UserWarning)


class GeneClassifierLightning(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        ia3_lr: float,
        head_lr: float,
        class_weights: torch.Tensor,
        dropout: float,
    ):
        super().__init__()
        self.ia3_lr = ia3_lr
        self.head_lr = head_lr
        self.dropout = dropout

        self.model = build_ia3_model(model_name, num_labels, self.dropout)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self._val_preds = []
        self._val_labels = []

    def _step(self, batch, batch_idx, stage: str):
        logits = self.model(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(logits, batch["labels"])
        preds = logits.argmax(dim=-1)
        acc = (preds == batch["labels"]).float().mean()

        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, sync_dist=True)

        if stage == "val":
            self._val_preds.extend(preds.cpu().numpy().tolist())
            self._val_labels.extend(batch["labels"].cpu().numpy().tolist())

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return

        preds = np.array(self._val_preds)
        labels = np.array(self._val_labels)

        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        acc = accuracy_score(labels, preds)

        self.log("val/macro_f1", macro_f1, prog_bar=True)
        self.log("val/weighted_f1", weighted_f1, prog_bar=True)
        self.log("val/epoch_acc", acc)

        print(f"\n{'=' * 60}")
        print(f"Epoch {self.current_epoch} — Validation Summary")
        print(f"  Accuracy:    {acc:.4f}")
        print(f"  Macro F1:    {macro_f1:.4f}")
        print(f"  Weighted F1: {weighted_f1:.4f}")
        print(classification_report(labels, preds, zero_division=0))
        print(f"{'=' * 60}\n")

        self._val_preds.clear()
        self._val_labels.clear()

    def configure_optimizers(self):
        # Only optimise trainable params (IA³ vectors + classifier head)
        ia3_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        head_params = [p for p in self.model.classifier.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": ia3_params, "lr": self.ia3_lr},
                {"params": head_params, "lr": self.head_lr},
            ],
        )
        return {
            "optimizer": optimizer,
        }


class GeneDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        tokenizer_name: str,
        train_batch_size: int,
        val_batch_size: int,
        max_length: int,
        seq_col: str,
        label_col: str,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.tokenizer_name = tokenizer_name
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.max_length = max_length
        self.seq_col = seq_col
        self.label_col = label_col
        self.class_weights = None

    def setup(self, stage=None):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=True)
        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        self.train_ds = DNASequenceDataset(
            tokenizer=tokenizer,
            max_length=self.max_length,
            sequences=train_df[self.seq_col].tolist(),
            labels=train_df[self.label_col].tolist(),
        )
        self.val_ds = DNASequenceDataset(
            tokenizer=tokenizer,
            max_length=self.max_length,
            sequences=val_df[self.seq_col].tolist(),
            labels=val_df[self.label_col].tolist(),
        )

        # Compute class weights from training set to handle imbalance
        labels = np.array(self.train_ds.labels)
        cw = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
        self.class_weights = torch.tensor(cw, dtype=torch.float32)
        print(f"Class weights: {self.class_weights.tolist()}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            pin_memory=True,
        )


class Config(BaseConfig):
    train_data_path: Path  # Path to training CSV
    val_data_path: Path  # Path to validation CSV
    output_dir: Path  # Path to output model checkpoints and logging
    seq_col: str
    label_col: str
    num_labels: int
    model_name: str
    max_seq_length: int  # NT V2 context window
    train_batch_size: int
    val_batch_size: int
    max_epochs: int
    ia3_lr: float
    head_lr: float
    patience: int  # Early stopping patience (epochs)
    dropout: float
    seed: int
    precision: str  # Training precision, choices: "32", "16-mixed", "bf16-mixed"


def main():
    config = Config.load_script_config()
    pl.seed_everything(config.seed)

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    dm = GeneDataModule(
        train_path=str(config.train_data_path),
        val_path=str(config.val_data_path),
        tokenizer_name=config.model_name,
        train_batch_size=config.train_batch_size,
        val_batch_size=config.val_batch_size,
        max_length=config.max_seq_length,
        seq_col=config.seq_col,
        label_col=config.label_col,
    )
    dm.setup()

    model = GeneClassifierLightning(
        model_name=config.model_name,
        num_labels=config.num_labels,
        ia3_lr=config.ia3_lr,
        head_lr=config.head_lr,
        class_weights=dm.class_weights,
        dropout=config.dropout,
    )

    callbacks = [
        EarlyStopping(
            monitor="val/macro_f1",
            patience=config.patience,
            mode="max",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=config.output_dir / "checkpoints",
            filename="nt50m-ia3-{epoch:02d}-{val/macro_f1:.4f}",
            monitor="val/macro_f1",
            mode="max",
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        ## Useful for debugging
        # overfit_batches=1,
        # limit_val_batches=0,
        # check_val_every_n_epoch=10000000,
        # enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        max_epochs=config.max_epochs,
        limit_train_batches=0.1,
        accelerator="auto",
        devices="auto",
        precision=config.precision,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        deterministic=False,  # True breaks some attention kernels
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=dm)

    best_path = callbacks[1].best_model_path
    print(f"Best checkpoint: {best_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
