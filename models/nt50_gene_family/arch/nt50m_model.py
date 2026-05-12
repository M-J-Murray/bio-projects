"""Module for the Nucleotide Transformer 50M parameter model code."""

from typing import Optional

import torch
import torch.nn as nn
from peft import IA3Config, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForMaskedLM


class DNASequenceDataset(Dataset):
    """Loads a CSV with columns: `sequence` and `label`."""

    def __init__(
        self,
        tokenizer,
        max_length: int,
        sequences: list[str],
        labels: Optional[list[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        encoding = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class NTGeneClassifier(nn.Module):
    """
    NT V2 backbone (frozen except IA³ params) + mean-pool + classification head.
    """

    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.0):
        super().__init__()
        self.backbone = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).esm
        hidden_size = self.backbone.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_labels),
        )

    def mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        """Masked mean pooling — ignores padding tokens."""
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)  # (B, S, D)
        pooled = self.mean_pool(out.last_hidden_state, attention_mask)
        return self.classifier(pooled)


def build_ia3_model(model_name: str, num_labels: int, dropout: float = 0.1) -> NTGeneClassifier:
    """
    Wraps the NT backbone with IA³ via PEFT.

    IA³ injects learnable scaling vectors into:
      - key & value projections in attention  (ff_key, ff_value targets)
      - the feed-forward up-projection        (ff_intermediate target)

    We target these layers by name as they appear in the NT/ESM architecture.
    Only ~0.01% of parameters become trainable, drastically reducing memory
    and overfitting risk for small biological datasets.
    """
    base = NTGeneClassifier(model_name, num_labels, dropout=dropout)

    ia3_config = IA3Config(
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["key", "value", "dense"],  # attention K/V + FFN dense
        feedforward_modules=["dense"],  # which of the above are FF
        modules_to_save=["classifier"],  # always train the head
    )

    # Apply IA³ only to the backbone, not the head
    base.backbone = get_peft_model(base.backbone, ia3_config)
    base.backbone.print_trainable_parameters()
    return base
