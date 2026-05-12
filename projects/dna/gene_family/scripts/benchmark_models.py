"""Script to benchmark model checkpoints against the test datasets."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from composite.util.config import BaseConfig
from models.nt50_gene_family.arch.nt50m_model import DNASequenceDataset, NTGeneClassifier, build_ia3_model


class Config(BaseConfig):
    # Data params
    test_data_path: Path
    seq_col: str
    label_col: str

    # Naive model params
    naive_model_path: Path

    # Nucleotide transformer model params
    nt_model_name: str
    nt_checkpont_path: Path
    num_labels: int
    max_seq_len: int
    batch_size: int
    device: str


def load_checkpoint(ckpt_path: Path, model_name: str, num_labels: int) -> NTGeneClassifier:
    """
    Rebuilds the IA3 model and loads the weights from a PyTorch Lightning checkpoint.
    """
    model = build_ia3_model(model_name, num_labels)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    # Replace "model." with whatever you named your model variable in your
    lightning_prefix = "model."
    clean_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(lightning_prefix):
            clean_key = key[len(lightning_prefix) :]
        else:
            clean_key = key

        clean_state_dict[clean_key] = value

    model.load_state_dict(clean_state_dict, strict=False)
    return model


def nt_model_inference(
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    dna_seqs: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Runs inference on a dataset and returns a DataFrame with true labels,
    predictions, and class probabilities.
    """
    dataset = DNASequenceDataset(tokenizer=tokenizer, max_length=max_length, sequences=dna_seqs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # No need to softmax as the class rank is the same
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu().numpy())

    return np.concat(all_preds)


def print_model_metrics(y_test: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(title)
    print(f"Overall Accuracy : {acc:.4f}")
    print(f"Macro F1-Score   : {macro_f1:.4f}")
    print(f"Weighted F1-Score   : {weighted_f1:.4f}")
    print(f"Multiclass MCC   : {mcc:.4f}")
    print(report)

@Config.main()
def main(config: Config) -> None:
    test_df = pd.read_csv(config.test_data_path)
    X_test, y_test = test_df[config.seq_col], test_df[config.label_col]

    naive_model = joblib.load(config.naive_model_path)
    y_naive = naive_model.predict(X_test)

    tokenizer = AutoTokenizer.from_pretrained(config.nt_model_name, trust_remote_code=True)
    device = torch.device(config.device)
    nt50m_model = load_checkpoint(config.nt_checkpont_path, config.nt_model_name, config.num_labels).to(device)
    y_nt50m = nt_model_inference(
        tokenizer,
        nt50m_model,
        X_test.tolist(),
        config.max_seq_len,
        config.batch_size,
        device,
    )

    print_model_metrics(np.array(y_test), y_naive, "Naive Model Test Performance")
    print_model_metrics(np.array(y_test), y_nt50m, "NT50M Model Test Performance")


if __name__ == "__main__":
    main()
