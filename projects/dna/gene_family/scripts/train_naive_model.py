"""Script to train and validate the naive kmer count logitic regressor model."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, matthews_corrcoef
from sklearn.pipeline import Pipeline

from composite.util.config import BaseConfig

class Config(BaseConfig):
    train_data_path: Path
    val_data_path: Path
    checkpoint_path: Path
    seq_col: str
    label_col: str
    kmer_length: int
    iters: int


def print_metrics(model: Pipeline, X: pd.Series, y: pd.Series, title: str) -> None:
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    # average='macro' prevents majority classes from hiding the poor performance of minority classes.
    acc = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y, y_pred)
    report = classification_report(y, y_pred, zero_division=0)

    print(title)
    print(f"Overall Accuracy : {acc:.4f}")
    print(f"Macro F1-Score   : {macro_f1:.4f}")
    print(f"Weighted F1-Score   : {weighted_f1:.4f}")
    print(f"Multiclass MCC   : {mcc:.4f}")
    print(report)


def main() -> None:
    config = Config.load_script_config()

    Path(config.checkpoint_path.parent).mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(config.train_data_path)
    val_df = pd.read_csv(config.val_data_path)

    X_train, y_train = train_df[config.seq_col], train_df[config.label_col]
    X_val, y_val = val_df[config.seq_col], val_df[config.label_col]

    # --- Build the Pipeline ---
    # 1. CountVectorizer with analyzer='char' and ngram_range=(2,2) creates our 2-mer histogram.
    # 2. LogisticRegression with class_weight='balanced' handles imbalanced gene families.
    print(f"Building and training Pipeline ({config.kmer_length}-mer extraction -> Logistic Regression)...")
    model = Pipeline(
        [
            ("kmer_extractor", CountVectorizer(analyzer="char", ngram_range=(config.kmer_length, config.kmer_length))),
            ("classifier", LogisticRegression(max_iter=config.iters, class_weight="balanced")),
        ]
    )

    # Train the model
    model.fit(X_train, y_train)

    # Print metrics
    print_metrics(model, X_train, y_train, "TRAIN METRICS")
    print_metrics(model, X_val, y_val, "VALIDATION METRICS")

    # Save the model
    joblib.dump(model, config.checkpoint_path)
    print(f"\nModel successfully saved to {config.checkpoint_path}")


if __name__ == "__main__":
    main()
