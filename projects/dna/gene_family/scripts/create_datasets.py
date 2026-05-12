"""Script to generate the train/val/test splits from a gene family classification dataset."""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from composite.util.config import BaseConfig


class Config(BaseConfig):
    dataset_path: Path
    output_path: Path
    seq_col: str
    label_col: str
    train_frac: float
    val_frac: float
    test_frac: float
    min_seq_len: int
    max_seq_len: int
    mmseqs_min_id: float  # The minimum clustering similarity, 0.5 means 50% sequence identity.
    seed: int


def run_mmseqs_clustering(fasta_path: Path, out_prefix: str, tmp_dir: Path, mmseqs_min_id: float) -> None:
    """Runs MMseqs2 easy-cluster on a fasta file."""
    cmd = [
        "mmseqs",
        "easy-cluster",
        fasta_path,
        tmp_dir / out_prefix,
        tmp_dir,
        "--min-seq-id",
        str(mmseqs_min_id),
        "--cov-mode",
        "1",  # coverage of target
        "-c",
        "0.8",  # 80% coverage required (prevents tiny fragments clustering with long ones)
        "-s",
        "7.5",  # For nucleotide clustering, or may miss valid matches
        "--alignment-mode",
        "3",
        "--threads",
        "4",
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"MMseqs2 failed: {e.stderr.decode('utf-8')}")
        raise


def split_clusters_by_size(
    clusters_dict: dict[str, list[str]], total_seqs: int, train_f: float, val_f: float
) -> tuple[list[str], list[str], list[str]]:
    """
    Optimally allocates clusters into train/val/test splits to maintain target distribution.
    Uses a greedy 'largest-first' allocation based on split deficits.
    """
    test_f = 1.0 - train_f - val_f

    # Calculate exact target number of sequences per split
    targets = {"train": total_seqs * train_f, "val": total_seqs * val_f, "test": total_seqs * test_f}

    # Sort clusters by size (number of members) in DESCENDING order.
    # Fitting the "biggest rocks" first ensures we don't accidentally
    # overshoot our targets drastically at the end of the allocation.
    sorted_clusters = sorted(clusters_dict.items(), key=lambda x: len(x[1]), reverse=True)

    lists = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for _, members in sorted_clusters:
        size = len(members)

        # Calculate current deficit for each split (target - current count)
        deficits = {
            "train": targets["train"] - counts["train"],
            "val": targets["val"] - counts["val"],
            "test": targets["test"] - counts["test"],
        }

        # Find the split that is furthest away from its target capacity
        # In the event of a tie, max() returns the first key encountered (train -> val -> test)
        best_split = max(deficits, key=deficits.get)

        # Allocate the cluster to that split
        lists[best_split].extend(members)
        counts[best_split] += size

    return lists["train"], lists["val"], lists["test"]


def main() -> None:
    config = Config.load_script_config()
    np.random.seed(config.seed)
    dataset_df = pd.read_csv(config.dataset_path)

    # Remove sequences with length less than threshold
    min_len_mask = dataset_df[config.seq_col].apply(len) >= config.min_seq_len
    dataset_df = dataset_df[min_len_mask]
    print(f"Removed {sum(~min_len_mask)} sequences with length less than {config.min_seq_len}, kept {len(dataset_df)}")

    # Remove sequences with length more than threshold
    max_len_mask = dataset_df[config.seq_col].apply(len) <= config.max_seq_len
    dataset_df = dataset_df[max_len_mask]
    print(f"Removed {sum(~max_len_mask)} sequences with length more than {config.max_seq_len}, kept {len(dataset_df)}")

    # Create a unique ID for every sequence so we can track them through MMseqs
    dataset_df["seq_id"] = [f"seq_{i}" for i in range(len(dataset_df))]
    dataset_df = dataset_df.set_index("seq_id", drop=False)

    train_indices, val_indices, test_indices = [], [], []

    # We use a temporary directory to avoid cluttering your workspace with mmseqs files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        grouped = dataset_df.groupby(config.label_col)
        print(f"Found {len(grouped)} gene families. Beginning clustering and splitting...")

        for family, group in grouped:
            print(f"  -> Processing family: {family} ({len(group)} sequences)")

            # 1. Write family sequences to a temporary FASTA file
            fasta_path = tmp_dir / f"temp_{family}.fasta"
            with open(fasta_path, "w") as f:
                for seq_id, row in group.iterrows():
                    f.write(f">{seq_id}\n{row[config.seq_col]}\n")

            # 2. Run MMseqs2 clustering
            out_prefix = f"out_{family}"
            mmseqs_tmp = tmp_dir / f"mmseqs_tmp_{family}"
            mmseqs_tmp.mkdir(parents=True, exist_ok=True)

            run_mmseqs_clustering(fasta_path, out_prefix, mmseqs_tmp, config.mmseqs_min_id)

            # 3. Parse MMseqs2 output (*_cluster.tsv)
            # Column 1: Representative sequence (Cluster ID) | Column 2: Member sequence
            cluster_tsv = mmseqs_tmp / f"{out_prefix}_cluster.tsv"

            if not cluster_tsv.exists():
                print(f"     Warning: Clustering failed or no output for {family}. Putting all in train.")
                train_indices.extend(group["seq_id"].tolist())
                continue

            clusters = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["rep", "member"])

            # Group into a dictionary: { rep_id: [member_id_1, member_id_2, ...] }
            clusters_dict = clusters.groupby("rep")["member"].apply(list).to_dict()
            num_clusters = len(clusters_dict)

            # 4. Handle Edge Cases & Split
            if num_clusters == 1:
                print(
                    f"     Warning: Family '{family}' collapsed into a single cluster. "
                    f"It cannot be split without data leakage. Assigning entirely to Train."
                )
                train_indices.extend(group["seq_id"].tolist())
            else:
                tr, v, ts = split_clusters_by_size(clusters_dict, len(group), config.train_frac, config.val_frac)
                train_indices.extend(tr)
                val_indices.extend(v)
                test_indices.extend(ts)

    # 5. Create final dataframes
    print("Splitting complete. Compiling final datasets...")
    train_df = dataset_df.loc[train_indices].drop(columns=["seq_id"])
    val_df = dataset_df.loc[val_indices].drop(columns=["seq_id"])
    test_df = dataset_df.loc[test_indices].drop(columns=["seq_id"])

    # Check all sequences exist in outputs
    recreated_df = pd.concat([train_df, val_df, test_df]).sort_values(config.seq_col).reset_index(drop=True)
    if recreated_df.equals(dataset_df.drop(columns=["seq_id"]).sort_values(config.seq_col).reset_index(drop=True)):
        print("Successfully matched all split datapoints to original dataset.")
    else:
        print("     Warning: Dataset splits do not contain all sequences from original dataset.")

    # Display final stats with family distributions
    total = len(dataset_df)
    print("Final Split Sizes:")
    print(f"Train: {len(train_df)} ({len(train_df) / total * 100:.1f}%)")
    print(
        ", ".join(
            [f"{fam}: {len(group_df) / len(train_df):.2%}" for fam, group_df in train_df.groupby(config.label_col)]
        )
    )
    print(f"Val:   {len(val_df)} ({len(val_df) / total * 100:.1f}%)")
    print(
        ", ".join([f"{fam}: {len(group_df) / len(val_df):.2%}" for fam, group_df in val_df.groupby(config.label_col)])
    )
    print(f"Test:  {len(test_df)} ({len(test_df) / total * 100:.1f}%)")
    print(
        ", ".join([f"{fam}: {len(group_df) / len(test_df):.2%}" for fam, group_df in test_df.groupby(config.label_col)])
    )

    # 6. Save to disk
    train_df.to_csv(config.output_path / "train.csv", index=False)
    val_df.to_csv(config.output_path / "val.csv", index=False)
    test_df.to_csv(config.output_path / "test.csv", index=False)
    print("Saved to train.csv, val.csv, and test.csv")


if __name__ == "__main__":
    main()
