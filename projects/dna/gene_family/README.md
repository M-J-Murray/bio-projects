# Gene Family Classification

This project attempts to categorise gene sequences into their relevant family

## Install instructions:
* Install UV: https://docs.astral.sh/uv/getting-started/installation/
* Install mmseqs2: https://github.com/soedinglab/mmseqs2
* Run `uv sync` to install the relevant dependencies to run the scripts.
* Place the DNA seq gene family data into a folder called "data" at the root of the project, and name the file "dna_seq_families.csv".

## Running the pipeline
All scripts can be found within the `scripts` folder. Each script is partnered with a config file with the same name. These config files contain the parameters for running the scripts.

The order of running scripts is:
1. `create_datasets.py` - this creates the dataset splits
2. `train_naive_model.py` - this train the naive model and saves a checkpoint in a `models/` folder.
3. `train_nt_model.py` - this trains the refined NT50M model and saves the checkpoints to the `models/`.
4. `benchmark_models.py` - this benchmarks the model checkpoints against the test dataset and prints the performance metrics. 