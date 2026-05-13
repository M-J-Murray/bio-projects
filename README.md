# Bio Projects

This repository contains machine learning projects applied to different biological domains.

## Machine Learning Architectures

AlphaGenome - [Gene Expression Regression](#using-alphagenome-to-predict-gene-expression-of-cancer-gene-variants)

MergeDNA - [Masked Language Modelling (MLM) Pre-training](#pre-training-mergedna)

NucleotideTransformer - [Gene Family Classification](#fine-tuning-nucleotide-transformer-with-ia3)

## DNA Projects

### Gene Expression Regression

#### Using AlphaGenome to predict gene expression of cancer gene variants 

A notebook evaluating AlphaGenome's RNA-seq scores against cancer-associated gene variants using experimentally observed RNA-seq data from lung cancer tissue samples. The aim of this evaluation is to determine if AlphaGenome can be used to identify cancer vaccine targets that are likely to be poorly expressed.

See here: [alpha_genome_performance.ipynb](./projects/dna/gene_expression/notebooks/alpha_genome_performance.ipynb)

#### Pre-training MergeDNA

A notebook demonstrating an implementation of the [MergeDNA paper](https://arxiv.org/pdf/2511.14806).

See here: [merge_dna_demo.ipynb](./projects/dna/gene_expression/notebooks/merge_dna_demo.ipynb)

This implementation was written completely in pytorch and includes:
- The MergeDNA architecture:
    - Local Encoder with windowed attention and windowed DTEM.
    - Latent Encoder with global attention and BSM for the latent reconstruction task.
    - Latent Decoder with global attention.
    - Local Decoder with windowed attention.
- Multi-objective pretraining on the NT Genome Multi-Species dataset:
    - Full sequence reconstruction from local encoder compressed embeddings.
    - Full sequence reconstruction from latent encoder compressed embeddings (frozen local encoder).
    - Adaptive masked token modelling derived from latent encoder source matrix.

### Gene Family Classification

#### Fine-tuning Nucleotide Transformer with IA3

We assess the predictive performance of two models at classifying a given gene’s DNA into the correct gene family. The two models we assessed were a “naive” kmer count logistic regression (KCLR) model and the “refined” Nucleotide Transformer 50M (NT50M) model fine-tuned with IA3.

See data analysis: [dna_seq_families_analysis.ipynb](./projects/dna/gene_family/notebooks/dna_seq_families_analysis.ipynb)

See code: [gene_family/scripts](./projects/dna/gene_family/scripts/)

See report: [gene_family_report.pdf](./projects/dna/gene_family/gene_family_report.pdf)


## RNA Projects
🚧 Coming Soon 🚧 

## Protein Projects
🚧 Coming Soon 🚧 

## Lab-in-the-loop Projects
🚧 Coming Soon 🚧 