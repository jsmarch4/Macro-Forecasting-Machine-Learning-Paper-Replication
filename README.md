# Macroeconomic Forecasting and Machine Learning Replication

This repository contains a Python replication of the forecasting framework presented in *Macroeconomic Forecasting and Machine Learning* by Domenico Giannone and coauthors.

The project reproduces the paper's empirical pipeline, including:

* FRED-MD data preprocessing using the official transformation codes
* Construction of the one-month-ahead unemployment forecasting target
* Naive historical quantile benchmark
* Ridge-regularized linear quantile regression implemented in PyTorch
* Recursive expanding-window forecasting with validation-based hyperparameter selection
* Pinball loss evaluation

The long-term goal is to replicate the paper's full forecasting framework, including the deep neural network models, and compare the reproduced results with those reported in the paper.

## Repository Structure

* `data/` – Raw FRED-MD data and processed datasets
* `results/` – Forecasts and evaluation results
* `archive/` – Previous implementations and experimental code

Run the scripts in the following order:

```text
01_dataset_construction.py
02_benchmark_quantiles.py
03_pinball_loss.py
04_run_linear_torch.py
```
