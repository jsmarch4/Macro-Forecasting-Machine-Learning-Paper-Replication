# Macroeconomic Forecasting and Machine Learning Replication

This repository contains an independent Python replication of the forecasting framework developed in *Macroeconomic Forecasting and Machine Learning* by Domenico Giannone and coauthors.

The objective of this project is to recreate the paper's empirical forecasting pipeline from scratch while closely matching the published methodology. The current implementation includes:

- FRED-MD preprocessing using the official transformation codes
- Construction of the one-month-ahead unemployment forecasting target
- FRED-MD outlier detection and removal
- Recursive expanding-window forecasting
- Naive historical quantile benchmark
- Linearized deep neural network quantile regression implemented in PyTorch
- Validation-based hyperparameter selection over network architecture and regularization
- Pinball loss evaluation on validation and test samples

The project is being extended toward a complete replication of the paper's forecasting framework, including the nonlinear deep neural network models and a detailed comparison with the published forecasting results.

## Repository Structure

- `data/` – Raw FRED-MD data and processed datasets
- `results/` – Forecasts, benchmark results, and evaluation metrics
- `archive/` – Previous implementations and experimental code

## Replication Pipeline

Run the scripts in the following order:

```text
01_dataset_construction.py
02_benchmark_quantiles.py
03_pinball_loss.py
04_run_linear_torch.py
```
