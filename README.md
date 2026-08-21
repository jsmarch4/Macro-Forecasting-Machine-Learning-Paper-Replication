# Macroeconomic Forecasting and Machine Learning Replication

This repository contains an independent Python replication of the forecasting framework developed in *Macroeconomic Forecasting and Machine Learning* by Domenico Giannone and coauthors.

The project reproduces the paper's recursive quantile forecasting methodology using Python and PyTorch, including the complete recursive forecasting pipeline, model selection procedure, and empirical evaluation described in the paper.

---

## Features

- FRED-MD data preprocessing using the official transformation codes
- One-month-ahead unemployment rate change forecasting target
- Official FRED-MD outlier detection and removal
- Recursive expanding-window forecasting
- Recursive predictor and target standardization using only information available at each forecast origin
- Historical unconditional quantile benchmark
- Linear quantile regression with L2 regularization
- Deep neural network quantile regression
- Validation-based hyperparameter selection over:
  - network architecture
  - hidden layer width
  - Leaky ReLU activation parameter
  - L2 regularization strength
- Recursive warm-start optimization
- Pinball loss evaluation on validation and holdout test samples
- Complexity-index model selection
- Figure 4 replication with predictive interval visualization
- Complexity-performance tables for validation and test samples

---

## Replication Pipeline

Run the scripts in the following order:

```text
01_dataset_construction.py
02_benchmark_quantiles.py
03_pinball_loss.py
04_run_linear_torch.py
05_run_dnn_torch.py
06_summarize_results.py
07_plot_figure4_bands.py
08_complexity_tables.py
09_complexity_table_figures.py
```

---

## Data

The replication uses the monthly FRED-MD macroeconomic database and follows the transformation and preprocessing procedures described in the original paper.

- Official FRED-MD transformation codes
- Official FRED-MD outlier removal procedure
- One-month-ahead unemployment rate change forecasting target

---

## Forecasting Framework

The forecasting procedure follows a recursive expanding-window design.

For each forecast origin:

1. Standardize predictors and the response using only historical information.
2. Train the model using all data available at that date.
3. Produce one-step-ahead quantile forecasts.
4. Expand the training sample and repeat.

Forecasts are evaluated using the pinball loss over both validation and holdout test samples.

---

## Model Selection

Hyperparameters are selected exclusively using the validation sample.

Validation period

```text
1980–1999
```

Test period

```text
2000–2024
```

The search considers:

- network depth
- hidden layer width
- Leaky ReLU activation parameter
- L2 regularization parameter
- model complexity index

The selected model is then evaluated on the holdout test sample without further tuning.

---

## Repository Structure

```text
01_dataset_construction.py      Build replication dataset
02_benchmark_quantiles.py       Historical quantile benchmark
03_pinball_loss.py              Benchmark pinball loss
04_run_linear_torch.py          Linear quantile model
05_run_dnn_torch.py             Deep neural network model
06_summarize_results.py         Aggregate forecasting results
07_plot_figure4_bands.py        Figure 4 replication
08_complexity_tables.py         Complexity-index model selection
09_complexity_table_figures.py  Complexity table visualization

data_utils.py
models.py
train_utils.py
losses.py
```

---

## Requirements

- Python 3.11
- PyTorch
- NumPy
- pandas
- matplotlib

---

## Reference

Giannone, D., Lenza, M., Primiceri, G., and coauthors.

*Macroeconomic Forecasting and Machine Learning.*