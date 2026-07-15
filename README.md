# Macroeconomic Forecasting and Machine Learning Replication

This repository contains an independent Python replication of the forecasting framework developed in *Macroeconomic Forecasting and Machine Learning* by Domenico Giannone and coauthors.

The objective of this project is to reproduce the paper's recursive forecasting pipeline as faithfully as possible using Python and PyTorch. The implementation follows the methodology described in the paper while documenting implementation details that were omitted from the original publication.

Current features include:

- FRED-MD preprocessing using the official transformation codes
- Construction of the one-month-ahead unemployment rate change forecasting target
- Official FRED-MD outlier detection and removal
- Recursive expanding-window estimation
- Recursive standardization using only information available at each forecast origin
- Historical unconditional quantile benchmark
- Linearized neural network quantile regression (Leaky ReLU with α = 1)
- Deep neural network quantile regression
- Validation-based hyperparameter selection over network architecture, activation function, and L2 regularization
- Recursive warm-start training
- Pinball loss evaluation on validation and test samples
- Figure 4 style density forecast visualization with prediction intervals

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
```

---

## Current Implementation

### Data

- Monthly FRED-MD predictor set
- Official FRED-MD transformations
- One-month-ahead unemployment rate change target

### Forecasting

- Recursive expanding-window estimation
- Recursive predictor standardization
- Warm-start optimization

### Model Selection

Validation period:

```
1980–1999
```

Test period:

```
2000–2024
```

Hyperparameters are selected exclusively on the validation sample before evaluation on the holdout test sample.

Current search includes

- network depth
- hidden dimension
- Leaky ReLU activation parameter
- L2 regularization parameter

---

## Current Status

The recursive forecasting pipeline is fully operational.

Current empirical findings are consistent with one of the paper's main conclusions:

- the regularized linear activation network performs similarly to the nonlinear deep neural network.

However, the current implementation does not yet match the published forecasting improvements over the unconditional benchmark. Remaining work focuses on reproducing undocumented optimization details (e.g., learning-rate selection, initialization, and optimization settings) and implementing additional diagnostics and benchmark models discussed in the referee reports.

---

## Future Work

- Full 40-value regularization grid
- Complexity-index replication
- Additional benchmark models
- Quantile crossing diagnostics
- Empirical coverage evaluation
- Calibration analysis
- Forecast interval diagnostics
- Additional macroeconomic targets and forecast horizons
