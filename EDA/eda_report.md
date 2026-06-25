# Exploratory Data Analysis Report

## Dataset Overview

- Rows: 284,807
- Columns: 31
- Target column: `Class` where `1` means fraud and `0` means non-fraud
- Missing values: 0
- Duplicate rows: 1,081

## Class Imbalance

The dataset is extremely imbalanced:

| Class | Count | Share |
| --- | ---: | ---: |
| Non-fraud (0) | 284,315 | 99.8273% |
| Fraud (1) | 492 | 0.1727% |

Implication: accuracy alone is not a useful metric. A model can look strong while failing to identify fraud.

## Amount Patterns

Summary statistics for `Amount` show that fraudulent transactions are not uniformly larger than legitimate ones, but the distribution is shifted and more concentrated at smaller values.

| Class | Mean | Median | 75th percentile | Max |
| --- | ---: | ---: | ---: | ---: |
| Non-fraud (0) | 88.291 | 22.00 | 77.05 | 25,691.16 |
| Fraud (1) | 122.211 | 9.25 | 105.89 | 2,125.87 |

Observed pattern:

- Most fraud cases are small-value transactions.
- Fraud also appears again at the high end of the amount distribution.
- The fraud amount distribution is more concentrated than the non-fraud distribution.

## Time Patterns

`Time` is a transaction sequence timestamp in seconds from the first transaction.

Observed pattern:

- Fraud is not evenly distributed across the transaction timeline.
- Some early and mid-sequence time bins show higher fraud rates than the quietest periods.
- This suggests temporal clustering, so time-aware features may help a downstream model.

## Strongest Fraud-Related Features

The strongest absolute correlations with `Class` are:

| Feature | Absolute correlation |
| --- | ---: |
| V17 | 0.3265 |
| V14 | 0.3025 |
| V12 | 0.2606 |
| V10 | 0.2169 |
| V16 | 0.1965 |
| V3 | 0.1930 |
| V7 | 0.1873 |
| V11 | 0.1549 |
| V4 | 0.1334 |
| V18 | 0.1115 |

Interpretation:

- Fraud is most strongly associated with a small subset of PCA-transformed variables.
- `V17`, `V14`, `V12`, and `V10` are especially informative.
- These features should be prioritized in model interpretation and feature importance review.

## Fraud vs Non-Fraud Signal Differences

For the highest-signal features, fraud transactions have much lower average values than legitimate ones.

| Feature | Non-fraud mean | Fraud mean |
| --- | ---: | ---: |
| V17 | 0.0115 | -6.6658 |
| V14 | 0.0121 | -6.9717 |
| V12 | 0.0108 | -6.2594 |
| V10 | 0.0098 | -5.6769 |
| V16 | 0.0072 | -4.1399 |

This is a strong sign that the fraud class occupies a distinct region in transformed feature space rather than being a simple amount-based anomaly.

## Key Business Insights

- The dataset is highly imbalanced, so recall, precision, PR-AUC, and cost-sensitive evaluation are more meaningful than accuracy.
- Fraud is not defined by transaction amount alone; the PCA-derived variables carry most of the separation power.
- Time-based clustering suggests fraud may come in bursts rather than being uniformly random.
- Small-value transactions are common in fraud, which means low-value payments should not be assumed safe.
- A production model should balance false positives carefully because legitimate transactions dominate the data.


