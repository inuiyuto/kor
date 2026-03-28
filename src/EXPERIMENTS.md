# Experiment Details

Hyperparameter search ranges and best configurations for KOR experiments.
Internal dynamics hyperparameters (γ, ε, Δt, etc.) follow [Ceni et al. 2024](https://github.com/AndreaCossu/RandomizedCoupledOscillators) and are kept fixed.

## Search Ranges

### RON

Only the RFF frequency scale σ_ω is tuned.

| Hyperparameter | Search range |
|---|---|
| σ_ω | {0.1, 1, 10} |

### coRNN / hcoRNN

| Hyperparameter | Search range |
|---|---|
| σ_ω | {0.1, 1, 10} |
| batch size B | {30, 120} |
| learning rate η | {0.0021, 0.0084} |

Note: For sMNIST, psMNIST, and npCIFAR-10, only σ_ω was tuned due to computational cost; B and η were fixed to the best values from Ceni et al. 2024.

## Best Configurations

Best hyperparameters selected by validation accuracy.

### RON

RON is a reservoir model with no trainable parameters, so only KOR has a tuned hyperparameter (σ_ω).

| Dataset | KOR (σ_ω) |
|---|---|
| sMNIST       | 0.1 |
| psMNIST      | 1   |
| npCIFAR-10   | 0.1 |
| FordA        | 10  |
| Adiac        | 10  |
| uWaveGesture | 0.1 |

### coRNN

Entries for last/mean report (B, η). Entries for KOR report (σ_ω, B, η).

| Dataset | last (B, η) | mean (B, η) | KOR (σ_ω, B, η) |
|---|---|---|---|
| sMNIST       | (30, 0.0021)   | (30, 0.0021)   | (1, 30, 0.0021)   |
| psMNIST      | (30, 0.0021)   | (30, 0.0021)   | (1, 30, 0.0021)   |
| npCIFAR-10   | (30, 0.0021)   | (30, 0.0021)   | (0.1, 30, 0.0021) |
| FordA        | (120, 0.0021)  | (30, 0.0084)   | (1, 30, 0.0084)   |
| Adiac        | (30, 0.0084)   | (30, 0.0084)   | (1, 30, 0.0084)   |
| uWaveGesture | (120, 0.0084)  | (30, 0.0084)   | (1, 30, 0.0084)   |

### hcoRNN

Entries for last/mean report (B, η). Entries for KOR report (σ_ω, B, η).

| Dataset | last (B, η) | mean (B, η) | KOR (σ_ω, B, η) |
|---|---|---|---|
| sMNIST       | (30, 0.0021)   | (30, 0.0021)   | (1, 30, 0.0021)   |
| psMNIST      | (30, 0.0021)   | (30, 0.0021)   | (0.1, 30, 0.0021) |
| npCIFAR-10   | (30, 0.0021)   | (30, 0.0021)   | (10, 30, 0.0021)  |
| FordA        | (120, 0.0021)  | (30, 0.0021)   | (0.1, 30, 0.0084) |
| Adiac        | (30, 0.0084)   | (30, 0.0084)   | (1, 30, 0.0084)   |
| uWaveGesture | (30, 0.0021)   | (120, 0.0084)  | (0.1, 30, 0.0021) |

## Compute Environment

All experiments were run on two servers:
1. Dual-socket AMD EPYC 7542 with three NVIDIA A100 PCIe GPUs (40 GB each)
2. Intel Core i7-6950X with four NVIDIA GeForce GTX 1080 Ti GPUs (11 GB each)
