# Experiment Details

Hyperparameter search ranges and best configurations for KOR experiments.
Internal dynamics hyperparameters (γ, ε, Δt, etc.) follow [Ceni et al. 2024](https://github.com/AndreaCossu/RandomizedCoupledOscillators) and are kept fixed.

## Search Ranges

### RON

Only the RFF frequency scale σ_ω is tuned.

| Hyperparameter | Search range |
|---|---|
| σ_ω | <!-- TODO --> |

### coRNN / hcoRNN

| Hyperparameter | Search range |
|---|---|
| σ_ω | <!-- TODO --> |
| batch size B | <!-- TODO --> |
| learning rate η | <!-- TODO --> |

Note: For sMNIST, psMNIST, and npCIFAR-10, only σ_ω was tuned due to computational cost; B and η were fixed to the best values from Ceni et al. 2024.

## Best Configurations

Best hyperparameters selected by validation accuracy.
RON entries: (σ_ω). coRNN / hcoRNN entries: (σ_ω, B, η).

| Dataset | RON | coRNN | hcoRNN |
|---|---|---|---|
| sMNIST       | (0.1) | (1, 30, 0.0021) | (1, 30, 0.0021) |
| psMNIST      | (1)   | (1, 30, 0.0021) | (0.1, 30, 0.0021) |
| npCIFAR-10   | (0.1) | (0.1, 30, 0.0021) | (10, 30, 0.0021) |
| FordA        | (10)  | (1, 30, 0.0084) | (0.1, 30, 0.0084) |
| Adiac        | (10)  | (1, 30, 0.0084) | (1, 30, 0.0084) |
| uWaveGesture | (0.1) | (1, 30, 0.0084) | (0.1, 30, 0.0021) |
