# Experiment Details

This document summarizes the hyperparameter search ranges and selected configurations used for the KOR paper experiments.

Internal dynamics hyperparameters for `RON`, `coRNN`, and `hcoRNN` follow the settings from [RandomizedCoupledOscillators](https://github.com/AndreaCossu/RandomizedCoupledOscillators). The public reproduction defaults used by this repository are stored in [`configs/paper_defaults.json`](configs/paper_defaults.json).

## Search Ranges

### RON

Only the RFF frequency scale `sigma_omega` is tuned.


| Hyperparameter | Search range   |
| -------------- | -------------- |
| `sigma_omega`  | `{0.1, 1, 10}` |


### coRNN / hcoRNN


| Hyperparameter      | Search range       |
| ------------------- | ------------------ |
| `sigma_omega`       | `{0.1, 1, 10}`     |
| batch size `B`      | `{30, 120}`        |
| learning rate `eta` | `{0.0021, 0.0084}` |


For `sMNIST`, `psMNIST`, and `npCIFAR-10`, only `sigma_omega` was tuned for KOR due to computational cost; `B` and `eta` were fixed to the best values from the original oscillator experiments.

## Best Configurations

Best hyperparameters were selected by validation accuracy.

### RON

RON is a reservoir model with no trainable recurrent parameters, so only KOR tunes `sigma_omega`.


| Dataset        | KOR (`sigma_omega`) |
| -------------- | ------------------- |
| `sMNIST`       | `0.1`               |
| `psMNIST`      | `1`                 |
| `npCIFAR-10`   | `0.1`               |
| `FordA`        | `10`                |
| `Adiac`        | `10`                |
| `uWaveGesture` | `0.1`               |


### coRNN

Entries for `last` and `mean` report `(B, eta)`. Entries for `KOR` report `(sigma_omega, B, eta)`.


| Dataset        | `last` `(B, eta)` | `mean` `(B, eta)` | `KOR` `(sigma_omega, B, eta)` |
| -------------- | ----------------- | ----------------- | ----------------------------- |
| `sMNIST`       | `(30, 0.0021)`    | `(30, 0.0021)`    | `(1, 30, 0.0021)`             |
| `psMNIST`      | `(30, 0.0021)`    | `(30, 0.0021)`    | `(1, 30, 0.0021)`             |
| `npCIFAR-10`   | `(30, 0.0021)`    | `(30, 0.0021)`    | `(0.1, 30, 0.0021)`           |
| `FordA`        | `(120, 0.0021)`   | `(30, 0.0084)`    | `(1, 30, 0.0084)`             |
| `Adiac`        | `(30, 0.0084)`    | `(30, 0.0084)`    | `(1, 30, 0.0084)`             |
| `uWaveGesture` | `(120, 0.0084)`   | `(30, 0.0084)`    | `(1, 30, 0.0084)`             |


### hcoRNN

Entries for `last` and `mean` report `(B, eta)`. Entries for `KOR` report `(sigma_omega, B, eta)`.


| Dataset        | `last` `(B, eta)` | `mean` `(B, eta)` | `KOR` `(sigma_omega, B, eta)` |
| -------------- | ----------------- | ----------------- | ----------------------------- |
| `sMNIST`       | `(30, 0.0021)`    | `(30, 0.0021)`    | `(1, 30, 0.0021)`             |
| `psMNIST`      | `(30, 0.0021)`    | `(30, 0.0021)`    | `(0.1, 30, 0.0021)`           |
| `npCIFAR-10`   | `(30, 0.0021)`    | `(30, 0.0021)`    | `(10, 30, 0.0021)`            |
| `FordA`        | `(120, 0.0021)`   | `(30, 0.0021)`    | `(0.1, 30, 0.0084)`           |
| `Adiac`        | `(30, 0.0084)`    | `(30, 0.0084)`    | `(1, 30, 0.0084)`             |
| `uWaveGesture` | `(30, 0.0021)`    | `(120, 0.0084)`   | `(0.1, 30, 0.0021)`           |


## Compute Environment

All experiments were run on the following servers:

1. Dual-socket AMD EPYC 7542 with three NVIDIA A100 PCIe GPUs (40 GB each)
2. Intel Core i7-6950X with four NVIDIA GeForce GTX 1080 Ti GPUs (11 GB each)
