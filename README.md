# Kernel Occupation Readout for Oscillatory Recurrent Neural Networks

This repository contains the code for the paper:

**Kernel Occupation Readout for Oscillatory Recurrent Neural Networks**
Yuto Inui, Masahiro Ikeda, Takuya Konishi, Yoshinobu Kawahara
*IEEE International Joint Conference on Neural Networks (IJCNN 2026)*

## Overview

We propose **Kernel Occupation Readout (KOR)**, a readout mechanism for oscillatory RNNs that views the hidden trajectory as inducing an empirical occupation measure and represents it via a kernel mean embedding (KME) in a reproducing kernel Hilbert space (RKHS).

KOR generalizes mean pooling without adding trainable parameters beyond the standard linear classifier. For Gaussian kernels, it is implemented efficiently via random Fourier features (RFFs), making it a drop-in replacement for last-state and mean-pooling readouts.

## Usage

<!-- TODO: add training/evaluation commands -->

Hyperparameter search ranges and best configurations are documented in [`src/`](src/).

## Experiments

Experiments cover six sequence classification benchmarks (sMNIST, psMNIST, npCIFAR-10, FordA, Adiac, uWaveGesture) across three oscillatory RNN architectures: RON, coRNN, and hcoRNN.

Robustness to phase shifts is evaluated on uWaveGesture under random missing observations and tail truncation.

## Citation

```bibtex
<!-- TODO: add bib -->
```

## Acknowledgements

This code is based on [RandomizedCoupledOscillators](https://github.com/AndreaCossu/RandomizedCoupledOscillators) by Andrea Cossu et al., which is licensed under the MIT License. The original copyright notice is preserved in `LICENSE`.

