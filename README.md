# Kernel Occupation Readout for Oscillatory Recurrent Neural Networks

This repository contains code for:

**Kernel Occupation Readout for Oscillatory Recurrent Neural Networks**  
Yuto Inui, Masahiro Ikeda, Takuya Konishi, Yoshinobu Kawahara  
*to appear at IJCNN 2026*

## Overview

Kernel Occupation Readout (KOR) treats the hidden trajectory of an oscillatory recurrent model as an empirical occupation measure and represents it with a kernel mean embedding in an RKHS.

This repository supports the experiments reported in the paper:

- Table I: sequence classification on `sMNIST`, `psMNIST`, `npCIFAR-10`, `FordA`, `Adiac`, and `uWaveGesture`
- Fig. 2: phase-shift robustness on `uWaveGesture` under random missing observations and tail truncation
- Fig. 3: polynomial-kernel ablation on `uWaveGesture`

Detailed hyperparameter search ranges and best configurations are documented in [EXPERIMENTS.md](EXPERIMENTS.md).

## Setup

Install the Python dependencies required by the experiment runners.

Verified environment:

```text
python==3.12.3
torch==2.7.0+cu126
torchvision==0.22.0+cu126
numpy==2.2.6
scipy==1.15.2
scikit-learn==1.6.1
aeon==1.3.0
tqdm==4.67.1
```

```bash
pip install torch torchvision numpy scipy scikit-learn aeon tqdm
```

The UCR datasets are downloaded through `aeon`. `MNIST` and `CIFAR-10` are downloaded through `torchvision`.

The full set of defaults used by the public reproduction runners is stored in [`configs/paper_defaults.json`](configs/paper_defaults.json). The file separates oscillator dynamics from readout-specific settings and is aligned with the experiment details summarized in [EXPERIMENTS.md](EXPERIMENTS.md).

## Reproduce Table I

Use `reproduce_table1.py` for the unified Table I runner.

```bash
python reproduce_table1.py --dataset sMNIST --model ron --readout kor --use_test
python reproduce_table1.py --dataset FordA --model cornn --readout last --use_test
python reproduce_table1.py --dataset uWaveGesture --model hcornn --readout mean --use_test
```

Arguments:

- `--dataset`: `sMNIST`, `psMNIST`, `npCIFAR-10`, `FordA`, `Adiac`, `uWaveGesture`
- `--model`: `ron`, `cornn`, `hcornn`
- `--readout`: `last`, `mean`, `kor`
- `--trials`: repeated runs with seeds `seed, seed+1, ...`
- `--use_test`: also evaluate the test split
- `--n_hid`, `--dt`, `--gamma`, `--epsilon`, `--rho`, `--inp_scaling`, `--std`, ...: optional overrides for the paper presets

Dataset-specific entry points are also available and forward into the same runner:

- `python sMNIST_task.py --model ron --readout kor`
- `python psMNIST_task.py --model cornn --readout last`
- `python noisy_cifar10_task.py --model hcornn --readout mean`
- `python test_FordA_task.py --model ron --readout last`
- `python test_Adiac_task.py --model cornn --readout kor`
- `python uwavegesture_task.py --model hcornn --readout mean`

## Reproduce Fig. 2

Run the standardized robustness script:

```bash
python reproduce_fig2.py --mode both --trials 5
```

Or use the dataset-specific wrappers:

```bash
python uwavegesture_missing_task.py --trials 5
python uwavegesture_truncate_task.py --trials 5
```

## Reproduce Fig. 3

Run the polynomial-kernel ablation:

```bash
python reproduce_fig3.py --degrees 1,2,3,4 --trials 5
```

Or use the dataset-specific wrapper:

```bash
python uwavegesture_moments_task.py --degrees 1,2,3,4 --trials 5
```

## Outputs

All reproducibility scripts now write into `outputs/`:

- `outputs/results/table1_classification.csv`
- `outputs/results/fig2_robustness.csv`
- `outputs/results/fig3_polynomial.csv`
- `outputs/logs/`

The CSV files are the canonical outputs for downstream analysis.

## Standardized Result Schemas

Classification results:

```text
dataset,model,readout,seed,split,metric,value
```

Robustness results:

```text
dataset,model,readout,corruption_type,observed_fraction,seed,metric,value
```

Polynomial-ablation results:

```text
dataset,model,kernel,degree,seed,metric,value
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{inui2026kernel,
  title     = {Kernel Occupation Readout for Oscillatory Recurrent Neural Networks},
  author    = {Inui, Yuto and Ikeda, Masahiro and Konishi, Takuya and Kawahara, Yoshinobu},
  booktitle = {IEEE International Joint Conference on Neural Networks (IJCNN)},
  year      = {2026},
  note      = {to appear}
}
```

## Acknowledgements

This code builds on [RandomizedCoupledOscillators](https://github.com/AndreaCossu/RandomizedCoupledOscillators) by Andrea Cossu et al. The original MIT license notice is preserved in [LICENSE](LICENSE).
