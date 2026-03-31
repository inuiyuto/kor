import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KernelCenterer, StandardScaler
from sklearn.svm import SVC
from torch import nn, optim
from tqdm import tqdm

from esn import DeepReservoir
from utils import (
    check,
    coESN,
    coESN_RFF,
    coESN_mean,
    coRNN,
    coRNN_RFF,
    coRNN_mean,
    get_Adiac_data,
    get_FordA_data,
    get_cifar_data,
    get_mnist_data,
    get_uwavegesture_data,
    seed_all,
)


OUTPUT_ROOT = Path("outputs")
PAPER_DEFAULTS_PATH = Path(__file__).resolve().parent / "configs" / "paper_defaults.json"

with open(PAPER_DEFAULTS_PATH) as handle:
    DATASET_CONFIGS = json.load(handle)


def ensure_output_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "results": root / "results",
        "logs": root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def append_rows(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_degree_list(value: str) -> list[int]:
    parsed = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed.append(int(item))
    if not parsed:
        raise argparse.ArgumentTypeError("Provide at least one degree.")
    return parsed


def degree_to_label(value: int) -> str:
    return str(int(value))


def select_device(cpu_only: bool) -> torch.device:
    return torch.device("cpu") if cpu_only or not torch.cuda.is_available() else torch.device("cuda")


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    parser.add_argument("--trials", type=int, default=1, help="Number of repeated trials.")
    parser.add_argument("--use_test", action="store_true", help="Evaluate the test split.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    parser.add_argument("--output_root", type=Path, default=OUTPUT_ROOT, help="Root output directory.")
    parser.add_argument("--check", action="store_true", help="Run the coRNN stability check when available.")
    return parser


def add_common_hyperparam_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Training batch size override.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs override.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override.")
    parser.add_argument("--n_hid", type=int, default=None, help="Hidden dimension override.")
    parser.add_argument("--dt", type=float, default=None, help="Time step override.")
    parser.add_argument("--gamma", type=float, default=None, help="Gamma override.")
    parser.add_argument("--epsilon", type=float, default=None, help="Epsilon override.")
    parser.add_argument("--gamma_range", type=float, default=None, help="Gamma range override.")
    parser.add_argument("--epsilon_range", type=float, default=None, help="Epsilon range override.")
    parser.add_argument("--rho", type=float, default=None, help="Reservoir spectral radius override.")
    parser.add_argument("--inp_scaling", type=float, default=None, help="Reservoir input scaling override.")
    parser.add_argument("--leaky", type=float, default=1.0, help="DeepReservoir leaky parameter.")
    parser.add_argument("--std", type=float, default=None, help="KOR/RFF frequency scale override.")
    parser.add_argument("--logreg_max_iter", type=int, default=1000, help="Max iterations for linear probes.")


def complete_params(args: argparse.Namespace, dataset_name: str, model_name: str, readout_name: str) -> dict:
    config = DATASET_CONFIGS[dataset_name]["models"][model_name]
    dynamics = config["dynamics"]
    readout = config["readouts"][readout_name]
    return {
        "batch": args.batch if args.batch is not None else readout["batch"],
        "epochs": args.epochs if args.epochs is not None else config["default_epochs"],
        "lr": args.lr if args.lr is not None else readout.get("lr"),
        "n_hid": args.n_hid if args.n_hid is not None else config["n_hid"],
        "dt": args.dt if args.dt is not None else dynamics["dt"],
        "gamma": args.gamma if args.gamma is not None else dynamics["gamma"],
        "epsilon": args.epsilon if args.epsilon is not None else dynamics["epsilon"],
        "gamma_range": args.gamma_range if args.gamma_range is not None else dynamics["gamma_range"],
        "epsilon_range": args.epsilon_range if args.epsilon_range is not None else dynamics["epsilon_range"],
        "rho": args.rho if args.rho is not None else dynamics["rho"],
        "inp_scaling": args.inp_scaling if args.inp_scaling is not None else dynamics["inp_scaling"],
        "std": args.std if args.std is not None else readout.get("std"),
        "leaky": args.leaky,
    }


def build_gamma_epsilon(params: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    gamma = (
        params["gamma"] - params["gamma_range"] / 2.0,
        params["gamma"] + params["gamma_range"] / 2.0,
    )
    epsilon = (
        params["epsilon"] - params["epsilon_range"] / 2.0,
        params["epsilon"] + params["epsilon_range"] / 2.0,
    )
    return gamma, epsilon


def load_dataset(dataset_name: str, batch_size: int, bs_test: int, trial_seed: int, reservoir: bool):
    loader_name = DATASET_CONFIGS[dataset_name]["loader"]
    if loader_name in {"mnist", "psmnist"}:
        return get_mnist_data(batch_size, bs_test, seed=trial_seed)
    if loader_name == "cifar":
        return get_cifar_data(batch_size, bs_test, seed=trial_seed)
    if loader_name == "forda":
        return get_FordA_data(batch_size, bs_test, RC=reservoir, seed=trial_seed)
    if loader_name == "adiac":
        return get_Adiac_data(batch_size, bs_test, RC=reservoir, seed=trial_seed)
    if loader_name == "uwave":
        return get_uwavegesture_data(batch_size, bs_test, seed=trial_seed)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def prepare_inputs(
    dataset_name: str,
    inputs: torch.Tensor,
    device: torch.device,
    permutation: Optional[torch.Tensor],
    cifar_noise: Optional[torch.Tensor],
) -> torch.Tensor:
    x = inputs.to(device)
    if dataset_name == "sMNIST":
        x = x.reshape(x.shape[0], 1, 784).permute(0, 2, 1)
    elif dataset_name == "psMNIST":
        x = x.reshape(x.shape[0], 1, 784).permute(0, 2, 1)
        x = x[:, permutation, :]
    elif dataset_name == "npCIFAR-10":
        x = x.permute(0, 2, 1, 3).reshape(x.shape[0], 32, 96)
        x = torch.cat((x, cifar_noise[: x.shape[0]]), dim=1)
    return x


def target_indices(targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim > 1:
        return targets.argmax(dim=1)
    return targets.long()


def evaluate_sequence_model(
    model: torch.nn.Module,
    loader,
    objective: nn.Module,
    dataset_name: str,
    device: torch.device,
    permutation: Optional[torch.Tensor],
    cifar_noise: Optional[torch.Tensor],
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    with torch.no_grad():
        for inputs, targets in loader:
            x = prepare_inputs(dataset_name, inputs, device, permutation, cifar_noise)
            y = targets.to(device)
            logits = model(x)
            loss = objective(logits, y)
            total_loss += float(loss.item()) * x.shape[0]
            pred = logits.argmax(dim=1)
            correct = pred.eq(target_indices(y)).sum().item()
            total_correct += correct
            total_items += x.shape[0]
    return total_loss / max(total_items, 1), 100.0 * total_correct / max(total_items, 1)


@torch.no_grad()
def collect_probe_features(
    model: torch.nn.Module,
    loader,
    dataset_name: str,
    device: torch.device,
    permutation: Optional[torch.Tensor],
    cifar_noise: Optional[torch.Tensor],
    kor_readout: bool,
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for inputs, targets in tqdm(loader, leave=False):
        x = prepare_inputs(dataset_name, inputs, device, permutation, cifar_noise)
        if kor_readout:
            feats = model(x)[0]
        else:
            feats = model(x)[-1][0]
        features.append(feats.cpu())
        labels.append(target_indices(targets))
    return torch.cat(features, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


def fit_linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    max_iter: int,
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(train_features)
    classifier = LogisticRegression(max_iter=max_iter, penalty=None)
    classifier.fit(scaler.transform(train_features), train_labels)
    return scaler, classifier


def score_linear_probe(
    model: torch.nn.Module,
    loader,
    dataset_name: str,
    device: torch.device,
    permutation: Optional[torch.Tensor],
    cifar_noise: Optional[torch.Tensor],
    kor_readout: bool,
    scaler: StandardScaler,
    classifier: LogisticRegression,
) -> float:
    features, labels = collect_probe_features(
        model, loader, dataset_name, device, permutation, cifar_noise, kor_readout
    )
    return 100.0 * classifier.score(scaler.transform(features), labels)


def build_classification_model(
    dataset_name: str,
    model_name: str,
    readout: str,
    params: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, bool]:
    config = DATASET_CONFIGS[dataset_name]
    gamma, epsilon = build_gamma_epsilon(params)
    no_friction = model_name == "hcornn"
    if model_name == "ron":
        if readout == "kor":
            model = coESN_RFF(
                config["n_inp"],
                params["n_hid"],
                params["dt"],
                gamma,
                epsilon,
                params["rho"],
                params["inp_scaling"],
                device=device,
                std=params["std"],
            ).to(device)
        elif readout == "mean":
            model = coESN_mean(
                config["n_inp"],
                params["n_hid"],
                params["dt"],
                gamma,
                epsilon,
                params["rho"],
                params["inp_scaling"],
                device=device,
            ).to(device)
        else:
            model = coESN(
                config["n_inp"],
                params["n_hid"],
                params["dt"],
                gamma,
                epsilon,
                params["rho"],
                params["inp_scaling"],
                device=device,
            ).to(device)
        return model, True

    if readout == "kor":
        model = coRNN_RFF(
            config["n_inp"],
            params["n_hid"],
            config["n_out"],
            params["dt"],
            gamma,
            epsilon,
            device=device,
            no_friction=no_friction,
            std=params["std"],
        ).to(device)
    elif readout == "mean":
        model = coRNN_mean(
            config["n_inp"],
            params["n_hid"],
            config["n_out"],
            params["dt"],
            gamma,
            epsilon,
            device=device,
            no_friction=no_friction,
        ).to(device)
    else:
        model = coRNN(
            config["n_inp"],
            params["n_hid"],
            config["n_out"],
            params["dt"],
            gamma,
            epsilon,
            device=device,
            no_friction=no_friction,
        ).to(device)
    return model, False


def maybe_run_check(args: argparse.Namespace, model: torch.nn.Module) -> None:
    if args.check and hasattr(model, "h2h") and hasattr(model, "dt"):
        if not check(model):
            raise ValueError("Stability check failed.")


def write_classification_rows(
    output_root: Path,
    dataset_name: str,
    model_name: str,
    readout: str,
    seed: int,
    metrics: dict[str, Optional[float]],
) -> None:
    rows = []
    for split, value in metrics.items():
        if value is None:
            continue
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "readout": readout,
                "seed": seed,
                "split": split,
                "metric": "accuracy",
                "value": f"{value:.6f}",
            }
        )
    append_rows(
        output_root / "results" / "table1_classification.csv",
        ["dataset", "model", "readout", "seed", "split", "metric", "value"],
        rows,
    )


def run_classification(args: argparse.Namespace) -> int:
    config = DATASET_CONFIGS[args.dataset]
    params = complete_params(args, args.dataset, args.model, args.readout)
    output_paths = ensure_output_dirs(args.output_root)
    device = select_device(args.cpu)

    summary_lines = []
    for trial_index in range(args.trials):
        trial_seed = args.seed + trial_index
        seed_all(trial_seed)
        permutation = None
        if args.dataset == "psMNIST":
            permutation = torch.randperm(784, device=device)

        model, is_reservoir = build_classification_model(args.dataset, args.model, args.readout, params, device)
        maybe_run_check(args, model)
        train_loader, valid_loader, test_loader = load_dataset(
            args.dataset, params["batch"], config["bs_test"], trial_seed, is_reservoir
        )
        cifar_noise = None
        if args.dataset == "npCIFAR-10":
            base = torch.randn((1, 1000 - 32, 96), device=device)
            cifar_noise = base.repeat(params["batch"], 1, 1)

        if is_reservoir:
            train_features, train_labels = collect_probe_features(
                model,
                train_loader,
                args.dataset,
                device,
                permutation,
                cifar_noise,
                args.readout == "kor",
            )
            scaler, classifier = fit_linear_probe(train_features, train_labels, args.logreg_max_iter)
            train_acc = 100.0 * classifier.score(scaler.transform(train_features), train_labels)
            valid_acc = score_linear_probe(
                model,
                valid_loader,
                args.dataset,
                device,
                permutation,
                cifar_noise,
                args.readout == "kor",
                scaler,
                classifier,
            )
            test_acc = None
            if args.use_test:
                test_acc = score_linear_probe(
                    model,
                    test_loader,
                    args.dataset,
                    device,
                    permutation,
                    cifar_noise,
                    args.readout == "kor",
                    scaler,
                    classifier,
                )
        else:
            objective = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=params["lr"])
            best_valid = float("-inf")
            best_metrics = {"train": None, "valid": None, "test": None}
            for _epoch in range(params["epochs"]):
                model.train()
                total_correct = 0
                total_items = 0
                for inputs, targets in tqdm(train_loader, leave=False):
                    x = prepare_inputs(args.dataset, inputs, device, permutation, cifar_noise)
                    y = targets.to(device)
                    optimizer.zero_grad()
                    logits = model(x)
                    loss = objective(logits, y)
                    loss.backward()
                    optimizer.step()
                    total_correct += logits.argmax(dim=1).eq(target_indices(y)).sum().item()
                    total_items += x.shape[0]
                train_acc = 100.0 * total_correct / max(total_items, 1)
                _valid_loss, valid_acc = evaluate_sequence_model(
                    model, valid_loader, objective, args.dataset, device, permutation, cifar_noise
                )
                test_acc = None
                if args.use_test:
                    _test_loss, test_acc = evaluate_sequence_model(
                        model, test_loader, objective, args.dataset, device, permutation, cifar_noise
                    )
                if valid_acc >= best_valid:
                    best_valid = valid_acc
                    best_metrics = {"train": train_acc, "valid": valid_acc, "test": test_acc}
            train_acc = best_metrics["train"]
            valid_acc = best_metrics["valid"]
            test_acc = best_metrics["test"]

        write_classification_rows(
            output_paths["root"],
            args.dataset,
            args.model,
            args.readout,
            trial_seed,
            {"train": train_acc, "valid": valid_acc, "test": test_acc},
        )
        summary_lines.append(
            f"seed={trial_seed} dataset={args.dataset} model={args.model} readout={args.readout} "
            f"train={train_acc:.4f} valid={valid_acc:.4f} test={'N/A' if test_acc is None else f'{test_acc:.4f}'}"
        )

    log_path = output_paths["logs"] / "table1_runs.log"
    with open(log_path, "a") as handle:
        for line in summary_lines:
            handle.write(line + "\n")
    for line in summary_lines:
        print(line)
    return 0


def fit_ron_readout_probes(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[
    tuple[StandardScaler, LogisticRegression],
    tuple[StandardScaler, LogisticRegression],
    tuple[StandardScaler, LogisticRegression],
]:
    nok_features = []
    last_states = []
    mean_states = []
    labels = []
    for inputs, targets in tqdm(loader, leave=False):
        x = inputs.to(device)
        mu, last, state_mean = model(x, return_state_mean=True)
        nok_features.append(mu.cpu())
        last_states.append(last[0].cpu())
        mean_states.append(state_mean.cpu())
        labels.append(target_indices(targets))
    y = torch.cat(labels, dim=0).numpy()
    return (
        fit_linear_probe(torch.cat(last_states, dim=0).numpy(), y, 10000),
        fit_linear_probe(torch.cat(mean_states, dim=0).numpy(), y, 10000),
        fit_linear_probe(torch.cat(nok_features, dim=0).numpy(), y, 10000),
    )


def thin_time_steps(x: torch.Tensor, retention_factor: float) -> torch.Tensor:
    if abs(retention_factor - 1.0) < 1e-8:
        return x
    batch, length, channels = x.shape
    target_len = max(1, min(length, int(math.floor(retention_factor * length))))
    rand_scores = torch.rand(batch, length, device=x.device)
    indices = torch.topk(rand_scores, k=target_len, dim=1, largest=True, sorted=False).indices
    indices = torch.sort(indices, dim=1)[0]
    expanded = indices.unsqueeze(-1).expand(-1, -1, channels)
    return torch.gather(x, 1, expanded)


def truncate_time_steps(x: torch.Tensor, retention_factor: float) -> torch.Tensor:
    if abs(retention_factor - 1.0) < 1e-8:
        return x
    target_len = max(1, min(x.shape[1], int(math.floor(retention_factor * x.shape[1]))))
    return x[:, :target_len, :]


@torch.no_grad()
def collect_ron_readout_features(
    loader,
    model: torch.nn.Module,
    device: torch.device,
    transform: Callable[[torch.Tensor, float], torch.Tensor],
    retention_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    last_states = []
    mean_states = []
    nok_features = []
    labels = []
    for inputs, targets in tqdm(loader, leave=False):
        x = transform(inputs.to(device), retention_factor)
        mu, last, state_mean = model(x, return_state_mean=True)
        last_states.append(last[0].cpu())
        mean_states.append(state_mean.cpu())
        nok_features.append(mu.cpu())
        labels.append(target_indices(targets))
    return (
        torch.cat(last_states, dim=0).numpy(),
        torch.cat(mean_states, dim=0).numpy(),
        torch.cat(nok_features, dim=0).numpy(),
        torch.cat(labels, dim=0).numpy(),
    )


def score_probe(features: np.ndarray, labels: np.ndarray, probe: tuple[StandardScaler, LogisticRegression]) -> float:
    scaler, classifier = probe
    return 100.0 * classifier.score(scaler.transform(features), labels)


def retention_values(start: float, end: float, step: float) -> np.ndarray:
    direction = -1 if start > end else 1
    values = np.arange(start, end + direction * 1e-9, direction * abs(step))
    return np.round(values, 4)


def write_robustness_rows(
    output_root: Path,
    corruption_type: str,
    seed: int,
    rows: list[dict],
) -> None:
    for row in rows:
        row["corruption_type"] = corruption_type
        row["seed"] = seed
        row["metric"] = "accuracy"
    append_rows(
        output_root / "results" / "fig2_robustness.csv",
        ["dataset", "model", "readout", "corruption_type", "observed_fraction", "seed", "metric", "value"],
        rows,
    )


def run_robustness(args: argparse.Namespace) -> int:
    if args.model != "ron":
        raise ValueError("Fig. 2 is restricted to RON.")
    output_paths = ensure_output_dirs(args.output_root)
    params = complete_params(args, "uWaveGesture", "ron", "kor")
    device = select_device(args.cpu)
    gamma, epsilon = build_gamma_epsilon(params)

    summary_lines = []
    for trial_index in range(args.trials):
        trial_seed = args.seed + trial_index
        seed_all(trial_seed)
        model = coESN_RFF(
            1,
            params["n_hid"],
            params["dt"],
            gamma,
            epsilon,
            params["rho"],
            params["inp_scaling"],
            device=device,
            std=params["std"],
        ).to(device)
        maybe_run_check(args, model)
        train_loader, _valid_loader, test_loader = get_uwavegesture_data(
            params["batch"], DATASET_CONFIGS["uWaveGesture"]["bs_test"], seed=trial_seed
        )
        last_probe, mean_probe, kor_probe = fit_ron_readout_probes(model, train_loader, device)

        if args.mode in {"missing", "both"}:
            missing_rows = []
            for retention in retention_values(args.missing_start, args.missing_end, args.missing_step):
                last_feats, mean_feats, nok_feats, labels = collect_ron_readout_features(
                    test_loader, model, device, thin_time_steps, float(retention)
                )
                missing_rows.extend(
                    [
                        {
                            "dataset": "uWaveGesture",
                            "model": "ron",
                            "readout": "last",
                            "observed_fraction": f"{retention:.4f}",
                            "value": f"{score_probe(last_feats, labels, last_probe):.6f}",
                        },
                        {
                            "dataset": "uWaveGesture",
                            "model": "ron",
                            "readout": "mean",
                            "observed_fraction": f"{retention:.4f}",
                            "value": f"{score_probe(mean_feats, labels, mean_probe):.6f}",
                        },
                        {
                            "dataset": "uWaveGesture",
                            "model": "ron",
                            "readout": "kor",
                            "observed_fraction": f"{retention:.4f}",
                            "value": f"{score_probe(nok_feats, labels, kor_probe):.6f}",
                        },
                    ]
                )
            write_robustness_rows(output_paths["root"], "missing", trial_seed, missing_rows)
            summary_lines.append(
                f"seed={trial_seed} fig2 missing points={len(missing_rows) // 3}"
            )

        if args.mode in {"truncate", "both"}:
            truncate_rows = []
            for retention in retention_values(args.truncate_start, args.truncate_end, args.truncate_step):
                last_feats, mean_feats, nok_feats, labels = collect_ron_readout_features(
                    test_loader, model, device, truncate_time_steps, float(retention)
                )
                truncate_rows.extend(
                    [
                        {
                            "dataset": "uWaveGesture",
                            "model": "ron",
                            "readout": "last",
                            "observed_fraction": f"{retention:.4f}",
                            "value": f"{score_probe(last_feats, labels, last_probe):.6f}",
                        },
                        {
                            "dataset": "uWaveGesture",
                            "model": "ron",
                            "readout": "mean",
                            "observed_fraction": f"{retention:.4f}",
                            "value": f"{score_probe(mean_feats, labels, mean_probe):.6f}",
                        },
                        {
                            "dataset": "uWaveGesture",
                            "model": "ron",
                            "readout": "kor",
                            "observed_fraction": f"{retention:.4f}",
                            "value": f"{score_probe(nok_feats, labels, kor_probe):.6f}",
                        },
                    ]
                )
            write_robustness_rows(output_paths["root"], "truncate", trial_seed, truncate_rows)
            summary_lines.append(
                f"seed={trial_seed} fig2 truncate points={len(truncate_rows) // 3}"
            )

    with open(output_paths["logs"] / "fig2_runs.log", "a") as handle:
        for line in summary_lines:
            handle.write(line + "\n")
    for line in summary_lines:
        print(line)
    return 0


def compute_kernel_matrix(X: np.ndarray, Y: np.ndarray, degree: int) -> np.ndarray:
    batch_x, len_x, hidden_x = X.shape
    batch_y, len_y, hidden_y = Y.shape
    if hidden_x != hidden_y:
        raise ValueError("Hidden dimensions must match.")
    c = 1.0

    if degree == 1:
        return c + X.mean(axis=1) @ Y.mean(axis=1).T

    if degree == 2:
        mX = X.mean(axis=1)
        mY = Y.mean(axis=1)
        M2X = np.matmul(X.transpose(0, 2, 1), X) / len_x
        M2Y = np.matmul(Y.transpose(0, 2, 1), Y) / len_y
        return (c**2) + 2.0 * c * (mX @ mY.T) + M2X.reshape(batch_x, -1) @ M2Y.reshape(batch_y, -1).T

    G = np.empty((batch_x, batch_y), dtype=np.float64)
    if batch_x <= batch_y:
        Y_t = np.transpose(Y, (0, 2, 1))
        for idx in range(batch_x):
            dot = np.matmul(X[idx], Y_t)
            G[idx, :] = np.mean((c + dot) ** degree, axis=(1, 2))
    else:
        for idx in range(batch_y):
            dot = np.matmul(X, Y[idx].T)
            G[:, idx] = np.mean((c + dot) ** degree, axis=(1, 2))
    return G


@torch.no_grad()
def collect_hidden_trajectories(loader, model: torch.nn.Module, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    trajectories = []
    labels = []
    for inputs, targets in tqdm(loader, leave=False):
        x = inputs.to(device)
        trajectory = model(x)[0]
        trajectories.append(trajectory.cpu())
        labels.append(target_indices(targets))
    return torch.cat(trajectories, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


def write_polynomial_rows(output_root: Path, rows: list[dict]) -> None:
    append_rows(
        output_root / "results" / "fig3_polynomial.csv",
        ["dataset", "model", "kernel", "degree", "seed", "metric", "value"],
        rows,
    )


def run_polynomial(args: argparse.Namespace) -> int:
    output_paths = ensure_output_dirs(args.output_root)
    params = complete_params(args, "uWaveGesture", "ron", "kor")
    device = select_device(args.cpu)
    gamma, epsilon = build_gamma_epsilon(params)
    degrees = parse_degree_list(args.degrees)

    summary_lines = []
    for trial_index in range(args.trials):
        trial_seed = args.seed + trial_index
        seed_all(trial_seed)
        model = coESN(
            1,
            params["n_hid"],
            params["dt"],
            gamma,
            epsilon,
            params["rho"],
            params["inp_scaling"],
            device=device,
        ).to(device)
        maybe_run_check(args, model)
        train_loader, _valid_loader, test_loader = get_uwavegesture_data(
            params["batch"], DATASET_CONFIGS["uWaveGesture"]["bs_test"], seed=trial_seed
        )
        train_trajectories, train_labels = collect_hidden_trajectories(train_loader, model, device)
        test_trajectories, test_labels = collect_hidden_trajectories(test_loader, model, device)
        rows = []
        for degree in degrees:
            K_train = compute_kernel_matrix(train_trajectories, train_trajectories, degree)
            centerer = KernelCenterer().fit(K_train)
            K_train_centered = centerer.transform(K_train)
            classifier = SVC(kernel="precomputed").fit(K_train_centered, train_labels)
            K_test = compute_kernel_matrix(test_trajectories, train_trajectories, degree)
            accuracy = 100.0 * classifier.score(centerer.transform(K_test), test_labels)
            rows.append(
                {
                    "dataset": "uWaveGesture",
                    "model": "ron",
                    "kernel": "polynomial",
                    "degree": degree_to_label(degree),
                    "seed": trial_seed,
                    "metric": "accuracy",
                    "value": f"{accuracy:.6f}",
                }
            )
        write_polynomial_rows(output_paths["root"], rows)
        summary_lines.append(f"seed={trial_seed} fig3 degrees={','.join(row['degree'] for row in rows)}")

    with open(output_paths["logs"] / "fig3_runs.log", "a") as handle:
        for line in summary_lines:
            handle.write(line + "\n")
    for line in summary_lines:
        print(line)
    return 0


def main_table1(default_dataset: Optional[str] = None) -> int:
    parser = build_parser("Reproduce Table I classification results.")
    if default_dataset is None:
        parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), required=True)
    else:
        parser.add_argument("--dataset", choices=[default_dataset], default=default_dataset)
    parser.add_argument("--model", choices=["ron", "cornn", "hcornn"], required=True)
    parser.add_argument("--readout", choices=["last", "mean", "kor"], required=True)
    add_common_hyperparam_arguments(parser)
    args = parser.parse_args()
    return run_classification(args)


def main_fig2(default_mode: Optional[str] = None) -> int:
    parser = build_parser("Reproduce Fig. 2 robustness results on uWaveGesture.")
    parser.add_argument("--model", choices=["ron"], default="ron")
    if default_mode is None:
        parser.add_argument("--mode", choices=["missing", "truncate", "both"], default="both")
    else:
        parser.add_argument("--mode", choices=[default_mode], default=default_mode)
    parser.add_argument("--missing_start", type=float, default=1.0)
    parser.add_argument("--missing_end", type=float, default=0.9)
    parser.add_argument("--missing_step", type=float, default=0.01)
    parser.add_argument("--truncate_start", type=float, default=1.0)
    parser.add_argument("--truncate_end", type=float, default=0.9)
    parser.add_argument("--truncate_step", type=float, default=0.01)
    add_common_hyperparam_arguments(parser)
    args = parser.parse_args()
    return run_robustness(args)


def main_fig3() -> int:
    parser = build_parser("Reproduce Fig. 3 polynomial-kernel ablation on uWaveGesture.")
    parser.add_argument("--degrees", default="1,2,3,4", help="Comma-separated polynomial degrees.")
    add_common_hyperparam_arguments(parser)
    args = parser.parse_args()
    return run_polynomial(args)
