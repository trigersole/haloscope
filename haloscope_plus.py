"""Opt-in HaloScope++ training on activations produced by the released code.

This module deliberately consumes the official split, embeddings, and BLEURT labels.
It changes only subspace scoring, pseudo-label selection, and probe training.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PlusConfig:
    score_mode: str
    probe_backend: str
    tail_fractions: tuple[float, ...]
    layers: tuple[int, ...]
    probe_layers: tuple[int, ...]
    hidden_dim: int
    dropout: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    probe_repeats: int
    validation_folds: int
    stability_penalty: float
    seed: int = 41


def parse_number_list(value, cast=float):
    values = tuple(cast(item.strip()) for item in value.split(',') if item.strip())
    if not values:
        raise ValueError('expected a non-empty comma-separated list')
    return values


def artifact_suffix(run_name):
    """Return a safe optional suffix for independent resumable experiments."""
    if run_name in (None, '', 'default'):
        return ''
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', run_name) is None:
        raise ValueError(
            'plus_run_name must contain only letters, numbers, dot, underscore, or dash'
        )
    return f'_{run_name}'


def roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(labels) != len(scores) or set(np.unique(labels)) != {0, 1}:
        raise ValueError('AUROC requires matching arrays containing both classes')
    order = np.argsort(scores, kind='mergesort')
    ordered_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    begin = 0
    while begin < len(scores):
        end = begin + 1
        while end < len(scores) and ordered_scores[end] == ordered_scores[begin]:
            end += 1
        ranks[order[begin:end]] = (begin + 1 + end) / 2.0
        begin = end
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = len(labels) - n_positive
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def fold_stable_score(labels, scores, folds=5, penalty=0.25, seed=41):
    pooled = roc_auc(labels, scores)
    if folds <= 1:
        return pooled, pooled, 0.0
    labels = np.asarray(labels, dtype=np.int64)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    folds = min(folds, len(positive), len(negative))
    rng = np.random.default_rng(seed)
    positive = rng.permutation(positive)
    negative = rng.permutation(negative)
    aucs = []
    for positive_fold, negative_fold in zip(
        np.array_split(positive, folds), np.array_split(negative, folds)
    ):
        indices = np.concatenate((positive_fold, negative_fold))
        aucs.append(roc_auc(labels[indices], np.asarray(scores)[indices]))
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    return mean_auc - penalty * std_auc, pooled, std_auc


def subspace_score(embeddings, pca, k, score_mode='equation7'):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if score_mode == 'official':
        projected = embeddings @ pca.components_[:k].T
        return np.abs(np.mean(projected * pca.singular_values_[:k], axis=1))
    if score_mode != 'equation7':
        raise ValueError('score_mode must be official or equation7')
    projected = (embeddings - pca.mean_) @ pca.components_[:k].T
    return np.mean(
        projected ** 2 * pca.singular_values_[:k][None, :], axis=1
    )


def confidence_tail_labels(scores, fraction):
    if not 0.0 < fraction <= 0.5:
        raise ValueError('tail fractions must be in (0, 0.5]')
    scores = np.asarray(scores, dtype=np.float64)
    lower = float(np.quantile(scores, fraction))
    upper = float(np.quantile(scores, 1.0 - fraction))
    low = scores <= lower
    high = scores >= upper
    selected = low | high
    labels = high[selected].astype(np.int64)
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = (np.arange(len(scores)) + 0.5) / len(scores)
    weights = (2.0 * np.abs(ranks - 0.5))[selected].astype(np.float32)
    return selected, labels, weights, lower, upper


def _build_probe(torch, input_dim, backend, hidden_dim, dropout, mean, scale):
    class SmallProbe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                'feature_mean', torch.as_tensor(mean, dtype=torch.float32)
            )
            self.register_buffer(
                'feature_scale', torch.as_tensor(scale, dtype=torch.float32)
            )
            if backend == 'linear':
                self.network = torch.nn.Linear(input_dim, 1)
            else:
                self.network = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_dim, 1),
                )

        def forward(self, features):
            standardized = (features - self.feature_mean) / self.feature_scale
            return self.network(standardized).squeeze(-1)

    return SmallProbe()


def train_probe(features, labels, weights, config, seed):
    import torch

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    model = _build_probe(
        torch,
        features.shape[1],
        config.probe_backend,
        config.hidden_dim,
        config.dropout,
        mean,
        scale,
    ).cuda()
    counts = np.bincount(labels.astype(np.int64), minlength=2).astype(np.float32)
    class_weights = len(labels) / (2.0 * counts)
    weights = weights * class_weights[labels.astype(np.int64)]
    weights /= max(float(weights.mean()), 1e-12)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(labels),
        torch.from_numpy(weights),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs)
    )
    for _ in range(config.epochs):
        model.train()
        for batch_features, batch_labels, batch_weights in loader:
            batch_features = batch_features.cuda(non_blocking=True)
            batch_labels = batch_labels.cuda(non_blocking=True)
            batch_weights = batch_weights.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                model(batch_features), batch_labels, reduction='none'
            )
            loss = (losses * batch_weights).sum() / batch_weights.sum()
            loss.backward()
            optimizer.step()
        scheduler.step()
    model.eval()
    return model


def ensemble_probabilities(models, features):
    import torch

    tensor = torch.as_tensor(features, dtype=torch.float32, device='cuda')
    with torch.inference_mode():
        predictions = [torch.sigmoid(model(tensor)).cpu().numpy() for model in models]
    return np.mean(np.stack(predictions, axis=0), axis=0)


def _atomic_torch_save(value, path):
    import torch

    temporary = str(path) + '.tmp'
    torch.save(value, temporary)
    os.replace(temporary, path)


def _select_subspace(wild, validation, labels, config):
    from sklearn.decomposition import PCA

    best = None
    pca_cache = {}
    max_k = min(10, len(wild), wild.shape[2])
    for layer in config.layers:
        pca_cache[layer] = PCA(n_components=max_k, whiten=False).fit(wild[:, layer, :])
    for k in range(1, max_k + 1):
        for layer in config.layers:
            raw = subspace_score(validation[:, layer, :], pca_cache[layer], k, config.score_mode)
            candidates = [(-1, -raw), (1, raw)]
            for sign, truth_score in candidates:
                selection, pooled, fold_std = fold_stable_score(
                    labels,
                    truth_score,
                    config.validation_folds,
                    config.stability_penalty,
                    config.seed,
                )
                result = (selection, pooled, fold_std, layer, k, sign)
                if best is None or result[0] > best[0]:
                    best = result
    return best, pca_cache[best[3]]


def _config_from_args(args, layer_count):
    layers = tuple(
        layer for layer in parse_number_list(args.plus_layers, int)
        if 0 <= layer < layer_count
    )
    probe_layers = tuple(
        layer for layer in parse_number_list(args.plus_probe_layers, int)
        if 0 <= layer < layer_count
    )
    if not layers or not probe_layers:
        raise ValueError('configured HaloScope++ layers are outside the model layer range')
    config = PlusConfig(
        score_mode=args.plus_score_mode,
        probe_backend=args.plus_probe_backend,
        tail_fractions=parse_number_list(args.plus_tail_fractions, float),
        layers=layers,
        probe_layers=probe_layers,
        hidden_dim=args.plus_hidden_dim,
        dropout=args.plus_dropout,
        epochs=args.plus_epochs,
        batch_size=args.plus_batch_size,
        learning_rate=args.plus_learning_rate,
        weight_decay=args.plus_weight_decay,
        probe_repeats=args.plus_probe_repeats,
        validation_folds=args.plus_validation_folds,
        stability_penalty=args.plus_stability_penalty,
    )
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError('plus_dropout must be in [0, 1)')
    if config.probe_backend not in {'linear', 'mlp'}:
        raise ValueError('plus_probe_backend must be linear or mlp')
    if min(
        config.epochs,
        config.batch_size,
        config.probe_repeats,
        config.validation_folds,
    ) < 1:
        raise ValueError('probe sizes, repeats, epochs, and folds must be positive')
    if config.probe_backend == 'mlp' and config.hidden_dim < 1:
        raise ValueError('plus_hidden_dim must be positive for the MLP probe')
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError('learning rate must be positive and weight decay non-negative')
    if config.stability_penalty < 0:
        raise ValueError('stability penalty must be non-negative')
    for fraction in config.tail_fractions:
        if not 0.0 < fraction <= 0.5:
            raise ValueError('all tail fractions must be in (0, 0.5]')
    return config


def run_haloscope_plus(
    wild,
    validation,
    test,
    validation_labels,
    test_labels,
    args,
):
    import torch
    from metric_utils import get_measures, print_measures

    config = _config_from_args(args, wild.shape[1])
    output_dir = Path('save_for_eval') / f'{args.dataset_name}_hal_det'
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = artifact_suffix(args.plus_run_name)
    checkpoint = output_dir / f'haloscope_plus_search_{args.model_name}{suffix}.pt'
    detector_path = output_dir / f'haloscope_plus_detector_{args.model_name}{suffix}.pt'
    results_path = output_dir / f'haloscope_plus_results_{args.model_name}{suffix}.json'

    best_subspace, pca = _select_subspace(wild, validation, validation_labels, config)
    _, validation_direct_auc, direct_fold_std, subspace_layer, k, sign = best_subspace
    wild_score = sign * subspace_score(
        wild[:, subspace_layer, :], pca, k, config.score_mode
    )
    test_score = sign * subspace_score(
        test[:, subspace_layer, :], pca, k, config.score_mode
    )
    direct_measures = get_measures(
        test_score[test_labels == 1], test_score[test_labels == 0], plot=False
    )
    print_measures(
        direct_measures[0], direct_measures[1], direct_measures[2],
        'haloscope-plus-direct-projection'
    )

    signature = asdict(config)
    completed = {}
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state.get('config') != signature:
            raise RuntimeError(
                f'HaloScope++ checkpoint settings differ: {checkpoint}. '
                'Move the old checkpoint or restore its command arguments.'
            )
        completed = state.get('completed', {})
        print(f'Resuming HaloScope++ search with {len(completed)} candidates complete')

    for fraction in config.tail_fractions:
        selected, pseudo_labels, weights, lower, upper = confidence_tail_labels(
            wild_score, fraction
        )
        for layer in config.probe_layers:
            key = f'{fraction:.8f}:{layer}'
            if key in completed:
                continue
            models = [
                train_probe(
                    wild[selected, layer, :],
                    pseudo_labels,
                    weights,
                    config,
                    config.seed + repeat,
                )
                for repeat in range(config.probe_repeats)
            ]
            probabilities = ensemble_probabilities(models, validation[:, layer, :])
            selection_score, pooled_auc, fold_std = fold_stable_score(
                validation_labels,
                probabilities,
                config.validation_folds,
                config.stability_penalty,
                config.seed,
            )
            completed[key] = {
                'fraction': fraction,
                'layer': layer,
                'selection_score': selection_score,
                'validation_auroc': pooled_auc,
                'fold_std': fold_std,
                'lower_threshold': lower,
                'upper_threshold': upper,
                'pseudo_truthful': int(pseudo_labels.sum()),
                'pseudo_hallucinated': int((1 - pseudo_labels).sum()),
                'pseudo_ignored': int(len(wild) - len(pseudo_labels)),
            }
            _atomic_torch_save(
                {'config': signature, 'completed': completed}, checkpoint
            )
            print(
                f'HaloScope++ fraction={fraction:.2f} layer={layer} '
                f'validation AUROC={pooled_auc:.4f} fold std={fold_std:.4f}'
            )

    best_probe = max(completed.values(), key=lambda value: value['selection_score'])
    fraction = best_probe['fraction']
    probe_layer = best_probe['layer']
    selected, pseudo_labels, weights, lower, upper = confidence_tail_labels(
        wild_score, fraction
    )
    models = [
        train_probe(
            wild[selected, probe_layer, :],
            pseudo_labels,
            weights,
            config,
            config.seed + repeat,
        )
        for repeat in range(config.probe_repeats)
    ]
    test_probabilities = ensemble_probabilities(models, test[:, probe_layer, :])
    measures = get_measures(
        test_probabilities[test_labels == 1],
        test_probabilities[test_labels == 0],
        plot=False,
    )
    print_measures(measures[0], measures[1], measures[2], 'haloscope-plus-probe')
    print('test AUROC: ', measures[0])

    result = {
        'method': 'haloscope_plus',
        'run_name': args.plus_run_name,
        'config': signature,
        'subspace_layer': subspace_layer,
        'probe_layer': probe_layer,
        'n_components': k,
        'truth_score_sign': sign,
        'validation_direct_auroc': validation_direct_auc,
        'direct_fold_std': direct_fold_std,
        'tail_fraction': fraction,
        **best_probe,
        'test_auroc': float(measures[0]),
        'test_aupr': float(measures[1]),
        'test_fpr95': float(measures[2]),
        'direct_test_auroc': float(direct_measures[0]),
    }
    results_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    _atomic_torch_save(
        {
            'config': signature,
            'result': result,
            'pca_mean': pca.mean_,
            'pca_components': pca.components_[:k],
            'pca_singular_values': pca.singular_values_[:k],
            'probe_state_dicts': [model.state_dict() for model in models],
        },
        detector_path,
    )
    print(f'HaloScope++ results saved to {results_path}')
    print(f'HaloScope++ detector saved to {detector_path}')
    return result
