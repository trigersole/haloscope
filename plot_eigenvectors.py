"""Plot saved HaloScope embeddings on their top two principal directions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from prompt_variants import PROMPT_PRESETS, prompt_artifact_tag
from run_haloscope_plus import official_split


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='llama2_chat_7B')
    parser.add_argument('--dataset_name', default='tqa')
    parser.add_argument('--prompt_name', choices=sorted(PROMPT_PRESETS), default='concise')
    parser.add_argument(
        '--layer', type=int, default=11,
        help='zero-based transformer-block output after dropping hidden_states[0]',
    )
    parser.add_argument(
        '--split', choices=['wild', 'validation', 'test', 'all'], default='wild',
        help='examples shown in the scatter plot',
    )
    parser.add_argument(
        '--fit_split', choices=['wild', 'plot'], default='wild',
        help='examples used to fit the two-component PCA',
    )
    parser.add_argument('--truth_threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--wild_ratio', type=float, default=0.75)
    parser.add_argument('--point_size', type=float, default=24.0)
    parser.add_argument('--alpha', type=float, default=0.72)
    parser.add_argument('--output', type=Path, default=None)
    return parser


def split_indices(length, split, wild_ratio, seed):
    wild, validation, test = official_split(length, wild_ratio, seed)
    choices = {
        'wild': wild,
        'validation': validation,
        'test': test,
        'all': np.arange(length, dtype=np.int64),
    }
    return choices[split], wild


def main():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    args = build_parser().parse_args()
    prompt_tag = prompt_artifact_tag(args.prompt_name)
    root = Path('save_for_eval') / f'{args.dataset_name}_hal_det'
    embedding_path = root / (
        f'most_likely_{args.model_name}_gene_embeddings_layer_wise'
        f'{prompt_tag}.npy'
    )
    score_path = Path(
        f'ml_{args.dataset_name}_bleurt_score{prompt_tag}.npy'
    )
    if not embedding_path.is_file():
        raise FileNotFoundError(f'missing saved embeddings: {embedding_path}')
    if not score_path.is_file():
        raise FileNotFoundError(f'missing BLEURT scores: {score_path}')

    embeddings = np.load(embedding_path, mmap_mode='r')
    scores = np.asarray(np.load(score_path), dtype=np.float64).reshape(-1)
    if len(embeddings) != len(scores):
        raise ValueError(
            f'embedding/score count mismatch: {len(embeddings)} vs {len(scores)}'
        )
    if embeddings.ndim != 3:
        raise ValueError(f'expected [example, hidden-state, feature], got {embeddings.shape}')

    # hidden_states[0] is the token embedding. The released feat_loc_svd=3 path
    # removes it before referring to transformer block layers.
    block_embeddings = embeddings[:, 1:, :]
    if not 0 <= args.layer < block_embeddings.shape[1]:
        raise ValueError(
            f'layer must be in [0, {block_embeddings.shape[1] - 1}]'
        )

    plotted_indices, wild_indices = split_indices(
        len(scores), args.split, args.wild_ratio, args.seed
    )
    fit_indices = wild_indices if args.fit_split == 'wild' else plotted_indices
    fit_features = np.asarray(
        block_embeddings[fit_indices, args.layer, :], dtype=np.float32
    )
    plot_features = np.asarray(
        block_embeddings[plotted_indices, args.layer, :], dtype=np.float32
    )
    pca = PCA(n_components=2, whiten=False).fit(fit_features)
    coordinates = pca.transform(plot_features)
    labels = scores > args.truth_threshold

    if args.output is None:
        prompt_name = args.prompt_name.replace('-', '_')
        args.output = root / 'plots' / (
            f'top2_eigenvectors_{args.model_name}_{prompt_name}_'
            f'layer{args.layer}_{args.split}.png'
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix('.csv')

    figure, axis = plt.subplots(figsize=(8.0, 6.5), dpi=160)
    plotted_labels = labels[plotted_indices]
    classes = (
        (False, 'Hallucinated', '#d62728'),
        (True, 'Correct', '#1f77b4'),
    )
    for class_label, name, color in classes:
        mask = plotted_labels == class_label
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=args.point_size,
            alpha=args.alpha,
            c=color,
            label=f'{name} (n={int(mask.sum())})',
            edgecolors='none',
        )
    explained = 100.0 * pca.explained_variance_ratio_
    axis.set_xlabel(f'Principal component 1 ({explained[0]:.2f}% variance)')
    axis.set_ylabel(f'Principal component 2 ({explained[1]:.2f}% variance)')
    axis.set_title(
        f'Top two embedding directions — layer {args.layer}, {args.split} split\n'
        f'prompt={args.prompt_name}; PCA fit={args.fit_split}'
    )
    axis.axhline(0.0, color='0.75', linewidth=0.7)
    axis.axvline(0.0, color='0.75', linewidth=0.7)
    axis.grid(alpha=0.16)
    axis.legend(frameon=True)
    figure.tight_layout()
    figure.savefig(args.output, bbox_inches='tight')
    plt.close(figure)

    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ['dataset_index', 'pc1', 'pc2', 'bleurt_score', 'truth_label']
        )
        for index, coordinate in zip(plotted_indices, coordinates):
            writer.writerow(
                [
                    int(index),
                    float(coordinate[0]),
                    float(coordinate[1]),
                    float(scores[index]),
                    int(labels[index]),
                ]
            )

    print(f'PCA fitted on {len(fit_indices)} examples from: {args.fit_split}')
    print(f'Plotted {len(plotted_indices)} examples from: {args.split}')
    print(f'Correct: {int(plotted_labels.sum())}')
    print(f'Hallucinated: {int((~plotted_labels).sum())}')
    print(f'Explained variance: PC1={explained[0]:.4f}% PC2={explained[1]:.4f}%')
    print(f'Saved plot: {args.output}')
    print(f'Saved coordinates: {csv_path}')


if __name__ == '__main__':
    main()
