"""Run HaloScope++ from the released implementation's saved artifacts."""

import argparse
from pathlib import Path

import numpy as np


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_chat_7B')
    parser.add_argument('--dataset_name', type=str, default='tqa')
    parser.add_argument('--wild_ratio', type=float, default=0.75)
    parser.add_argument('--thres_gt', type=float, default=0.5)
    parser.add_argument('--use_rouge', type=int, default=0)
    parser.add_argument(
        '--plus_run_name', type=str, default='default',
        help='safe name appended to checkpoint, detector, and results files',
    )
    parser.add_argument('--plus_score_mode', choices=['official', 'equation7'], default='equation7')
    parser.add_argument('--plus_probe_backend', choices=['linear', 'mlp'], default='mlp')
    parser.add_argument('--plus_tail_fractions', type=str, default='0.15,0.20,0.25,0.30,0.40')
    parser.add_argument('--plus_layers', type=str, default='4,5,6,7,8,9,10,11,12,13,14,15,16')
    parser.add_argument('--plus_probe_layers', type=str, default='4,5,6,7,8,9,10,11,12,13,14,15,16')
    parser.add_argument('--plus_hidden_dim', type=int, default=128)
    parser.add_argument('--plus_dropout', type=float, default=0.2)
    parser.add_argument('--plus_epochs', type=int, default=100)
    parser.add_argument('--plus_batch_size', type=int, default=128)
    parser.add_argument('--plus_learning_rate', type=float, default=0.001)
    parser.add_argument('--plus_weight_decay', type=float, default=0.001)
    parser.add_argument('--plus_probe_repeats', type=int, default=3)
    parser.add_argument('--plus_validation_folds', type=int, default=5)
    parser.add_argument('--plus_stability_penalty', type=float, default=0.25)
    return parser


def official_split(length, wild_ratio=0.75, seed=41):
    np.random.seed(seed)
    permutation = np.random.permutation(length)
    wild_and_validation = permutation[:int(wild_ratio * length)]
    wild_set = set(wild_and_validation[:-100].tolist())
    validation_set = set(wild_and_validation[-100:].tolist())
    wild = np.asarray([index for index in range(length) if index in wild_set])
    validation = np.asarray(
        [index for index in range(length) if index in validation_set]
    )
    test = np.asarray(
        [
            index
            for index in range(length)
            if index not in wild_set and index not in validation_set
        ]
    )
    return wild, validation, test


def main():
    args = build_parser().parse_args()
    root = Path('save_for_eval') / f'{args.dataset_name}_hal_det'
    embedding_path = root / (
        f'most_likely_{args.model_name}_gene_embeddings_layer_wise.npy'
    )
    score_suffix = 'rouge' if args.use_rouge else 'bleurt'
    score_path = Path(f'ml_{args.dataset_name}_{score_suffix}_score.npy')
    if not embedding_path.is_file():
        raise FileNotFoundError(
            f'Official layer embeddings are missing: {embedding_path}. Run detect once first.'
        )
    if not score_path.is_file():
        raise FileNotFoundError(
            f'Official answer scores are missing: {score_path}. Run label first.'
        )

    embeddings = np.load(embedding_path, allow_pickle=True)
    scores = np.load(score_path)
    if len(embeddings) != len(scores):
        raise ValueError(
            f'Embedding/label count mismatch: {len(embeddings)} vs {len(scores)}'
        )
    labels = np.asarray(scores > args.thres_gt, dtype=np.int32)

    # Reproduce the released seed-41 split exactly. Selection preserves original
    # dataset order, matching the label arrays constructed in hal_det_llama.py.
    wild_indices, validation_indices, test_indices = official_split(
        len(labels), args.wild_ratio
    )

    # hidden_states[0] is the token embedding; released feat_loc_svd=3 drops it.
    embeddings = embeddings[:, 1:, :]
    print(
        f'Using official artifacts: wild={len(wild_indices)}, '
        f'validation={len(validation_indices)}, test={len(test_indices)}'
    )
    from haloscope_plus import run_haloscope_plus

    run_haloscope_plus(
        embeddings[wild_indices],
        embeddings[validation_indices],
        embeddings[test_indices],
        labels[validation_indices],
        labels[test_indices],
        args,
    )


if __name__ == '__main__':
    main()
