import numpy as np
import pytest

from haloscope_plus import (
    artifact_suffix,
    confidence_tail_labels,
    hard_threshold_labels,
    roc_auc,
    subspace_score,
)
from run_haloscope_plus import official_split
from prompt_variants import format_qa_prompt, prompt_artifact_tag, tagged_path
from plot_eigenvectors import split_indices


class FakePCA:
    mean_ = np.array([1.0, 1.0])
    components_ = np.eye(2)
    singular_values_ = np.array([2.0, 1.0])


def test_equation7_score_is_centered_squared_energy():
    values = np.array([[2.0, 3.0], [0.0, 1.0]])
    expected = np.array([(2.0 * 1.0 ** 2 + 1.0 * 2.0 ** 2) / 2.0, 1.0])
    np.testing.assert_allclose(
        subspace_score(values, FakePCA(), 2, 'equation7'), expected
    )


def test_confidence_tails_abstain_on_middle_examples():
    selected, labels, weights, lower, upper = confidence_tail_labels(
        np.arange(10, dtype=np.float64), 0.2
    )
    assert selected.sum() == 4
    np.testing.assert_array_equal(labels, [0, 0, 1, 1])
    assert np.all(weights > 0)
    assert lower < upper


def test_roc_auc_handles_tied_scores():
    assert roc_auc([0, 1, 0, 1], [0.0, 1.0, 0.5, 1.0]) == 1.0


def test_official_truthfulqa_split_sizes_and_order():
    wild, validation, test = official_split(817, wild_ratio=0.75, seed=41)
    assert (len(wild), len(validation), len(test)) == (512, 100, 205)
    assert np.all(np.diff(wild) > 0)
    assert np.all(np.diff(validation) > 0)
    assert np.all(np.diff(test) > 0)
    assert len(set(wild) | set(validation) | set(test)) == 817


def test_artifact_suffix_keeps_ablation_runs_separate_and_safe():
    assert artifact_suffix('default') == ''
    assert artifact_suffix('linear-official') == '_linear-official'
    with pytest.raises(ValueError):
        artifact_suffix('../outside')


def test_prompt_presets_preserve_baseline_and_separate_alternatives():
    assert format_qa_prompt('concise', 'Why?', 'Because.') == (
        'Answer the question concisely. Q: Why? A:Because.'
    )
    assert prompt_artifact_tag('concise') == ''
    assert prompt_artifact_tag('short-factual') == '_short-factual'
    assert str(tagged_path('scores.npy', 'most-accurate')) == 'scores_most-accurate.npy'


def test_official_hard_threshold_uses_all_examples():
    labels, weights, threshold = hard_threshold_labels(
        np.arange(10, dtype=np.float64), 0.2
    )
    assert threshold == 2.0
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(weights, np.ones(10, dtype=np.float32))


def test_eigenvector_plot_uses_official_split():
    plotted, wild = split_indices(817, 'test', wild_ratio=0.75, seed=41)
    assert len(wild) == 512
    assert len(plotted) == 205
    assert len(set(plotted) & set(wild)) == 0
