# HaloScope++ on the released implementation

This branch is based on `codex/environment-compatible` in the official-code fork.
The released detector remains unchanged and is still selected by:

```bash
sbatch run_llama.sbatch detect
```

The opt-in improved detector uses the exact same saved answers, BLEURT labels,
layer embeddings, seed-41 split, validation set, and test set:

```bash
sbatch run_llama.sbatch detect-plus
```

It changes only the detection method:

- fits PCA on the 512 wild examples rather than on validation examples;
- uses the centered, squared Equation 7 score by default;
- searches middle layers 4 through 16 and `k=1..10`;
- pseudo-labels only confident upper and lower score tails;
- ignores ambiguous middle examples;
- weights examples by their distance from the median score rank;
- trains a standardized 128-unit dropout MLP with AdamW;
- averages three deterministic probe seeds;
- selects using five stratified validation folds with a stability penalty.

Search progress is saved after every `(tail fraction, probe layer)` candidate to:

```text
save_for_eval/tqa_hal_det/haloscope_plus_search_llama2_chat_7B.pt
```

If Slurm stops the job, submit `detect-plus` again. Completed candidates are
skipped. If arguments change, move the old checkpoint first because results from
different search settings must not be mixed.

Final artifacts are written separately:

```text
save_for_eval/tqa_hal_det/haloscope_plus_results_llama2_chat_7B.json
save_for_eval/tqa_hal_det/haloscope_plus_detector_llama2_chat_7B.pt
```

The official baseline files are not overwritten. Compare the `test AUROC` printed
by `detect` with the `test_auroc` field in the HaloScope++ JSON result.

## Optional ablations

Reproduce the released projection formula while keeping the improved pseudo-label
and probe stages:

```bash
sbatch run_llama.sbatch detect-plus --plus_score_mode official
```

## Controlled probe ablations

Give every experiment a unique `--plus_run_name`. Its search checkpoint, detector,
and JSON result then coexist with all other runs and remain independently resumable.

The following commands hold the official projection score and all training settings
fixed while changing only probe capacity:

```bash
# Standardized logistic regression (one linear layer).
sbatch run_llama.sbatch detect-plus \
  --plus_run_name official-linear \
  --plus_score_mode official \
  --plus_probe_backend linear

# Current small MLP.
sbatch run_llama.sbatch detect-plus \
  --plus_run_name official-mlp128 \
  --plus_score_mode official \
  --plus_probe_backend mlp \
  --plus_hidden_dim 128

# Capacity ablation matching the released probe's 1024 hidden units. This deliberately
# retains HaloScope++ standardization, weighting, AdamW, and repeated-seed ensemble so
# that hidden width is the only change; it is not a duplicate of the baseline detector.
sbatch run_llama.sbatch detect-plus \
  --plus_run_name official-mlp1024 \
  --plus_score_mode official \
  --plus_probe_backend mlp \
  --plus_hidden_dim 1024
```

The direct-projection metrics are printed and stored in every result JSON. Run the
released baseline separately with `sbatch run_llama.sbatch detect`; do not describe
the 1024-unit capacity ablation above as the released baseline because its remaining
training procedure intentionally stays identical to HaloScope++.

## Improved probe on a frozen official run

Use `probe-plus` to isolate probe training from all other HaloScope++ changes. It
reuses the saved official answers, BLEURT labels, block embeddings, split, released
projection score, hard pseudo-label threshold, and all 512 wild examples. Defaults
are the configuration printed by the completed official TruthfulQA run: subspace
layer 11, `k=10`, threshold quantile `0.07692307692307693`, and probe layer 6.

```bash
sbatch run_llama.sbatch probe-plus \
  --plus_run_name fixed-official-mlp128 \
  --plus_score_mode official \
  --plus_probe_backend mlp \
  --plus_hidden_dim 128
```

This stage does not load Llama or regenerate embeddings. It changes only probe
training: feature standardization, class balancing, a 128-unit dropout MLP, AdamW,
and a three-seed ensemble. Results are saved to:

```text
save_for_eval/tqa_hal_det/official_probe_plus_results_llama2_chat_7B_fixed-official-mlp128.json
```

If a different official run selects different values, pass them explicitly:

```bash
--official_subspace_layer 11 \
--official_components 10 \
--official_threshold_quantile 0.07692307692307693 \
--official_probe_layer 6
```

The configurable options are visible with:

```bash
python run_haloscope_plus.py --help
```

## Prompt ablations

Three named TruthfulQA prompt presets are available:

- `concise`: `Answer the question concisely.` (released default)
- `short-factual`: `Provide a short factual answer.`
- `most-accurate`: `Give only the most accurate answer.`

Each prompt has separate answer, BLEURT-score, embedding, threshold-search,
HaloScope++ checkpoint, detector, and result filenames. The `concise` preset keeps
the released filenames unchanged. Run every alternative through the complete
pipeline, using exactly the same `--prompt_name` at every stage:

```bash
PROMPT_NAME=short-factual

sbatch run_llama.sbatch generate --prompt_name "$PROMPT_NAME"
# Wait for generation to finish.
sbatch run_llama.sbatch label --prompt_name "$PROMPT_NAME"
# Wait for labeling to finish.
sbatch run_llama.sbatch detect --prompt_name "$PROMPT_NAME"
# Wait for embedding extraction/detection to finish.
sbatch run_llama.sbatch detect-plus \
  --prompt_name "$PROMPT_NAME" \
  --plus_run_name official-mlp128 \
  --plus_score_mode official \
  --plus_probe_backend mlp \
  --plus_hidden_dim 128
```

Repeat with `PROMPT_NAME=most-accurate`. Generation resumes independently for each
prompt. Do not submit a downstream stage until the preceding stage has completed;
otherwise it will fail because that prompt's artifacts do not exist yet.
