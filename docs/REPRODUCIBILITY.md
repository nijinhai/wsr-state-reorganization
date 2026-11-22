# Reproduction guide

Run commands from the repository root. The tested environment is Linux/WSL with Python 3.12 and PyTorch 2.6.0; Python 3.11+ is required because asset verification uses `hashlib.file_digest`. The small sandbox and contract tests run on CPU. Some historical pipeline utilities use POSIX file locks.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[analysis,pretrained,v02]'
```

For GPU language-model runs, install the appropriate CUDA-enabled Torch build instead. Native Mamba CUDA extensions were not used in the archived results; the functional adapter keeps gradients to edited SSM states.

## 2. Offline correctness and evidence checks

```bash
python -m unittest discover -s tests -v
python scripts/verify_published_results.py
python scripts/check_public_release.py
```

The original 49 tests are retained, plus public-release checks against bundled official SSD references. No weights or corpora are needed. `verify_published_results.py` independently recomputes means/SD, compares per-seed CSV and JSON, verifies parameter/MAC claims and hashes the exported numerical evidence.

## 3. Small CPU demo

```bash
python -m wsr.demo --model C8 --steps 512
python -m wsr.demo --model B --steps 512
```

This is a **training-query fit check on 32 fixed instances**, with held query combinations reported separately. It demonstrates final-answer gradients and reuse of one query-blind workspace. It is not the benchmark and does not write research results into `results/`.

## 4. Full query-blind sandbox

```bash
python -m wsr.reproduce_precompute --stage smoke
python -m wsr.reproduce_precompute --stage full
```

The wrapper orchestrates the original protocol without changing the task, models, seeds, optimizer, data schedule, query count or budget:

1. Generate 3,072/384/384 train/validation/test worlds, with disjoint link graphs and balanced color rotations.
2. Check 32-instance learning for B/C8/D8. A failed gate stops expansion; it does not redesign the task.
3. Run all seven configurations for 512 updates at seed0; record the unchanged predeclared continuation decision.
4. Continue seed0 and run seeds1/2 to 2,048 updates each. B is exactly C1; no duplicate C1 run.
5. Evaluate model test performance only after all 21 runs complete. Generate reports, curves, per-hop metrics, query costs and a preservation audit.

Outputs are resumable under `artifacts/wsr-v0.3-precompute/`. Use a fresh clone/directory for a fresh run; existing mismatched configurations are rejected. No test score is used for checkpoint selection.

Published wall-clock values are measurements from the original machine, not timing guarantees. Repeated queries in the cost study are sampled with replacement from 24 combinations; the reported D protocol recomputes each query and does not memoize answers. Training query count is fixed at eight, so the study does not isolate training-time multi-query supervision gains.

## 5. Earlier independent workspace study

The historical protocol and archived results are in [WSR_V03_PRE.md](WSR_V03_PRE.md) and [results/wsr-v0.3-pre](../results/wsr-v0.3-pre). Entry point: `python -m wsr.workspace_experiment --help`. The original gating procedure and task-screening history remain documented; the query-blind study is the recommended fully orchestrated reproduction.

## 6. Optional frozen-Mamba experiments

These need separate network downloads and substantially more compute. The public repository contains source, pinned reference definitions and numerical tables; **not** the pretrained weights, corpus text, prefix caches or trained Dream checkpoints.

```bash
python -m wsr.fetch_pretrained_assets
python -m wsr.pretrained_data
python -m wsr.verify_pretrained --device cuda
python -m wsr.experiment smoke
python -m wsr.experiment pilot
```

The fetcher pins Mamba2-130M and tokenizer revisions, verifies model hashes, and obtains source references. Follow [the historical v0.1 procedure](V01_ORIGINAL_README.md) for the training matrix, then [v0.2a](WSR_V02A.md) for expanded documents/training. Later diagnostics depend on earlier generated banks/checkpoints and are not standalone downloads. The original corpus downloads depend on external endpoints and source terms; no private local artifact is silently supplied.

The LM model remains frozen; gradients flow through continuation to Dream. LM training has a **64-token gradient horizon**, while **8K/32K** denotes evaluation histories. v0.2c's 512-token offline gradient analysis uses connected activation-checkpoint blocks and true future supervision; it is a diagnostic, not inference-time improvement.

## Evidence scope

`results/` is a curated, versioned historical snapshot. Numeric files and figures retain original hashes in [export-provenance.json](../results/export-provenance.json); Markdown links are rewritten for this release. Full multi-gigabyte run directories and private machine paths are omitted. Some archival reports refer to local files by inline code; these references are not claims that the corresponding caches/checkpoints are distributed.
