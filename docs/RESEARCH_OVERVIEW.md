# Research overview / 研究总结

The research question is whether additional shared recurrent computation can create useful intermediate results in a frozen model or in a query-blind workspace. The project progressively separates four possible bottlenecks: **stored information, readable representation, learned computation, and computational cost**.

## Evidence map

| Stage | Main control | What the evidence supports | Main limit |
|---|---|---|---|
| Initialization diagnostics | Original/slow/multiple decay scales, GRU, small-sample fit | Time-scale initialization affects basic state retention | Early simplified SSM, not the official pretrained architecture |
| Frozen Mamba state editing | Residual/recurrent editors, denoising, corruption, no-Dream | Correct differentiable state-editing interface | Tiny/inconsistent LM improvements; short training horizon |
| Training signals | B0 / shared queries S / future-feature P | Diagnose later degradation and head-vs-state contributions | Auxiliary metrics do not establish improved LM reasoning |
| Query composition | Oracle facts / raw-reader / edited-reader; MLP/recurrent cores | Final-answer supervision can train structured recurrent composition | Oracle facts, node slots, address structure and loop depth are provided |
| Layer sensitivity | Equal-relative-norm perturbations at layers6/13/20 | Local state-gradient directions can lower continuation loss | True futures used offline; no learned multilayer coordination |
| Read inhibition | Ordinary/round-encoding/soft-Coverage/hard-mask | Useful count-related intermediate states, but no stable Coverage gain | Explicit query addresses and fixed read cursor |
| Query-blind precomputation | A / B / B-budget / C1–8 / D8 | Reusable precomputation is learnable; parameters and compute explain important differences | No unique recurrent generalization win at the tested budget |

Read the phase-specific [findings](../results/README.md) for sample sizes, seed scope, interventions and negative controls. Three seeds describe initialization/schedule variation on fixed datasets, not population confidence intervals.

## Engineering work

- Functional SSM/cache objects, selected-layer injection and preservation of convolution caches.
- Alignment around a pending last-visible-token anchor, avoiding stale pre-edit logits.
- Loss gradients through frozen continuation, including full connected offline horizons with activation recomputation.
- Immutable configuration checks, resumable optimizer checkpoints, paired world/query schedules and file-preservation audits.
- Causal diagnosis by state/evidence replacement, with answer correctness and meaningful counterfactuals separated from mere output changes.
- Explicit comparison of parameter count, optimizer updates, final-answer supervision, linear MACs and measured inference cost.

## What remains open

No autonomous stopping or autonomous address discovery is established. Different-looking recurrent states are not evidence of distinct useful stages. An oracle sandbox result is not a language-model result. The independent sandboxes have not been integrated into Mamba, and there is no established stable real-text LM gain from Direct Dream.

The query-blind study does show a useful parameter/compute tradeoff: C8 reuses about 67K parameters, while the MAC-matched wider single pass has about 461K. However, the wider model generalizes better to held query combinations. This is a measured tradeoff, not a proof of recurrence's superiority or impossibility.

## Task-screening record

Before the Coverage study, harder relation-chain/lookup/affine pilots did not pass the baseline's generalization gate; a four-read count task reached a ceiling. Their outcomes and adjustment reasons remain in [WSR_V03_PRE.md](WSR_V03_PRE.md). The subsequent query-blind experiment used **one fixed graph task** without searching for a setting where C beats B.

This public release includes selected pilot summaries if available, but does not redistribute the original complete checkpoint directories. The main comparisons and contradictory results are retained.

## Contribution and assistance

Research direction, experimental constraints and iteration decisions were led by Jinhai Ni. AI coding tools assisted implementation, experimental execution and documentation. The contribution is the combined experimental system and hypothesis-driven investigation; the project makes no publication, institutional affiliation or manual-authorship claim beyond that record.
