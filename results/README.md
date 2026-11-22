# Published evidence

These files are curated historical results, not regenerated showcase numbers. Fresh runs write to ignored `artifacts/` and do not replace this directory.

| Evidence | Reports |
|---|---|
| Initialization/time scales | [Report](timescale-v1/REPORT.md) |
| Frozen pretrained state editing | [Report](pretrained-v1/REPORT.md), [findings](pretrained-v1/FINDINGS.md) |
| Training budgets and S/P signals | [Report](wsr-v0.2a/REPORT.md), [findings](wsr-v0.2a/FINDINGS.md) |
| Amplitude/readability/predictability diagnosis | [Report](wsr-v0.2a-diagnostics/REPORT.md), [findings](wsr-v0.2a-diagnostics/FINDINGS.md) |
| Query composition | [Report](wsr-v0.2b/REPORT.md), [findings](wsr-v0.2b/FINDINGS.md) |
| Editing layer diagnosis | [Report](wsr-v0.2c/REPORT.md), [findings](wsr-v0.2c/FINDINGS.md) |
| Coverage/workspace precursor | [Report](wsr-v0.3-pre/REPORT.md), [findings](wsr-v0.3-pre/FINDINGS.md) |
| Query-blind precomputation | [Report](wsr-v0.3-precompute/REPORT.md), [findings](wsr-v0.3-precompute/FINDINGS.md) |

Included: reports, scalar numeric tables, figures, selected protocols, per-run curves/metadata for the two recent sandbox studies, and a provenance manifest. Excluded: external weights, corpus document bodies, token caches, serialized hidden states, optimizer/checkpoint binaries, shell history, credentials and private machine paths.

[export-provenance.json](export-provenance.json) records the original SHA-256 for exported evidence. CSV, numeric JSON and image/PDF bytes remain unchanged. Report Markdown links are relocated to the public layout. Original path references that do not exist in the curated release are rendered as archival references rather than broken links.

Published local preservation-audit counts describe the original research archive; they do not mean that its thousands of binary artifacts are redistributed. `scripts/verify_published_results.py` verifies the smaller public evidence set. Hardware timings are observations, not promises for another machine.
