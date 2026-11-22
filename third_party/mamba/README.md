# Pinned Mamba Python references

Unmodified source files from [state-spaces/mamba, tag v2.2.5](https://github.com/state-spaces/mamba/tree/v2.2.5), retained under Apache-2.0 with original copyright headers:

| Local file | Upstream path |
|---|---|
| `mamba2.py` | `mamba_ssm/modules/mamba2.py` |
| `ssd_minimal.py` | `mamba_ssm/modules/ssd_minimal.py` |
| `layernorm_gated.py` | `mamba_ssm/ops/triton/layernorm_gated.py` |
| `block.py` | `mamba_ssm/modules/block.py` |
| `mixer_seq_simple.py` | `mamba_ssm/models/mixer_seq_simple.py` |

`wsr/official_reference.py` loads selected Python definitions to validate the functional adapter without importing optional CUDA kernels. This does not validate the fused CUDA implementation. Checksums are recorded in `manifest.json`.
