"""Configurable simultaneous SSM editing; legacy StateDream and checkpoints are unchanged."""
from collections.abc import Mapping

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .state_dream import history_input
from .pretrained_model import Cache, LayerCache


def validate_layers(edit_layers, count):
    layers = tuple(edit_layers)
    if not layers or len(set(layers)) != len(layers):
        raise ValueError('edit_layers must be nonempty and contain no duplicates')
    if any(type(i) is not int or not 0 <= i < count for i in layers):
        raise ValueError('edit_layers uses zero-based, in-range integer indices')
    return layers


def cache_inventory(model, cache):
    """Inspect each layer, including its actual parameter-derived cache contract."""
    if len(cache.layers) != len(model.backbone.layers):
        raise ValueError('Cache/model layer count mismatch')
    rows = []
    for i, (block, state) in enumerate(zip(model.backbone.layers, cache.layers)):
        mixer = block.mixer
        expected = (state.ssm.shape[0], mixer.nheads, mixer.headdim, mixer.d_state)
        expected_conv = (state.ssm.shape[0], mixer.conv1d.weight.shape[0], mixer.d_conv)
        if tuple(state.ssm.shape) != expected or tuple(state.conv.shape) != expected_conv:
            raise ValueError(f'Layer {i} does not match its own mixer cache contract')
        rows.append({'index': i, 'human_layer': i + 1, 'ssm_shape': list(state.ssm.shape),
                     'conv_shape': list(state.conv.shape), 'dtype': str(state.ssm.dtype),
                     'device': str(state.ssm.device), 'nheads': mixer.nheads,
                     'headdim': mixer.headdim, 'd_state': mixer.d_state})
    return rows


def edit_layers_cache(model, dream, cache, tokens, boundary, history_window=128,
                      edit_layers=(12,), rounds=None):
    """Edit original per-layer snapshots, then inject all edits atomically.

    A shared StateDream requires matching head/state dimensions, checked per layer.
    A mapping of layer -> StateDream also supports heterogeneous cache shapes.
    Returns (cache, per_layer_snapshots, per_layer_original_states).
    No token is consumed, and no convolution cache is edited.
    """
    layers = validate_layers(edit_layers, len(model.backbone.layers))
    inventory = cache_inventory(model, cache)
    if cache.processed != boundary - 1:
        raise ValueError('Anchor/cache boundary misalignment')
    if not 1 <= boundary <= tokens.shape[1] or history_window < 0:
        raise ValueError('Invalid visible boundary or history window')
    if isinstance(dream, Mapping) and set(dream) != set(layers):
        raise ValueError('Per-layer editors must exactly match edit_layers')
    # Validate all editor shapes before running any edit. Never assume layer equality.
    for i in layers:
        editor = dream[i] if isinstance(dream, Mapping) else dream
        if editor is not None:
            row = inventory[i]
            if (editor.read.in_features != row['d_state'] or
                    editor.head_embedding.num_embeddings != row['nheads'] or
                    editor.history_project.in_features != model.width):
                raise ValueError(f'Editor incompatible with actual cache at layer {i}')
    history = history_input(model, tokens, boundary, history_window) if dream is not None else None
    edits, traces, initial = {}, {}, {}
    for i in layers:
        editor = dream[i] if isinstance(dream, Mapping) else dream
        initial[i] = cache.layers[i].ssm
        if editor is None:
            edits[i], traces[i] = initial[i], []
        else:
            edits[i], traces[i] = editor(initial[i], history, rounds=rounds)
    return cache.edit_many(edits), traces, initial


def relative_edit_norm(initial, edited):
    """Concatenated norm across layers; also usable when shapes differ."""
    numerator = sum((edited[i] - h).double().square().sum() for i, h in initial.items())
    denominator = sum(h.double().square().sum() for h in initial.values())
    return float((numerator / denominator.clamp_min(1e-30)).sqrt())


def checkpointed_continuation_loss(model, tokens, cache, boundary, length, chunk=64):
    """Exact full-horizon gradient using temporal activation recomputation, without detach.

    Cache tensors connect successive chunks in the graph, including convolution states.
    This changes memory/compute only; it does not truncate the supervision or gradient.
    """
    if cache.processed != boundary - 1 or boundary < 1 or boundary + length > tokens.shape[1]:
        raise ValueError('Invalid continuation/anchor boundary')
    if chunk < 1 or length < 1:
        raise ValueError('Chunk and continuation lengths must be positive')
    state = cache
    terms = []
    for start in range(0, length, chunk):
        end = min(length, start + chunk)
        inputs = tokens[:, boundary - 1 + start:boundary - 1 + end]
        targets = tokens[:, boundary + start:boundary + end]
        def compute(*flat, inputs=inputs, targets=targets, processed=state.processed):
            restored = Cache(tuple(LayerCache(flat[i], flat[i+1]) for i in range(0, len(flat), 2)), processed)
            logits, advanced = model.run(inputs, restored)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            return (loss,) + tuple(t for layer in advanced.layers for t in (layer.conv, layer.ssm))
        flat = tuple(t for layer in state.layers for t in (layer.conv, layer.ssm))
        outputs = checkpoint(compute, *flat, use_reentrant=False, preserve_rng_state=False)
        terms.append(outputs[0] * ((end - start) / length))
        flat = outputs[1:]
        state = Cache(tuple(LayerCache(flat[i], flat[i+1]) for i in range(0, len(flat), 2)),
                      state.processed + end - start)
    return torch.stack(terms).sum()
