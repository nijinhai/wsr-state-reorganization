import unittest

import torch

from wsr.layer_edit import cache_inventory, edit_layers_cache, checkpointed_continuation_loss
from wsr.pretrained_model import PretrainedMamba2
from wsr.pretrained_model import Mixer
from wsr.pretrained_experiment import continuation
from wsr.state_dream import StateDream, edit_cache


class LayerEditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        self.model = PretrainedMamba2(width=4, layers=24, vocab=16, state=4,
                                      headdim=2, block=4).requires_grad_(False)
        self.tokens = torch.randint(0, 16, (1, 16))
        self.boundary = 7
        self.cache = self.model.wake(self.tokens[:, :6])
        self.dream = StateDream(memory=4, heads=4, width=4, hidden=8)

    def test_default_exactly_reproduces_legacy_and_zero_edit(self):
        old, trace, initial = edit_cache(self.model, self.dream, self.cache, self.tokens, 7, 4, 12)
        new, traces, initials = edit_layers_cache(self.model, self.dream, self.cache, self.tokens, 7, 4)
        self.assertTrue(torch.equal(old.layers[12].ssm, new.layers[12].ssm))
        self.assertIs(initial, initials[12])
        for a, b in zip(trace, traces[12]):
            self.assertTrue(torch.equal(a, b))
        zero, _, _ = edit_layers_cache(self.model, None, self.cache, self.tokens, 7, 4)
        old_loss, _ = continuation(self.model, self.tokens, self.cache, 7, 4)
        new_loss, _ = continuation(self.model, self.tokens, zero, 7, 4)
        self.assertTrue(torch.equal(old_loss, new_loss))

    def test_multilayer_preserves_unselected_and_conv_and_input_cache(self):
        originals = [s.ssm.clone() for s in self.cache.layers]
        changed, _, _ = edit_layers_cache(self.model, self.dream, self.cache, self.tokens, 7,
                                          4, [11, 12, 13])
        self.assertEqual(changed.processed, 6)
        for i in range(24):
            self.assertIs(changed.layers[i].conv, self.cache.layers[i].conv)
            self.assertTrue(torch.equal(originals[i], self.cache.layers[i].ssm))
            if i not in [11, 12, 13]:
                self.assertIs(changed.layers[i], self.cache.layers[i])
            else:
                self.assertFalse(torch.equal(changed.layers[i].ssm, originals[i]))

    def test_each_injected_layer_receives_gradient_and_no_future_leak(self):
        hs = {i: self.cache.layers[i].ssm.detach().clone().requires_grad_() for i in [5, 12, 19]}
        cache = self.cache.edit_many(hs)
        loss, _ = continuation(self.model, self.tokens, cache, 7, 4)
        grads = torch.autograd.grad(loss, tuple(hs.values()))
        for g in grads:
            self.assertTrue(torch.isfinite(g).all())
            self.assertGreater(float(g.norm()), 0)
        changed, _, _ = edit_layers_cache(self.model, self.dream, self.cache, self.tokens, 7, 4, [11, 12, 13])
        other = self.tokens.clone()
        other[:, 7:] = (other[:, 7:] + 1) % 16
        comparison, _, _ = edit_layers_cache(self.model, self.dream, self.cache, other, 7, 4, [11, 12, 13])
        for i in [11, 12, 13]:
            self.assertTrue(torch.equal(changed.layers[i].ssm, comparison.layers[i].ssm))
        loss, advanced = continuation(self.model, self.tokens, changed, 7, 4)
        loss.backward()
        self.assertGreater(sum(float(p.grad.norm()) for p in self.dream.parameters() if p.grad is not None), 0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertEqual(advanced.processed, 10)

    def test_rejects_invalid_indices_and_actual_layer_shape_mismatch(self):
        for layers in [[], [12, 12], [-1], [24], [True], [1.0]]:
            with self.assertRaises(ValueError):
                edit_layers_cache(self.model, self.dream, self.cache, self.tokens, 7, 4, layers)
        with self.assertRaises(ValueError):
            self.cache.edit_many({12: self.cache.layers[12].ssm.double()})
        with self.assertRaises(ValueError):
            self.cache.edit_many({24: self.cache.layers[12].ssm})
        bad = StateDream(memory=5, heads=4, width=4, hidden=8)
        with self.assertRaises(ValueError):
            edit_layers_cache(self.model, bad, self.cache, self.tokens, 7)
        self.assertEqual(len(cache_inventory(self.model, self.cache)), 24)

    def test_heterogeneous_layers_require_compatible_per_layer_editors(self):
        self.model.backbone.layers[19].mixer = Mixer(width=4, state=6, headdim=4, block=4).requires_grad_(False)
        cache = self.model.wake(self.tokens[:, :6])
        rows = cache_inventory(self.model, cache)
        self.assertEqual(rows[12]['ssm_shape'], [1, 4, 2, 4])
        self.assertEqual(rows[19]['ssm_shape'], [1, 2, 4, 6])
        with self.assertRaises(ValueError):
            edit_layers_cache(self.model, self.dream, cache, self.tokens, 7, 4, [12, 19])
        different = StateDream(memory=6, heads=2, width=4, hidden=8)
        edited, _, _ = edit_layers_cache(self.model, {12: self.dream, 19: different}, cache,
                                         self.tokens, 7, 4, [12, 19])
        loss, _ = continuation(self.model, self.tokens, edited, 7, 4)
        loss.backward()
        for editor in [self.dream, different]:
            self.assertGreater(sum(float(p.grad.norm()) for p in editor.parameters() if p.grad is not None), 0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_checkpointed_loss_and_state_gradient_match_direct_full_horizon(self):
        for layer in [5, 12, 19]:
            h = self.cache.layers[layer].ssm.detach().clone().requires_grad_()
            cache = self.cache.edit_many({layer: h})
            direct, _ = continuation(self.model, self.tokens, cache, 7, 8)
            gd, = torch.autograd.grad(direct, h)
            checked = checkpointed_continuation_loss(self.model, self.tokens, cache, 7, 8, chunk=4)
            gc, = torch.autograd.grad(checked, h)
            torch.testing.assert_close(checked, direct, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(gc, gd, atol=2e-6, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
