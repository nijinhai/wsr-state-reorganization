"""Independent numerical checks run offline against pinned official definitions."""
import hashlib
import json
from pathlib import Path
import unittest

import torch

from wsr.official_reference import namespace
from wsr.pretrained_model import ssd_chunk


class PublicReferenceTests(unittest.TestCase):
    def test_vendored_reference_checksums(self):
        directory = Path(__file__).resolve().parents[1]/'third_party/mamba'
        manifest = json.loads((directory/'manifest.json').read_text())
        for filename, record in manifest.items():
            self.assertEqual(hashlib.sha256((directory/filename).read_bytes()).hexdigest(), record['sha256'])

    def test_official_ssd_output_final_state_and_state_gradient(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        ns = namespace()
        x = torch.randn(1, 8, 2, 3, dtype=torch.float64)
        dt = torch.rand(1, 8, 2, dtype=torch.float64)*.1
        a = -torch.tensor([.5, 2.], dtype=torch.float64)
        b = torch.randn(1, 8, 5, dtype=torch.float64)
        c = torch.randn_like(b)
        h = torch.randn(1, 2, 3, 5, dtype=torch.float64, requires_grad=True)
        actual, final = ssd_chunk(x, dt, a, b, c, h)
        expected, expected_final = ns['ssd_minimal_discrete'](
            x*dt[..., None], dt*a, b[:, :, None].expand(-1, -1, 2, -1),
            c[:, :, None].expand(-1, -1, 2, -1), 8, initial_states=h[:, None])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(final, expected_final)
        ga, = torch.autograd.grad(actual.square().mean()+final.square().mean(), h, retain_graph=True)
        ge, = torch.autograd.grad(expected.square().mean()+expected_final.square().mean(), h)
        torch.testing.assert_close(ga, ge)


if __name__ == '__main__':
    unittest.main()
