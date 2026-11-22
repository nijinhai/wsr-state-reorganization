import unittest
import torch
from wsr.precompute_model import Precompute, budget_hidden
from wsr.precompute_tasks import make_data, audit, answers


class PrecomputeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data = make_data(sizes=(16, 8, 8))

    def test_generator_and_balance(self):
        audit(self.data)
        # Separate scalar symbolic executor, not the vectorized label routine.
        b = self.data['train']
        for l, c, labels in zip(b['links'].tolist(), b['colors'].tolist(), b['labels'].tolist()):
            for (s, h), y in zip(self.data['queries'].tolist(), labels):
                node = s
                for _ in range(h):
                    node = l[node]
                self.assertEqual(c[node], y)

    def test_blind_interface_and_reuse(self):
        b, q = self.data['train'], self.data['queries'][None]
        model = Precompute('C', rounds=2)
        l, c = b['links'][:1], b['colors'][:1]
        w, _ = model.prepare(l, c)
        self.assertTrue(torch.equal(model.readout(w, q), model(l, c, q)))
        self.assertTrue(torch.equal(model.readout(w, q[:, :4]), model.readout(w, q)[:, :4]))
        with self.assertRaises(ValueError):
            model.prepare(l, c, query=q)

    def test_head_can_express_all_answers_without_hidden_compute(self):
        model = Precompute('A')
        labels = self.data['train']['labels'][:8]
        q = self.data['queries']
        w = torch.zeros(8, 6, 32)
        with torch.no_grad():
            model.head_weight.zero_()
            model.head_bias.zero_()
            for h in range(4):
                model.head_weight[h, h*3:(h+1)*3] = torch.eye(3)
            for i, (s, h) in enumerate(q.tolist()):
                w[torch.arange(8), s, (h-1)*3 + labels[:, i]] = 10.
        self.assertTrue(torch.equal(model.readout(w, q[None].expand(8, -1, -1)).argmax(-1), labels))

    def test_B_equals_C1(self):
        torch.manual_seed(9)
        b = Precompute('B', rounds=1)
        torch.manual_seed(9)
        c = Precompute('C', rounds=1)
        block = self.data['train']
        q = self.data['queries'][None].expand(2, -1, -1)
        self.assertTrue(torch.equal(b(block['links'][:2], block['colors'][:2], q),
                                    c(block['links'][:2], block['colors'][:2], q)))

    def test_gradients_and_D(self):
        block = self.data['train']
        q = self.data['queries'][None].expand(2, -1, -1)
        for mode in ('A', 'B', 'B-budget', 'C', 'D'):
            model = Precompute(mode, rounds=1 if mode in ('B', 'B-budget') else 2)
            logits = model(block['links'][:2], block['colors'][:2], q)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), block['labels'][:2].flatten())
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            if mode == 'D':
                single = model(block['links'][:2], block['colors'][:2], q[:, :1])
                self.assertTrue(torch.allclose(single, logits[:, :1], atol=1e-6))

    def test_budget_and_cost_amortization(self):
        c = Precompute('C', rounds=8)
        b = Precompute('B-budget', hidden=budget_hidden(), rounds=1)
        d = Precompute('D', rounds=8)
        self.assertEqual(c.macs()['prepare'], b.macs()['prepare'])
        self.assertLess(c.macs(64)['total']/64, c.macs(1)['total'])
        self.assertAlmostEqual(d.macs(64)['total']/64, d.macs(1)['total'])

    def test_node_permutation_executor(self):
        block, q = self.data['train'], self.data['queries']
        p = torch.tensor([2, 0, 5, 4, 1, 3])
        l, c = torch.empty_like(block['links']), torch.empty_like(block['colors'])
        l[:, p], c[:, p] = p[block['links']], block['colors']
        qp = q.clone()
        qp[:, 0] = p[q[:, 0]]
        self.assertTrue(torch.equal(answers(l, c, qp), block['labels']))

    def test_permutation_diagnostic_accounts_for_mapped_query_groups(self):
        from wsr.precompute_experiment import diagnostics
        result = diagnostics(Precompute('A'), self.data['test'], self.data)
        metadata = result['node_permutation_metadata']
        self.assertEqual(metadata['novel_worlds'] + metadata['excluded_worlds_matching_training_graph'],
                         len(self.data['test']['links']))
        groups = result['node_permutation_mapped_split_novel_rules']
        count = groups['seen_query']['all']['answers'] + groups['held_query']['all']['answers']
        self.assertEqual(count, metadata['novel_worlds'] * 24)


if __name__ == '__main__':
    unittest.main()
