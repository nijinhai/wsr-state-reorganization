import unittest

import torch
from torch.nn import functional as F

from wsr.workspace_model import Workspace
from wsr.workspace_tasks import execute, make_data, validate_data


class WorkspaceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data = make_data()

    def test_symbolic_labels_and_world_program_splits(self):
        metrics = validate_data(self.data)
        self.assertEqual(metrics['train']['examples'], 8008)
        self.assertTrue(all(x['symbolic_accuracy'] == 1 for x in metrics.values()))

    def test_common_initialization_and_zero_penalty_equivalence(self):
        models = []
        for mode in 'ABCD':
            torch.manual_seed(2)
            models.append(Workspace(mode))
        for b in models[1:]:
            for k, v in models[0].state_dict().items():
                self.assertTrue(torch.equal(v, b.state_dict()[k]))
        block = self.data['train']
        args = [block[k][:8] for k in ['memory', 'start', 'program']]
        a, _ = models[0](*args)
        b, _ = models[1](*args)
        c, _ = models[2](*args, penalty=0)
        d, _ = models[3](*args, disable_mask=True)
        for value in [b, c, d]:
            torch.testing.assert_close(a, value, atol=0, rtol=0)

    def test_soft_coverage_formula_memory_readonly_and_final_loss_gradient(self):
        torch.manual_seed(5)
        model = Workspace('C', penalty=1.5, rho=.7)
        block = self.data['train']
        memory = block['memory'][:8].clone()
        original = memory.clone()
        logits, trace = model(memory, block['start'][:8], block['program'][:8], return_trace=True)
        self.assertTrue(torch.equal(memory, original))
        self.assertTrue(torch.equal(trace['coverage_before'][:, 0], torch.zeros(8, 16)))
        for k in range(1, 8):
            torch.testing.assert_close(trace['coverage_before'][:, k],
                .7 * trace['coverage_before'][:, k-1] + trace['attention'][:, k-1])
            address = model.node(block['program'][:8, k])
            scores = (address + model.read_query(trace['states'][:, k-1])) @ model.node.weight.T / model.hidden**.5
            expected = (scores - 1.5 * trace['coverage_before'][:, k]).softmax(-1)
            torch.testing.assert_close(trace['attention'][:, k], expected)
        self.assertTrue((trace['attention'] > 0).all())
        # The stacked trace is a reporting view; retain the actual recurrence with a hook.
        outputs = []
        handle = model.update.register_forward_hook(lambda m, x, y: outputs.append(y))
        logits, _ = model(memory, block['start'][:8], block['program'][:8])
        outputs[0].retain_grad()
        F.cross_entropy(logits, block['label'][:8]).backward()
        handle.remove()
        self.assertGreater(float(outputs[0].grad.norm()), 0)
        self.assertGreater(float(model.update.weight_hh.grad.norm()), 0)
        self.assertGreater(float(model.value[0].weight.grad.norm()), 0)

    def test_hard_mask_and_no_bank_bypass(self):
        block = self.data['train']
        args = [block[k][:16] for k in ['memory', 'start', 'program']]
        model = Workspace('D')
        logits, trace = model(*args, rounds=8, return_trace=True)
        choices = trace['attention'].argmax(-1)
        for row in choices:
            self.assertEqual(len(row.unique()), 8)
        for k in range(1, 8):
            used = choices[:, :k]
            self.assertTrue((trace['attention'][:, k].gather(1, used) == 0).all())
        changed, _ = model(*args, intervention={'kind': 'zero_evidence', 'round': 2})
        reference, _ = model(*args)
        self.assertGreater(float((changed-reference).norm()), 0)

    def test_counterfactual_weighted_labels_match_symbolic_execution(self):
        block = self.data['validation_new']
        for i in range(40):
            memory = block['memory'][i].clone()
            program = block['program'][i].tolist()
            label = int(block['label'][i])
            slot = program[0]
            bit = int(memory[slot, 0])
            memory[slot, 0] = 1-bit
            coefficient = sum(program[8+k]+1 for k in range(8) if program[k]==slot)
            actual, _ = execute(memory.tolist(), 0, program)
            self.assertEqual(actual, label+coefficient*(1-2*bit))


if __name__ == '__main__':
    unittest.main()
