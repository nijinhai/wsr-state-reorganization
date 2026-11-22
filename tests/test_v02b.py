import unittest
import torch
from torch.nn import functional as F

from wsr.v02b_query import RecurrentQuery,QueryMLP,balanced_rules,generate_rules,rule_key,oracle_features,labels_for


class QueryContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(3)

    def test_balanced_rules_are_disjoint_and_exclude_shortcuts(self):
        used={rule_key(r) for pool in generate_rules().values() for r in pool}
        for mapping,colors in balanced_rules():
            self.assertNotIn((mapping,colors),used)
            self.assertEqual(len(set(colors)),4)
            for start in range(4):
                nodes=[start]
                for _ in range(4):nodes.append(mapping[nodes[-1]])
                self.assertEqual(len(set(nodes[:4])),4)
                self.assertEqual(nodes[4],start)

    def test_final_answer_loss_reaches_shared_update_and_first_state(self):
        model=RecurrentQuery();facts=torch.tensor([[1,2,3,0,2,1,0,3],[2,3,1,0,0,3,1,2]])
        queries=torch.tensor([[2,5],[2,5]])
        calls=[];hook=model.update.register_forward_hook(lambda *args:calls.append(1))
        logits,trace=model(oracle_features(facts),queries)
        trace['states'][0].retain_grad()
        y=labels_for(facts).gather(1,queries)
        F.cross_entropy(logits.flatten(0,1),y.flatten()).backward();hook.remove()
        self.assertEqual(len(calls),3)
        self.assertGreater(float(model.update.weight_hh.grad.norm()),0)
        self.assertGreater(float(model.fact_value[1].weight.grad.norm()),0)
        self.assertGreater(float(trace['states'][0].grad.norm()),0)

    def test_intervention_is_after_round_one_and_does_not_touch_finished_queries(self):
        model=RecurrentQuery();source=torch.randn(3,128);q=torch.arange(12)[None].expand(3,-1)
        intact,trace=model(source,q);changed,other=model(source,q,intervention='zero')
        torch.testing.assert_close(trace['states'][0],other['states'][0],rtol=0,atol=0)
        torch.testing.assert_close(intact[:,::3],changed[:,::3],rtol=0,atol=0)
        self.assertGreater(float((intact[:,1::3]-changed[:,1::3]).norm()),0)

    def test_source_conditions_have_paired_core_initialization(self):
        torch.manual_seed(7);a=RecurrentQuery()
        torch.manual_seed(7);c=RecurrentQuery(source='raw_reader')
        for key,value in a.state_dict().items():torch.testing.assert_close(value,c.state_dict()[key],rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in a.parameters()),3092)
        self.assertEqual(sum(p.numel() for p in QueryMLP(compact=True,hidden=57).parameters()),3081)


if __name__=='__main__':unittest.main()
