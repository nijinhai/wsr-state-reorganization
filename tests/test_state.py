import unittest
import torch
from wsr.model import Backbone, Dream, corrupt

class StateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model=Backbone()
        self.tokens=torch.randint(0,40,(2,9))

    def test_split_continuation(self):
        full,end=self.model.run(self.tokens)
        first,state=self.model.run(self.tokens[:,:4])
        second,split=self.model.run(self.tokens[:,4:],state)
        torch.testing.assert_close(full,torch.cat([first,second],1))
        torch.testing.assert_close(end.ssm,split.ssm)
        torch.testing.assert_close(end.conv,split.conv)

    def test_frozen_backbone_passes_state_gradient(self):
        self.model.requires_grad_(False)
        _,state=self.model.run(self.tokens[:,:4])
        for mode in ['A','D']:
            dream=Dream(self.model,mode)
            edited,changes=dream(state,self.model.embedding(self.tokens[:,:4]))
            self.assertEqual(changes.shape[1],8)
            self.assertIs(edited.conv,state.conv)
            logits,_=self.model.run(self.tokens[:,4:],edited)
            logits.square().mean().backward()
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in dream.parameters() if p.grad is not None),0)
            self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_state_injection_changes_future(self):
        from wsr.model import State
        _,state=self.model.run(self.tokens[:,:4])
        reference,_=self.model.run(self.tokens[:,4:],state)
        changed=State(state.ssm+1,state.conv)
        output,_=self.model.run(self.tokens[:,4:],changed)
        self.assertGreater(float((output-reference).detach().abs().max()),1e-5)

    def test_corruption_only_ssm(self):
        _,state=self.model.run(self.tokens)
        damaged=corrupt(state,1)
        self.assertEqual(float(damaged.ssm.detach().abs().sum()),0)
        self.assertIs(damaged.conv,state.conv)

if __name__=='__main__': unittest.main()
