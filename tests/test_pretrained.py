import unittest
import torch
from wsr.pretrained_model import PretrainedMamba2
from wsr.state_dream import StateDream,edit_cache
from wsr.pretrained_experiment import continuation

class PretrainedContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(3)
        self.model=PretrainedMamba2(width=4,layers=2,vocab=16,state=4,headdim=2,block=4).requires_grad_(False)
        self.tokens=torch.randint(0,16,(1,14))

    def test_anchor_and_future_nonleakage(self):
        boundary=6
        cache=self.model.wake(self.tokens[:,:boundary-1])
        dream=StateDream(memory=4,heads=4,width=4,hidden=8)
        other=self.tokens.clone();other[:,boundary:]=(other[:,boundary:]+1)%16
        a,_,_=edit_cache(self.model,dream,cache,self.tokens,boundary,4,0)
        b,_,_=edit_cache(self.model,dream,cache,other,boundary,4,0)
        torch.testing.assert_close(a.layers[0].ssm,b.layers[0].ssm,rtol=0,atol=0)
        self.assertIs(a.layers[0].conv,cache.layers[0].conv)
        self.assertIs(a.layers[1],cache.layers[1])
        loss,_=continuation(self.model,self.tokens,a,boundary,4)
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in dream.parameters() if p.grad is not None),0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_zero_edit_alignment_and_cycle_continuity(self):
        boundary=6
        cache=self.model.wake(self.tokens[:,:boundary-1])
        full,_=self.model.run(self.tokens[:,:9])
        split,end=self.model.run(self.tokens[:,boundary-1:9],cache)
        torch.testing.assert_close(split,full[:,boundary-1:9])
        self.assertEqual(end.processed,9)
        # Last target at position 9 becomes the pending observed anchor for the next cycle.
        edited,_,_=edit_cache(self.model,None,end,self.tokens,10,4,0)
        next_logits,next_state=self.model.run(self.tokens[:,9:12],edited)
        reference,_=self.model.run(self.tokens[:,:12])
        torch.testing.assert_close(next_logits,reference[:,9:12])
        self.assertEqual(next_state.processed,12)

if __name__=='__main__':unittest.main()
