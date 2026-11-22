import unittest
import torch
from torch.nn import functional as F
from wsr.model import Backbone
from wsr.tasks import READ, world
from wsr.diagnose import oracle
from wsr.experiment import loss_at


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_visible_information_suffices(self):
        for family in ['memory','redundancy','rules']:
            for seed in range(12):
                data=world(seed,family,episodes=3)
                self.assertEqual(oracle(data,family),[int(y) for _,y in data])

    def test_read_position_and_batch(self):
        model=Backbone()
        tokens=torch.randint(0,24,(3,9))
        target=torch.tensor([0,1,7])
        _,state=model.run(tokens)
        loss,logits=loss_at(model,state,target)
        full,_=model.run(torch.cat([tokens,torch.full((3,1),READ)],1))
        torch.testing.assert_close(logits,full[:,-1])
        torch.testing.assert_close(loss,F.cross_entropy(full[:,-1],target))

    def test_dtype_and_zero_initialization(self):
        model=Backbone().double()
        state=model.initial(2,torch.device('cpu'))
        self.assertEqual(state.ssm.dtype,torch.float64)
        self.assertEqual(state.conv.dtype,torch.float64)
        self.assertEqual(int(state.ssm.count_nonzero()),0)
        self.assertEqual(int(state.conv.count_nonzero()),0)
        _,end=model.run(torch.tensor([[1,2],[3,0]]),state)
        self.assertEqual(end.ssm.dtype,torch.float64)

    def test_scalar_recurrence_and_state_gradient(self):
        model=Backbone(width=1,heads=1,memory=1,kernel=1).double()
        with torch.no_grad():
            model.embedding.weight.fill_(0.7)
            model.conv_weight.fill_(0.8)
            model.select.weight.zero_()
            model.select.bias.copy_(torch.tensor([0.4,0.6,-0.3],dtype=torch.float64))
            model.log_a.fill_(-0.2)
        start=model.initial(1,torch.device('cpu'))
        start.ssm=torch.tensor([[[[0.9]]]],dtype=torch.float64,requires_grad=True)
        _,end=model.run(torch.tensor([[1]]),start)
        dt=F.softplus(torch.tensor(-0.3,dtype=torch.float64))
        decay=torch.exp(-dt*torch.exp(torch.tensor(-0.2,dtype=torch.float64)))
        expected=decay*0.9+dt*F.silu(torch.tensor(0.7*0.8,dtype=torch.float64))*0.4
        torch.testing.assert_close(end.ssm.flatten()[0],expected)
        end.ssm.sum().backward()
        torch.testing.assert_close(start.ssm.grad.flatten()[0],decay)

    def test_causality(self):
        model=Backbone()
        first=torch.tensor([[1,2,3,4,5]])
        other=torch.tensor([[1,2,3,8,9]])
        a,_=model.run(first)
        b,_=model.run(other)
        torch.testing.assert_close(a[:,:3],b[:,:3])

if __name__=='__main__':
    unittest.main()
