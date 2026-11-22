import unittest
import torch
from wsr.model import Backbone, State
from wsr.timescales import TARGETS, initialize, probe
from wsr.diagnose import Profile

class TimescaleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)

    def test_observer_is_passive(self):
        model=Backbone()
        tokens=torch.randint(0,40,(2,7))
        reference,state=model.run(tokens)
        seen=[]
        actual,observed=model.run(tokens,observer=lambda dt,decay,state: seen.append(decay.detach().clone()))
        torch.testing.assert_close(reference,actual,rtol=0,atol=0)
        torch.testing.assert_close(state.ssm,observed.ssm,rtol=0,atol=0)
        self.assertEqual(len(seen),7)

    def test_calibrated_survival_reaches_half_per_head(self):
        model=Backbone().double()
        # Make dt constant, so nominal half-life has an exact survival interpretation.
        with torch.no_grad():
            weight=model.select.weight.reshape(model.heads,2*model.memory+1,model.width)
            bias=model.select.bias.reshape(model.heads,2*model.memory+1)
            weight[:,-1].zero_()
            bias[:,-1].zero_()
        original={k:v.clone() for k,v in model.state_dict().items() if k!='log_a'}
        initialize(model,'multiscale')
        for k,v in original.items():
            torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
        for head,target in enumerate(TARGETS):
            zero=model.initial(1,torch.device('cpu'))
            edited=State(zero.ssm+1,zero.conv)
            tokens=torch.ones((1,int(target)),dtype=torch.long)
            _,a=model.run(tokens,zero)
            _,b=model.run(tokens,edited)
            torch.testing.assert_close((b.ssm-a.ssm)[:,head],torch.full_like(a.ssm[:,head],.5))

    def test_probe_retains_distribution_and_per_world_values(self):
        model=Backbone()
        initialize(model,'multiscale')
        summary,raw=probe(model,Profile('memory',length=4),seeds=range(123,126))
        self.assertEqual(raw['decay_rate'].shape,(3,5,4))
        self.assertEqual(len(summary['heads']),4)
        for head in summary['heads']:
            self.assertEqual(len(head['effective_half_life_per_world']),3)
            self.assertEqual(sum(head['half_life_histogram']['counts']),15)
            self.assertLessEqual(head['instant_half_life_quantiles']['p10'],head['instant_half_life_quantiles']['p90'])

if __name__=='__main__': unittest.main()
