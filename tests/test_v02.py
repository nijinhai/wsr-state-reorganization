import random
import unittest
import torch

from wsr.pretrained_model import PretrainedMamba2
from wsr.state_dream import StateDream
from wsr.v02_aux import generate_rules,answer,rule_text,TRAIN_QUERIES,HELDOUT_QUERIES,TargetBank,selected_state
from wsr.v02_intervene import rollout
from wsr.v02_train import schedule
from wsr.v02_aux import QueryDecoder,FeaturePredictor,StateReader
from torch.nn import functional as F


class V02Contracts(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(1)

    def test_rules_solvable_and_splits_disjoint(self):
        pools=generate_rules()
        self.assertFalse(set(pools['train'])&set(pools['validation']))
        self.assertFalse(set(TRAIN_QUERIES)&set(HELDOUT_QUERIES))
        for rule in pools['validation']:
            for e,h in TRAIN_QUERIES+HELDOUT_QUERIES:
                node=e
                for _ in range(h):node=rule[0][node]
                self.assertEqual(answer(rule,(e,h)),rule[1][node])
            self.assertNotIn('Starting at',rule_text(rule))

    def test_targets_use_same_document_nonoverlapping_negatives(self):
        bank=object.__new__(TargetBank)
        bank.keys=[('a',i*80) for i in range(12)]+[('b',i*80) for i in range(12)]
        bank.lookup={k:i for i,k in enumerate(bank.keys)};bank.documents={'a':list(range(12)),'b':list(range(12,24))}
        bank.features=torch.eye(24)
        candidates,label=bank.candidates(('a',160),random.Random(3))
        ids=candidates.argmax(-1).tolist()
        self.assertEqual(ids[label],2);self.assertTrue(all(i<12 for i in ids));self.assertEqual(len(set(ids)),8)

    def test_rollout_matches_original_and_interventions_affect_later_computation(self):
        dream=StateDream(memory=4,heads=4,width=4,hidden=8)
        state=torch.randn(1,4,2,4);history=torch.randn(1,5,4)
        expected,trace=dream(state,history)
        actual,other,_=rollout(dream,state,history)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for a,b in zip(trace,other):torch.testing.assert_close(a,b,rtol=0,atol=0)
        for intervention in ['reset_state','reset_memory','reverse_state']:
            final,changed,_=rollout(dream,state,history,intervention)
            torch.testing.assert_close(changed[3],trace[3],rtol=0,atol=0)
            self.assertGreater(float((final-expected).abs().max()),0)

    def test_hidden_path_and_selected_state_match_full_model(self):
        model=PretrainedMamba2(width=4,layers=14,vocab=16,state=4,headdim=2,block=4).requires_grad_(False)
        tokens=torch.randint(0,16,(1,12))
        logits,cache=model.run(tokens);hidden,other=model.run(tokens,return_hidden=True)
        torch.testing.assert_close(model.lm_head(hidden),logits,rtol=0,atol=0)
        torch.testing.assert_close(selected_state(model,tokens),cache.layers[12].ssm,rtol=0,atol=0)

    def test_schedule_paired_and_expands_coverage(self):
        rows=[{'id':f'{name}{i}','document':{'dataset':name}} for name in ['wikitext103','pg19'] for i in range(7)]
        one=schedule(rows,0,28);two=schedule(rows,0,28)
        self.assertEqual(one,two);self.assertEqual(len(set(r['id'] for r in one)),14)

    def test_shared_queries_and_contrastive_targets_have_Dream_gradients(self):
        dream=StateDream(memory=128,heads=24,width=4,hidden=8)
        state=torch.randn(1,24,2,128);history=torch.randn(1,5,4)
        calls=[];hook=dream.register_forward_hook(lambda *args:calls.append(1))
        shared,_=dream(state,history)
        decoder=QueryDecoder();predictor=FeaturePredictor()
        q=torch.tensor([[0,1,4,5]]);labels=torch.tensor([[1,2,0,3]])
        s=F.cross_entropy(decoder(shared,q).flatten(0,1),labels.flatten())
        prediction=predictor(shared,torch.tensor([[0,1,2]]))
        targets=F.normalize(torch.randn(1,3,8,768),dim=-1).detach()
        logits=torch.einsum('bdh,bdkh->bdk',prediction,targets)/.1
        p=F.cross_entropy(logits.flatten(0,1),torch.tensor([0,2,4]))
        (s+p).backward();hook.remove()
        self.assertEqual(len(calls),1)
        self.assertFalse(targets.requires_grad)
        self.assertGreater(sum(float(x.grad.abs().sum()) for x in dream.parameters() if x.grad is not None),0)
        self.assertTrue(any(x.grad is not None for x in decoder.parameters()))
        self.assertTrue(any(x.grad is not None for x in predictor.parameters()))

    def test_auxiliary_reader_gradient_in_normalization_scale_direction(self):
        reader=StateReader().double();state=torch.randn(1,24,2,128,dtype=torch.double)
        scale=torch.tensor(1.,dtype=torch.double,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda value:reader(state*value).square().sum(),(scale,)))


if __name__=='__main__':unittest.main()
