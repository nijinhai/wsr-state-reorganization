import unittest
import torch

from wsr.v02_small_diagnostics import parse_rule, ridge_predict, rotate_within_documents, state_axes
from wsr.v02_aux import rule_text


class DiagnosticContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        torch.set_num_threads(1)

    def test_parser_uses_actual_visible_text(self):
        rule=((3,2,0,1),(2,1,3,0))
        self.assertEqual(parse_rule(rule_text(rule)),rule)
        altered=rule_text(rule).replace('color blue','color red')
        self.assertEqual(parse_rule(altered)[1][0],0)

    def test_standardization_does_not_use_validation_statistics(self):
        train=torch.randn(12,4);target=train@torch.randn(4,3)
        first=torch.randn(5,4)
        a=ridge_predict(train,target,[first],.01)[0]
        b=ridge_predict(train,target,[first,torch.randn(8,4)*1e6],.01)[0]
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_ridge_recovers_a_visible_linear_signal(self):
        train=torch.randn(40,5);test=torch.randn(8,5);weight=torch.randn(5,3)
        pred=ridge_predict(train,train@weight,[test],.001)[0]
        self.assertLess(float((pred-test@weight).square().mean()),1e-5)

    def test_history_shuffle_stays_inside_document(self):
        rows=[{'document':{'dataset':'wiki','id':x}} for x in ['a','b','a','b','c']]
        rotation=rotate_within_documents(rows)
        self.assertEqual(rotation.tolist(),[2,3,0,1,4])
        for i,j in enumerate(rotation):self.assertEqual(rows[i],rows[j])

    def test_state_projection_preserves_channel_position(self):
        h=torch.randn(2,24,4,128);matrix=torch.randn(128,8)
        a=state_axes(h,matrix);b=state_axes(h.flip(2),matrix)
        self.assertGreater(float((a-b).norm()),0)
        torch.testing.assert_close(a[:,-24:],b[:,-24:])


if __name__=='__main__':unittest.main()
