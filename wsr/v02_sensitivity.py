"""Read-only gradient/causal probe of the S history pathway; no optimization."""
import argparse
import json
import statistics

import torch
from torch.nn import functional as F

from .pretrained_experiment import model_load
from .v02_data import ROOT
from .v02_train import load_run,write_json
from .v02_aux import COLORS,TRAIN_QUERIES,query_tensors


def main(seeds):
    from tokenizers import Tokenizer
    model=model_load('cpu');tokenizer=Tokenizer.from_file('artifacts/pretrained-v1/model/tokenizer.json')
    color_tokens=[tokenizer.encode(' '+c,add_special_tokens=False).ids for c in COLORS]
    assert all(len(ids)==1 for ids in color_tokens)
    ids={tokens[0] for tokens in color_tokens}
    data=torch.load(ROOT/'aux-s/validation.pt',weights_only=True,mmap=True)
    examples=[row for row in data if len(set(row['rule'][1]))>1][:32]
    path=ROOT/'S-history-sensitivity.json';result=json.loads(path.read_text()) if path.exists() else {}
    for seed in seeds:
        key=str(seed)
        if key in result:continue
        measurements={}
        for step in [0,2048]:
            dream,head,_=load_run(ROOT/'runs'/f'S-seed-{seed}'/f'checkpoint-{step:04d}.pt','cpu')
            rows=[]
            for example in examples:
                h=example['state'];tokens=example['tokens'];history=model.backbone.embedding(tokens).detach().requires_grad_(True)
                final,_=dream(h,history);q,labels=query_tensors(example['rule'],TRAIN_QUERIES,'cpu')
                logits=head(final,q);loss=F.cross_entropy(logits.flatten(0,1),labels.flatten())
                gradient=torch.autograd.grad(loss,history)[0].norm(dim=-1)[0]
                positions=[i for i,token in enumerate(tokens[0].tolist()) if token in ids]
                assert len(positions)==4
                with torch.no_grad():
                    changed=history.detach().clone();changed[:,positions]=changed[:,positions].roll(1,1)
                    edited,_=dream(h,changed);altered=head(edited,q)
                    output_difference=float((logits.softmax(-1)-altered.softmax(-1)).abs().max())
                    state_difference=float((edited-final).norm()/h.norm())
                total=float(gradient.sum())
                rows.append({'color_word_positions':positions,'history_tokens':tokens.numel(),
                    'gradient_mass_color_words':float(gradient[positions].sum())/max(total,1e-30),
                    'gradient_mass_last16':float(gradient[-16:].sum())/max(total,1e-30),
                    'history_gradient_sum_norms':total,'probability_max_change_color_permutation':output_difference,
                    'edited_state_relative_change_color_permutation':state_difference,
                    'position_gradient_norms':gradient.tolist()})
            measurements[str(step)]={'mean_color_gradient_mass':statistics.mean(r['gradient_mass_color_words'] for r in rows),
                'mean_last16_gradient_mass':statistics.mean(r['gradient_mass_last16'] for r in rows),
                'mean_probability_change_color_permutation':statistics.mean(r['probability_max_change_color_permutation'] for r in rows),
                'mean_state_relative_change_color_permutation':statistics.mean(r['edited_state_relative_change_color_permutation'] for r in rows),'rows':rows}
        result[key]={'checkpoints':measurements,'method':'32 held-out rules; differentiate LS wrt visible history embeddings; independently cyclically permute only color-word embeddings with original SSM held fixed; no parameters updated',
                     'limitation':'isolates the explicit history pathway; does not test how much information remains in the original SSM or prove absence of information'}
        write_json(path,result);print('S history sensitivity',seed,{k:{m:v for m,v in r.items() if m!='rows'} for k,r in measurements.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);a=p.parse_args();main(a.seeds)
