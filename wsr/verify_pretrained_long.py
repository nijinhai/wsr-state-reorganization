"""Long-prefix segmentation and no-op numerical floor, with the actual pretrained model."""
import json
from pathlib import Path
import torch
from .pretrained_experiment import ROOT,model_load
from .pretrained_data import documents
from .pretrained_evaluate import prefix_caches,score

@torch.no_grad()
def main():
    model=model_load('cuda')
    record=documents('pg19','validation')[0]
    tokens=record['tokens'][None].to('cuda')
    cache=prefix_caches(model,record,[8192])[8192]
    alternate=model.wake(tokens[:,:8191],chunk=128)
    original_loss,_=score(model,tokens,cache,8192,512)
    alternate_loss,_=score(model,tokens,alternate,8192,512)
    noop_loss,_=score(model,tokens,cache.edit(12,cache.layers[12].ssm+0),8192,512)
    differences=[abs(a-b) for a,b in zip(original_loss,alternate_loss)]
    noops=[abs(a-b) for a,b in zip(original_loss,noop_loss)]
    result={'document':record['metadata'],'history_visible':8192,'history_processed':8191,
            'wake_chunk_sizes':[256,128],
            'ssm_max_absolute':max(float((a.ssm-b.ssm).abs().max()) for a,b in zip(cache.layers,alternate.layers)),
            'conv_max_absolute':max(float((a.conv-b.conv).abs().max()) for a,b in zip(cache.layers,alternate.layers)),
            'mean_absolute_token_nll_difference':sum(differences)/len(differences),
            'mean_signed_token_nll_difference':sum(a-b for a,b in zip(alternate_loss,original_loss))/len(original_loss),
            'no_op_max_absolute_token_nll_difference':max(noops),'no_op_mean_absolute_token_nll_difference':sum(noops)/len(noops)}
    assert result['mean_absolute_token_nll_difference']<1e-4
    assert result['no_op_max_absolute_token_nll_difference']<1e-6
    result['passed']=True
    (ROOT/'long-interface-verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
