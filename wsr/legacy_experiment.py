import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .model import Backbone, Dream, State, corrupt
from .tasks import READ, world

CONFIGS = {i+1: (mode,training,damage) for i,(mode,training,damage) in enumerate(
    [(m,t,c) for m in ['A','D'] for t in ['T1','T2'] for c in [False,True]]+
    [('D','T3',False),('D','T3',True),(None,None,False),(None,None,True)])}

def loss_at(model,state,answer):
    logits,_ = model.run(torch.full((answer.shape[0],1),READ,dtype=torch.long,device=answer.device),state)
    return F.cross_entropy(logits[:,0],answer), logits[:,0]

def train_base(args):
    torch.manual_seed(args.base_seed)
    model = Backbone()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.002)
    for step in range(args.base_steps):
        state = None
        losses = []
        for history,answer in world(step,['memory','redundancy'][step%2],args.episodes):
            _,state = model.run(history,state)
            loss,_ = loss_at(model,state,answer)
            losses.append(loss)
        optimizer.zero_grad()
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
        if (step+1)%200==0:
            print(f"base step={step+1} loss={float(torch.stack(losses).mean().detach()):.4f}",flush=True)
    Path(args.checkpoint).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(),'steps':args.base_steps,'seed':args.base_seed},args.checkpoint)

def run(args):
    checkpoint = torch.load(args.checkpoint,weights_only=True)
    model = Backbone()
    model.load_state_dict(checkpoint['model'])
    model.requires_grad_(False)
    results=[]
    for number in args.configs:
      mode,training,damage = CONFIGS[number]
      for seed in args.seeds:
        torch.manual_seed(seed)
        dream = Dream(model,mode) if mode else None
        optimizer = torch.optim.Adam(dream.parameters(),lr=0.001) if dream else None
        presteps = args.denoise_steps if training=='T2' else 0
        for step in range(presteps+ (args.steps if dream else 0)):
            state=None
            history_all=[]
            losses=[]
            probes=[]
            family=['memory','redundancy'][step%2]
            for history,answer in world(10000+step,family,args.episodes):
                _,state=model.run(history,state)
                history_all.append(history)
                embeddings=model.embedding(torch.cat(history_all,1)).detach()
                original=state.detach()
                if step<presteps:
                    state,_=dream(corrupt(original),embeddings)
                    loss=F.mse_loss(state.ssm,original.ssm)
                else:
                    state,_=dream(corrupt(state) if damage else state,embeddings)
                    loss,_=loss_at(model,state,answer)
                    if training=='T3':
                        # Compare behavior on prior episode continuations in the same world.
                        for old_history,old_answer in probes:
                            _,student_state=model.run(old_history,state)
                            _,student=loss_at(model,student_state,old_answer)
                            with torch.no_grad():
                                _,teacher_state=model.run(old_history,original)
                                _,teacher=loss_at(model,teacher_state,old_answer)
                            loss=loss+0.3*F.kl_div(student.log_softmax(-1),teacher.softmax(-1),reduction='batchmean')/len(probes)
                losses.append(loss)
                state=state.detach()
                probes.append((history,answer))
            optimizer.zero_grad()
            objective=torch.stack(losses).mean()
            if not torch.isfinite(objective):
                raise RuntimeError(f"Nonfinite loss: config={number}, seed={seed}, step={step}")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(dream.parameters(),1)
            optimizer.step()
        torch.manual_seed(2000000+seed)
        metrics=[]
        for family in ['memory','redundancy','rules']:
          for index in range(args.eval_worlds):
            state=None
            histories=[]
            for episode,(history,answer) in enumerate(world(1000000+index,family,args.episodes)):
                _,state=model.run(history,state)
                histories.append(history)
                before,_=loss_at(model,state,answer)
                original=state
                if damage: state=corrupt(state)
                changes=None
                if dream:
                    state,changes=dream(state,model.embedding(torch.cat(histories,1)))
                loss,logits=loss_at(model,state,answer)
                metrics.append({'family':family,'world':index,'episode':episode,'loss':float(loss.detach()),
                    'baseline_loss':float(before.detach()),'correct':int(logits.argmax(-1)==answer),
                    'edit_norms':changes.detach().flatten(2).norm(dim=-1).flatten().tolist() if changes is not None else [],
                    'state_change':float((state.ssm-original.ssm).norm().detach())})
                state=state.detach()
        if dream:
            weights=Path(args.output).parent / f"dream-{number:02d}-seed-{seed}.pt"
            weights.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"dream":dream.state_dict(),"config":number,"seed":seed,"base_checkpoint":args.checkpoint},weights)
        entry={'torch_version':str(torch.__version__),'config':number,'seed':seed,'base':{'steps':checkpoint['steps'],'seed':checkpoint['seed']},
               'prediction_steps':args.steps if dream else 0,'denoise_steps':presteps,'metrics':metrics}
        results.append(entry)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(results,indent=2))
        print(number,seed,sum(m['correct'] for m in metrics)/len(metrics),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['base','run'])
    p.add_argument('--checkpoint',default='artifacts/base.pt')
    p.add_argument('--base-seed',type=int,default=0)
    p.add_argument('--base-steps',type=int,default=2000)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--denoise-steps',type=int,default=200)
    p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--eval-worlds',type=int,default=30)
    p.add_argument('--configs',type=int,nargs='+',choices=CONFIGS,default=[5,6,7,8,11,12])
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--output',default='artifacts/results.json')
    args=p.parse_args()
    torch.set_num_threads(1)
    (train_base if args.command=='base' else run)(args)
if __name__=='__main__': main()
