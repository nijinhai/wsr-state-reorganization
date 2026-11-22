"""Bounded post-v0.2a probes. No backbone or Dream optimization."""
import argparse
import hashlib
import json
import random
import re
import shutil
from pathlib import Path

import torch
from torch.nn import functional as F

from .pretrained_experiment import model_load
from .pretrained_evaluate import score, loss_summary
from .state_dream import edit_cache, history_input
from .v02_data import ROOT as OLD, LAYER, DISTANCES, digest
from .v02_train import Inputs, load_run, write_json
from .v02_aux import TargetBank, p_key, COLORS, NAMES, answer, TRAIN_QUERIES, HELDOUT_QUERIES

ROOT = Path('artifacts/wsr-v0.2a-diagnostics')
SCALES = (-.25, 0., .125, .25, .5, 1., 1.5)


def initialize():
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'protocol.json'
    protocol = {'base': 'unchanged frozen official Mamba2-130M and v0.2a checkpoints',
                'Dream_optimizer_updates': 0, 'backbone_optimizer_updates': 0,
                'amplitude': {'groups': ['B0', 'P'], 'seeds': [0, 1, 2], 'checkpoint': 2048,
                              'scales': SCALES, 'formula': 'H0 + scale * (H8 - H0)',
                              'future_tokens': 2048, 'validation_only': True,
                              'panel': 'two Wiki documents (one curve, one outside), at 512/2048; both PG books at 8192/32768',
                              'no_periodic_scan': True},
                'S_levels': ['visible text facts / position-preserving token readout',
                             'unedited SSM: original reader versus axis-preserving fixed projection',
                             'frozen S-seed0 edited SSM: identical readouts; history GRU context checked separately'],
                'S_probe': 'linear ridge to 4 link + 4 color facts, symbolic query execution on predicted facts; no Dream training',
                'P_probe': 'train-only ridge feature prediction; same-document InfoNCE evaluation; visible last64 frozen encoding versus raw-state readouts',
                'ridge_penalties': [.001, .01, .1, 1., 10.],
                'probe_model_seed': 0, 'projection_seed': 20261117,
                'test_split_used': False, 'S_plus_P': False, 'W1_W2': False}
    normalized = json.loads(json.dumps(protocol))
    if path.exists():
        assert json.loads(path.read_text()) == normalized, 'Diagnostic protocol changed'
    else:
        write_json(path, protocol)
    archive = ROOT / 'preservation-before.json'
    if not archive.exists():
        files = list((OLD / 'runs').glob('*/*.pt')) + list((OLD / 'runs').glob('*/*.json'))
        files += list((OLD / 'evaluation').glob('*/*.json'))
        files += [OLD / n for n in ['protocol.json', 'calibration.json', 'results.json', 'REPORT.md', 'FINDINGS.md', 'verification.json']]
        write_json(archive, {str(p): digest(p) for p in files})


def panel_rows(inputs):
    rows = inputs.index['validation']
    wiki = sorted({r['document']['id'] for r in rows if r['document']['dataset'] == 'wikitext103'})
    selected = {wiki[0], wiki[4]}
    return [r for r in rows if (r['document']['dataset'] == 'wikitext103' and r['document']['id'] in selected)
            or (r['document']['dataset'] == 'pg19' and r['boundary'] in [8192, 32768])]


@torch.no_grad()
def amplitude(seeds):
    model = model_load('cuda'); inputs = Inputs(model); rows = panel_rows(inputs)
    baseline = {r['id']: r for r in json.loads((OLD / 'evaluation/baseline-seed-0/lm.json').read_text())}
    for seed in seeds:
        for group in ['B0', 'P']:
            dest = ROOT / f'amplitude-{group}-seed-{seed}.json'
            records = json.loads(dest.read_text()) if dest.exists() else []
            done = {(r['id'], r['scale']) for r in records}
            dream, _, _ = load_run(OLD / 'runs' / f'{group}-seed-{seed}' / 'checkpoint-2048.pt', 'cuda')
            dream.requires_grad_(False)
            old = {r['id']: r for r in json.loads((OLD / 'evaluation' / f'{group}-seed-{seed}' / 'lm.json').read_text())}
            for row in rows:
                b = row['boundary']; tokens = inputs.tokens[row['token_path']]; cache = inputs.load(row)
                edited, _, _ = edit_cache(model, dream, cache, tokens, b, 128, LAYER)
                initial = cache.layers[LAYER].ssm; delta = edited.layers[LAYER].ssm - initial
                for scale in SCALES:
                    if (row['id'], scale) in done: continue
                    # Preserve exact FP32 endpoints instead of subtract/add roundoff.
                    h = initial if scale == 0 else edited.layers[LAYER].ssm if scale == 1 else initial + scale * delta
                    changed = cache.edit(LAYER, h)
                    assert changed.layers[LAYER].conv is cache.layers[LAYER].conv
                    assert all(changed.layers[i] is cache.layers[i] for i in range(24) if i != LAYER)
                    if scale in [0., 1.]:
                        reference = baseline[row['id']] if scale == 0 else old[row['id']]
                        values = reference['losses']; reused = True
                        # Recompute both anchors on the first panel case for every group/seed.
                        if row is rows[0]:
                            actual, _ = score(model, tokens, changed, b, len(values))
                            difference = max(abs(x-y) for x,y in zip(actual, values))
                            write_json(ROOT/f'endpoint-{group}-{seed}-{scale}.json', {'scale':scale,'max_token_loss_difference':difference,
                                       'mean_loss_difference':sum(x-y for x,y in zip(actual,values))/len(values)})
                            assert difference < 2e-5, (group,seed,scale,difference)
                    else:
                        values, _ = score(model, tokens, changed, b, min(2048, tokens.shape[1]-b)); reused = False
                    records.append({'id': row['id'], 'dataset': row['document']['dataset'], 'document': row['document']['id'],
                                    'group': group, 'seed': seed, 'history': b, 'scale': scale,
                                    'relative_edit': float((h-initial).norm()/initial.norm()),
                                    'first64_nll': sum(values[:64])/64, **loss_summary(values), 'losses': values,
                                    'reused_exact_v02_evaluation': reused})
                    write_json(dest, records)
                print('amplitude', group, seed, row['id'], flush=True)


def projection(device):
    generator = torch.Generator().manual_seed(20261117)
    return torch.randn(128, 8, generator=generator).to(device) / 8**.5


def state_axes(state, matrix):
    scale = state.square().mean((2,3), keepdim=True).sqrt().clamp_min(1e-4)
    return torch.cat([((state / scale) @ matrix).flatten(1), scale.flatten(1).log()], 1)


def standardize(train, other):
    center = train.mean(0); scale = train.std(0).clamp_min(1e-6)
    return (train-center)/scale, [(x-center)/scale for x in other]


def ridge_predict(train, targets, others, penalty):
    """Dual ridge, intercept via target centering, normalized Gram for comparable penalties."""
    train, others = standardize(train.float(), [x.float() for x in others])
    center = targets.mean(0); dimension = train.shape[1]
    gram = train @ train.T / dimension
    weights = torch.linalg.solve(gram + penalty*torch.eye(len(train), device=train.device), targets-center)
    return [x @ train.T @ weights / dimension + center for x in others]


def fact_metrics(scores, facts):
    predicted = scores.reshape(-1, 8, 4).argmax(-1)
    result = {'link_accuracy': float((predicted[:, :4] == facts[:, :4]).float().mean()),
              'color_accuracy': float((predicted[:, 4:] == facts[:, 4:]).float().mean()),
              'all8_facts_accuracy': float((predicted == facts).all(1).float().mean())}
    for name, queries in [('seen_queries', TRAIN_QUERIES), ('heldout_queries', HELDOUT_QUERIES)]:
        answers = []; labels = []
        for p, f in zip(predicted.tolist(), facts.tolist()):
            for q in queries:
                answers.append(answer((p[:4], p[4:]), q)); labels.append(answer((f[:4], f[4:]), q))
        result[name+'_symbolic_accuracy'] = sum(a==b for a,b in zip(answers,labels))/len(labels)
    return result


def select_fact_probe(features, labels, validation, validation_labels):
    rng = torch.Generator().manual_seed(9); order = torch.randperm(len(features), generator=rng)
    fit, hold = order[:768], order[768:]
    targets = F.one_hot(labels, 4).float().flatten(1)
    trials = []
    for penalty in [.001, .01, .1, 1., 10.]:
        pred = ridge_predict(features[fit], targets[fit], [features[hold]], penalty)[0]
        trials.append({'penalty': penalty, 'inner_MSE': float((pred-targets[hold]).square().mean())})
    best = min(trials, key=lambda r:r['inner_MSE'])['penalty']
    train, test = ridge_predict(features, targets, [features, validation], best)
    shuffled = targets[torch.randperm(len(targets), generator=torch.Generator().manual_seed(19))]
    shuffled_test = ridge_predict(features, shuffled, [validation], best)[0]
    small = ridge_predict(features[:32], targets[:32], [features[:32]], .00001)[0]
    return {'penalty': best, 'selection_train_rules_only': trials,
            'train': fact_metrics(train, labels), 'validation': fact_metrics(test, validation_labels),
            'shuffled_training_labels_validation': fact_metrics(shuffled_test, validation_labels),
            'overfit32_train': fact_metrics(small, labels[:32]),
            'feature_dimension': features.shape[1], 'mean_input_std': float(features.std(0).mean())}


def parse_rule(history):
    matches = re.findall(r'Location ([ABCD]) has color (red|green|blue|gold)\. Its link leads to ([ABCD])\.', history)
    assert len(matches) == 4
    mapping = [0]*4; colors = [0]*4
    for source, color, target in matches:
        mapping[NAMES.index(source)] = NAMES.index(target); colors[NAMES.index(source)] = COLORS.index(color)
    return tuple(mapping), tuple(colors)


@torch.no_grad()
def prepare_probes():
    model = model_load('cuda'); matrix = projection('cuda')
    generator = torch.Generator().manual_seed(20261117)
    token_matrix = torch.randn(768, 16, generator=generator, device='cpu').cuda()/4
    if not (ROOT/'S-features.pt').exists():
        dream, head, _ = load_run(OLD/'runs/S-seed-0/checkpoint-2048.pt', 'cuda')
        dream.requires_grad_(False); head.requires_grad_(False)
        splits = {}
        for split in ['train', 'validation']:
            examples = torch.load(OLD/'aux-s'/(split+'.pt'), weights_only=True, mmap=True)
            collected = {k: [] for k in ['raw_reader', 'edited_reader', 'raw_axes', 'edited_axes', 'history_context', 'visible_token_positions']}
            labels = []; parsed = 0
            for start in range(0, len(examples), 8):
                batch = examples[start:start+8]
                state = torch.cat([r['state'] for r in batch]).cuda()
                tokens = torch.cat([r['tokens'] for r in batch]).cuda()
                history = model.backbone.embedding(tokens)
                edited, _ = dream(state, history)
                _, context = dream.history_gru(F.silu(dream.history_project(history)))
                values = {'raw_reader': head.reader(state), 'edited_reader': head.reader(edited),
                          'raw_axes': state_axes(state, matrix), 'edited_axes': state_axes(edited, matrix),
                          'history_context': context[0], 'visible_token_positions': (history@token_matrix).flatten(1)}
                for key, value in values.items(): collected[key].append(value.cpu())
                for row in batch:
                    rule = parse_rule(row['history'])
                    assert list(rule[0]) == list(row['rule'][0]) and list(rule[1]) == list(row['rule'][1])
                    parsed += 1; labels.append(list(rule[0])+list(rule[1]))
            splits[split] = {'features': {k:torch.cat(v) for k,v in collected.items()}, 'facts':torch.tensor(labels),
                             'text_parser_exact_rules':parsed, 'samples':len(examples)}
            print('S features',split,len(examples),flush=True)
        torch.save(splits,ROOT/'S-features.pt')
    if not (ROOT/'P-features.pt').exists():
        dream, head, _ = load_run(OLD/'runs/P-seed-0/checkpoint-2048.pt','cuda');head.requires_grad_(False)
        index = json.loads((OLD/'bank/index.json').read_text())
        normalization = torch.load(OLD/'aux-p/train.pt',weights_only=True)
        center=normalization['train_center'].cuda();scale=normalization['train_scale'].cuda()
        splits={};tokens_by_path={}
        for split in ['train','validation']:
            bank=TargetBank(split,'cpu')
            rows=[r for r in index[split] if all(p_key(r,0,d) in bank.lookup for d in DISTANCES)]
            collected={k:[] for k in ['raw_reader','raw_axes','visible64']};targets=[]
            for start in range(0,len(rows),8):
                batch=rows[start:start+8];states=[];contexts=[]
                for row in batch:
                    cached=torch.load(row['path'],weights_only=True,mmap=True)
                    assert cached['processed']==row['boundary']-1
                    states.append(cached['layers'][LAYER]['ssm'])
                    if row['token_path'] not in tokens_by_path:
                        tokens_by_path[row['token_path']]=torch.load(row['token_path'],weights_only=True)['tokens']
                    b=row['boundary'];contexts.append(tokens_by_path[row['token_path']][b-64:b])
                state=torch.cat(states).cuda();context=torch.stack(contexts).cuda()
                hidden,_=model.run(context,return_hidden=True)
                visible=F.normalize((hidden.mean(1)-center)/scale,dim=-1)
                values={'raw_reader':head.reader(state),'raw_axes':state_axes(state,matrix),'visible64':visible}
                for key,value in values.items():collected[key].append(value.cpu())
                targets.extend(torch.stack([bank.features[bank.lookup[p_key(row,0,d)]] for d in DISTANCES]) for row in batch)
                if start%128==0:print('P features',split,start,'/',len(rows),flush=True)
            splits[split]={'rows':rows,'features':{k:torch.cat(v) for k,v in collected.items()},'targets':torch.stack(targets)}
        torch.save(splits,ROOT/'P-features.pt')


def s_probes():
    torch.set_num_threads(4)
    data=torch.load(ROOT/'S-features.pt',weights_only=True);train=data['train'];validation=data['validation']
    result={'levels':{},'train_rules':len(train['facts']),'validation_rules':len(validation['facts']),
            'oracle_text_parser_accuracy':1.,'oracle_symbolic_query_accuracy':1.,
            'limitations':'Only linear probes are fitted; successful decoding demonstrates accessible facts, unsuccessful decoding does not prove absence. Symbolic execution is an oracle using predicted facts, not a trained Dream query decoder.',
            'checkpoint':'S-seed-0/checkpoint-2048.pt; all Mamba/Dream/original query-head parameters frozen'}
    for name,features in train['features'].items():
        result['levels'][name]=select_fact_probe(features,train['facts'],validation['features'][name],validation['facts'])
        write_json(ROOT/'S-results.json',result)
        print('S probe',name,result['levels'][name]['validation'],flush=True)


def candidates_for(rows, split):
    bank=TargetBank(split,'cpu');candidates=[];labels=[]
    for row in rows:
        seed=int(hashlib.sha256(row['id'].encode()).hexdigest()[:8],16)
        values,label=bank.targets(row,0,random.Random(seed))
        candidates.append(values[0]);labels.append(label[0])
    return torch.stack(candidates),torch.stack(labels)


def contrastive_metrics(prediction,candidates,labels):
    prediction=F.normalize(prediction.reshape(-1,3,768),dim=-1)
    logits=torch.einsum('ndh,ndkh->ndk',prediction,candidates)/.1
    ce=F.cross_entropy(logits.flatten(0,1),labels.flatten(),reduction='none').reshape(-1,3)
    correct=(logits.argmax(-1)==labels).float()
    return {'ce':float(ce.mean()),'accuracy':float(correct.mean()),
            'per_distance':{str(d):{'ce':float(ce[:,i].mean()),'accuracy':float(correct[:,i].mean())} for i,d in enumerate(DISTANCES)},
            'row_ce':ce.mean(1).tolist(),'row_accuracy':correct.mean(1).tolist()}


def document_key(row):
    return row['document']['dataset']+'-'+row['document']['id']


def rotate_within_documents(rows):
    groups={}
    for i,row in enumerate(rows):groups.setdefault(document_key(row),[]).append(i)
    indices=list(range(len(rows)))
    for values in groups.values():
        for old,new in zip(values,values[1:]+values[:1]):indices[old]=new
    return torch.tensor(indices)


def p_probes():
    torch.set_num_threads(4)
    data=torch.load(ROOT/'P-features.pt',weights_only=True);train=data['train'];test=data['validation']
    rows=train['rows'];fit_docs=set()
    for dataset in ['wikitext103','pg19']:
        docs=sorted({document_key(r) for r in rows if r['document']['dataset']==dataset})
        random.Random(72).shuffle(docs);fit_docs.update(docs[max(2,len(docs)//5):])
    fit=torch.tensor([i for i,r in enumerate(rows) if document_key(r) in fit_docs]);hold=torch.tensor([i for i,r in enumerate(rows) if document_key(r) not in fit_docs])
    train_candidates,train_labels=candidates_for(rows,'train');test_candidates,test_labels=candidates_for(test['rows'],'validation')
    targets=train['targets'].flatten(1)
    inner_prior=F.normalize(train['targets'][fit].mean(0),dim=-1)
    prior=F.normalize(train['targets'].mean(0),dim=-1)
    balanced_prior=F.normalize(torch.stack([train['targets'][torch.tensor([r['document']['dataset']==d for r in rows])].mean(0) for d in ['wikitext103','pg19']]).mean(0),dim=-1)
    train_rotation=rotate_within_documents(rows);test_rotation=rotate_within_documents(test['rows'])
    result={'train_prefixes':len(rows),'validation_prefixes':len(test['rows']),
            'fit_documents':len(fit_docs),'inner_holdout_documents':len({document_key(rows[i]) for i in hold.tolist()}),
            'validation_documents':len({document_key(r) for r in test['rows']}),
            'test_rotation_changed_fraction':float((test_rotation!=torch.arange(len(test_rotation))).float().mean()),
            'inner_split_document_disjoint':True,'ridge':{},'controls':{},
            'description':'No Dream training. Future features only supervise readouts. Input is unedited selected SSM or last64 visible tokens encoded independently with frozen Mamba. Same-document 8 candidates, exact original distances/feature normalization/temp=.1.'}
    for name,prediction in [('uniform_logits',torch.zeros_like(test['targets'])),
                            ('prior_all_train',prior[None].expand(len(test['rows']),-1,-1)),
                            ('prior_balanced_datasets',balanced_prior[None].expand(len(test['rows']),-1,-1)),
                            ('visible64_persistence',test['features']['visible64'][:,None].expand(-1,3,-1))]:
        result['controls'][name]=contrastive_metrics(prediction,test_candidates,test_labels)
    # Oracle pipeline control, not evidence of future predictability.
    result['controls']['future_target_self_match']=contrastive_metrics(test['targets'],test_candidates,test_labels)
    for name,x in train['features'].items():
        trials=[]
        for penalty in [.001,.01,.1,1.,10.]:
            p=ridge_predict(x[fit],targets[fit],[x[hold]],penalty)[0]
            trials.append({'penalty':penalty,'inner_InfoNCE':contrastive_metrics(p,train_candidates[hold],train_labels[hold])['ce']})
        best=min(trials,key=lambda r:r['inner_InfoNCE'])['penalty']
        pred=ridge_predict(x,targets,[test['features'][name]],best)[0]
        shuffled=ridge_predict(x[train_rotation],targets,[test['features'][name]],best)[0]
        swapped=ridge_predict(x,targets,[test['features'][name][test_rotation]],best)[0]
        train_prediction=ridge_predict(x,targets,[x],best)[0]
        small=ridge_predict(x[:32],targets[:32],[x[:32]],.00001)[0]
        result['ridge'][name]={'penalty':best,'train_document_holdout_selection':trials,
                               'validation':contrastive_metrics(pred,test_candidates,test_labels),
                               'same_document_shuffled_training_pairing':contrastive_metrics(shuffled,test_candidates,test_labels),
                               'same_document_swapped_validation_history':contrastive_metrics(swapped,test_candidates,test_labels),
                               'train':contrastive_metrics(train_prediction,train_candidates,train_labels),
                               'overfit32_train':contrastive_metrics(small,train_candidates[:32],train_labels[:32]),
                               'feature_dimension':x.shape[1]}
        write_json(ROOT/'P-results.json',result)
        print('P ridge',name,'validation',result['ridge'][name]['validation']['ce'],result['ridge'][name]['validation']['accuracy'],flush=True)
    # History-free inner prior is recorded to interpret the regularization choice.
    result['inner_prior_InfoNCE']=contrastive_metrics(inner_prior[None].expand(len(hold),-1,-1),train_candidates[hold],train_labels[hold])['ce']
    result['validation_rows']=[{'id':r['id'],'document':document_key(r),'history':r['boundary']} for r in test['rows']]
    write_json(ROOT/'P-results.json',result)


def p_nonlinear():
    """Small fixed-budget InfoNCE heads; Mamba/Dream are never instantiated here."""
    torch.set_num_threads(4)
    data=torch.load(ROOT/'P-features.pt',weights_only=True);train=data['train'];test=data['validation']
    candidates,labels=candidates_for(train['rows'],'train');tc,tl=candidates_for(test['rows'],'validation')
    candidates=candidates.cuda();labels=labels.cuda();tc=tc.cuda();tl=tl.cuda()
    rotation=rotate_within_documents(train['rows']);test_rotation=rotate_within_documents(test['rows'])
    out={}
    for name in ['raw_axes','visible64']:
        x,others=standardize(train['features'][name],[test['features'][name]])
        x=x.cuda();tx=others[0].cuda()
        for pairing in ['aligned','same_document_shuffled']:
            key=name+'-'+pairing;dest=ROOT/f'P-head-{key}.pt'
            if dest.exists():
                out[key]=torch.load(dest,weights_only=True)['results'];continue
            torch.manual_seed(0)
            head=torch.nn.Sequential(torch.nn.Linear(x.shape[1],128),torch.nn.SiLU(),torch.nn.Linear(128,3*768)).cuda()
            optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.001)
            generator=torch.Generator().manual_seed(12);logs=[]
            paired=x if pairing=='aligned' else x[rotation]
            for step in range(512):
                indices=torch.randint(len(x),(32,),generator=generator).cuda()
                pred=F.normalize(head(paired[indices]).reshape(-1,3,768),dim=-1)
                logits=torch.einsum('ndh,ndkh->ndk',pred,candidates[indices])/.1
                loss=F.cross_entropy(logits.flatten(0,1),labels[indices].flatten())
                optimizer.zero_grad();loss.backward()
                gradient=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
                assert torch.isfinite(loss) and torch.isfinite(gradient)
                optimizer.step()
                if step+1 in [1,32,128,512]:logs.append({'step':step+1,'train_batch_InfoNCE':float(loss.detach()),'gradient_norm':float(gradient)})
            with torch.no_grad():
                metrics={'validation':contrastive_metrics(head(tx),tc,tl),
                         'validation_swapped_history':contrastive_metrics(head(tx[test_rotation]),tc,tl),
                         'train':contrastive_metrics(head(paired),candidates,labels),'training':logs,
                         'updates':512,'batch':32,'seed':0,'parameters':sum(p.numel() for p in head.parameters()),
                         'future_targets_are_supervision_only':True}
            torch.save({'head':{k:v.cpu() for k,v in head.state_dict().items()},'results':metrics},dest)
            out[key]=metrics;write_json(ROOT/'P-nonlinear.json',out)
            print('P nonlinear',key,metrics['validation']['ce'],metrics['validation']['accuracy'],flush=True)
    write_json(ROOT/'P-nonlinear.json',out)


def p_visible_control():
    """Positive control: decode an already visible segment, using the same feature space."""
    torch.set_num_threads(4)
    data=torch.load(ROOT/'P-features.pt',weights_only=True)
    arrays={}
    for split in ['train','validation']:
        block=data[split];bank=TargetBank(split,'cpu');candidates=[];labels=[]
        for i,row in enumerate(block['rows']):
            doc,_=p_key(row,0,128);position=row['boundary']-64
            pool=list(bank.documents[doc]);random.Random(702+i).shuffle(pool)
            chosen=[]
            for index in pool:
                pos=bank.keys[index][1]
                if abs(pos-position)<64:continue
                if all(abs(pos-bank.keys[j][1])>=64 for j in chosen):chosen.append(index)
                if len(chosen)==7:break
            assert len(chosen)==7
            values=torch.cat([block['features']['visible64'][i:i+1],bank.features[chosen]])
            order=torch.randperm(8,generator=torch.Generator().manual_seed(702+i))
            candidates.append(values[order]);labels.append(int((order==0).nonzero()[0]))
        arrays[split]=(torch.stack(candidates),torch.tensor(labels))
    def metrics(prediction,candidates,labels):
        logits=torch.einsum('nh,nkh->nk',F.normalize(prediction,dim=-1),candidates)/.1
        return {'ce':float(F.cross_entropy(logits,labels)),'accuracy':float((logits.argmax(-1)==labels).float().mean())}
    result={'target':'last64 visible tokens, same frozen encoder/normalization; not a future target',
            'penalty':.1,'description':'fixed-penalty linear positive control; candidates are seven non-overlapping same-document windows plus the visible target, randomized positions','readouts':{}}
    for name,x in data['train']['features'].items():
        p=ridge_predict(x,data['train']['features']['visible64'],[data['validation']['features'][name]],.1)[0]
        result['readouts'][name]=metrics(p,*arrays['validation'])
    result['exact_visible_identity']=metrics(data['validation']['features']['visible64'],*arrays['validation'])
    write_json(ROOT/'P-visible-control.json',result)
    print('P visible positive control',result,flush=True)


def s_query_control():
    """Readout-only positive controls; explicitly higher label draws than original S."""
    torch.set_num_threads(4)
    data=torch.load(ROOT/'S-features.pt',weights_only=True);train=data['train'];test=data['validation']
    train_q=torch.tensor([[e*3+h-1 for e,h in TRAIN_QUERIES]])
    held_q=torch.tensor([[e*3+h-1 for e,h in HELDOUT_QUERIES]])
    def labels(facts,queries):
        return torch.tensor([[answer((f[:4],f[4:]),q) for q in queries] for f in facts.tolist()])
    train_y=labels(train['facts'],TRAIN_QUERIES)
    out={}
    for name in ['raw_reader','edited_reader','oracle_facts']:
        if name=='oracle_facts':
            x=F.pad(F.one_hot(train['facts'],4).float().flatten(1),(0,96))
            tx=F.pad(F.one_hot(test['facts'],4).float().flatten(1),(0,96))
        else:x=train['features'][name];tx=test['features'][name]
        x,others=standardize(x,[tx]);tx=others[0]
        torch.manual_seed(0)
        entity=torch.nn.Embedding(4,8);hops=torch.nn.Embedding(3,8)
        head=torch.nn.Sequential(torch.nn.Linear(144,64),torch.nn.SiLU(),torch.nn.Linear(64,4))
        parameters=list(head.parameters())+list(entity.parameters())+list(hops.parameters())
        optimizer=torch.optim.Adam(parameters,lr=.001)
        generator=torch.Generator().manual_seed(21)
        def logits(features,q):
            queries=torch.cat([entity(q//3),hops(q%3)],-1).expand(len(features),-1,-1)
            return head(torch.cat([features[:,None].expand(-1,q.shape[1],-1),queries],-1))
        for step in range(512):
            indices=torch.randint(len(x),(16,),generator=generator)
            loss=F.cross_entropy(logits(x[indices],train_q).flatten(0,1),train_y[indices].flatten())
            optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():
            metrics={}
            for split,features,facts,q,qs in [('train_seen_queries',x,train['facts'],train_q,TRAIN_QUERIES),
                                            ('validation_seen_queries',tx,test['facts'],train_q,TRAIN_QUERIES),
                                            ('validation_unseen_queries',tx,test['facts'],held_q,HELDOUT_QUERIES)]:
                p=logits(features,q);y=labels(facts,qs)
                metrics[split]={'accuracy':float((p.argmax(-1)==y).float().mean()),'ce':float(F.cross_entropy(p.flatten(0,1),y.flatten()))}
        metrics.update({'optimizer_updates':512,'query_label_draws':512*16*8,'original_S_query_label_draws':2048*8,
                        'scope':'new diagnostic head only; frozen cached reader features, no Dream/reader optimization; oracle_facts are privileged positive-control facts'})
        torch.save({'head':head.state_dict(),'entity':entity.state_dict(),'hops':hops.state_dict(),'results':metrics},ROOT/f'S-query-head-{name}.pt')
        out[name]=metrics;write_json(ROOT/'S-query-control.json',out)
        print('S query control',name,metrics['validation_unseen_queries'],flush=True)


def audit():
    before=json.loads((ROOT/'preservation-before.json').read_text())
    altered=[p for p,h in before.items() if digest(p)!=h]
    assert not altered,altered
    original=json.loads((OLD/'v01-preservation.json').read_text())
    changed=[p for p,h in original.items() if digest(p)!=h]
    assert not changed,changed
    rows=[r for path in ROOT.glob('amplitude-*.json') for r in json.loads(path.read_text())]
    assert len(rows)==336
    assert len({(r['group'],r['seed'],r['id'],r['scale']) for r in rows})==336
    assert all(torch.isfinite(torch.tensor(r['losses'])).all() for r in rows)
    # Recheck the one zero endpoint that passed before the initial FP32 endpoint issue.
    zero_path=ROOT/'endpoint-B0-0-0.0.json'
    if not zero_path.exists():
        model=model_load('cuda');inputs=Inputs(model);row=panel_rows(inputs)[0]
        state=inputs.load(row);tokens=inputs.tokens[row['token_path']]
        with torch.no_grad():values,_=score(model,tokens,state,row['boundary'],2048)
        reference=next(r['losses'] for r in rows if r['group']=='B0' and r['seed']==0 and r['id']==row['id'] and r['scale']==0)
        difference=max(abs(a-b) for a,b in zip(values,reference));assert difference<2e-5
        write_json(zero_path,{'scale':0.,'max_token_loss_difference':difference,'mean_loss_difference':mean_difference(values,reference)})
    endpoints=[json.loads(p.read_text()) for p in ROOT.glob('endpoint-*.json')]
    assert len(endpoints)==12 and all(r['max_token_loss_difference']<2e-5 for r in endpoints)
    tests=Path('/tmp/wsr-small-tests.log').read_text();assert 'Ran 26 tests' in tests and 'OK' in tests
    verification={'v02_unchanged_files':len(before),'v01_unchanged_files':len(original),'altered_files':[],
                  'amplitude_records':len(rows),'finite_losses':True,'exact_endpoints_checks':len(endpoints),
                  'max_endpoint_token_loss_difference':max(r['max_token_loss_difference'] for r in endpoints),
                  'Mamba_optimizer_updates':0,'Dream_optimizer_updates':0,'unit_tests_passed':26,
                  'S_original_training_resumed':False,'test_split_used':False,'S_plus_P_W1_W2_started':False,
                  'new_lightweight_head_optimizer_updates':{'S_query_controls':3*512,'P_InfoNCE_controls':4*512},
                  'closed_form_ridge':'diagnostic fits and train-only regularization selection; no gradient updates to frozen features'}
    write_json(ROOT/'verification.json',verification)
    target=ROOT/'source-run'
    for directory in ['wsr','tests']:
        for path in Path(directory).glob('*.py'):
            dest=target/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    for name in ['WSR_V02A_DIAGNOSTICS.md','STATUS.md','pyproject.toml']:
        shutil.copy2(name,target/name)
    write_json(ROOT/'source-run-sha256.json',{str(p.relative_to(target)):digest(p) for p in target.rglob('*') if p.is_file()})
    print('Diagnostic audit passed',verification,flush=True)


def mean_difference(values,reference):return sum(a-b for a,b in zip(values,reference))/len(values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['initialize', 'amplitude', 'prepare', 'S', 'S-query', 'P', 'P-nonlinear', 'P-visible', 'audit'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0,1,2]); args = parser.parse_args()
    initialize()
    if args.stage == 'amplitude': amplitude(args.seeds)
    elif args.stage == 'prepare': prepare_probes()
    elif args.stage == 'S': s_probes()
    elif args.stage == 'P': p_probes()
    elif args.stage == 'P-nonlinear': p_nonlinear()
    elif args.stage == 'S-query': s_query_control()
    elif args.stage == 'P-visible': p_visible_control()
    elif args.stage == 'audit': audit()
