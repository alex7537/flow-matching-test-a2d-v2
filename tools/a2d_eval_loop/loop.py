#!/usr/bin/env python3
"""Local diagnostic control plane; simulator/training processes remain external."""
import argparse
import collections
import datetime as dt
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

VERSION = 'a2d-loop-v0.1'
TERMINAL = {'completed', 'policy_rejected', 'reset_invalid', 'infrastructure_error'}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())
def dump(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def require(condition, message):
    if not condition: raise ValueError(message)

def bundle_receipt(path, expected):
    path = Path(path).resolve()
    require(digest(path) == expected, f'Bundle hash mismatch: {path}')
    with tarfile.open(path) as t:
        members = {}
        for m in t.getmembers():
            if not m.isfile(): continue
            name = Path(m.name).name
            require(name not in members, f'Duplicate bundle basename: {name}')
            members[name] = m
        manifest = json.load(t.extractfile(members['manifest.json']))
        require(all(n in members for n in ['ckpt.pt','config.yaml','norm_stats.json']), 'Missing runtime member')
        files = manifest.get('files', {})
        require(all(n in files for n in ['ckpt.pt','config.yaml','norm_stats.json']), 'Missing internal checksums')
        for name, spec in files.items():
            require(name in members, f'Missing bundle member {name}')
            require(members[name].size == spec['bytes'], f'Internal size mismatch: {name}')
            h = hashlib.sha256()
            with t.extractfile(members[name]) as f:
                for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
            require(h.hexdigest() == spec['sha256'], f'Internal hash mismatch: {name}')
    return {'path':str(path),'sha256':expected,'internal_hashes_verified':True,'manifest':manifest,
            'training_run':manifest.get('source_checkpoint'),
            'limitations':['Manifest verifies declared provenance, not training loader semantics or input parity.']}

def wilson(k, n):
    if not n: return None
    z=1.95996398454; p=k/n; den=1+z*z/n
    mid=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,mid-half),min(1,mid+half)]

def validate_plan(p):
    require(p['episodes_per_configuration']>0, 'Positive episode budget required')
    require(p['modes']==['step_gt'] and p['max_steps']==300 and p['horizon']==16,
            'v0 adapter only supports step_gt/H16/300; use another version for different protocols')
    require(p['policy_seed']==42 and p['scene_seeds']==list(range(42,42+p['episodes_per_configuration'])), 'Unsupported seed protocol')
    require('5 action observations' in p['metric'] and '0.05' in p['metric'], 'Unknown metric declaration')
    require(set(p['models'])==set(p['expected_hashes']), 'Model/hash keys differ')

def init_cycle(cycle, taskdirs, hypothesis, parent=None):
    cycle=Path(cycle).resolve();require(not cycle.exists(), 'Cycle already exists; use tick or choose a new cycle')
    tasks={};receipts={}
    for td in taskdirs:
        td=Path(td).resolve();p=read(td/'plan.json');validate_plan(p)
        task=p['scene'];require(task not in tasks,'Duplicate scene')
        tasks[task]={'source':str(td),'plan':p,'plan_sha256':digest(td/'plan.json'),
                     'evaluator_sha256':digest(td/'evaluate.py')}
        for name,path in p['models'].items():
            sha=p['expected_hashes'][name]
            if name in receipts: require(receipts[name]['sha256']==sha,'Same policy ID has different bundle hashes')
            else: receipts[name]=bundle_receipt(path,sha)
    require(len(tasks)>0,'No tasks')
    cycle.mkdir(parents=True)
    dump(cycle/'manifest.json',{'schema_version':1,'loop_version':VERSION,'loop_code_sha256':digest(__file__),
        'cycle_id':cycle.name,'parent_cycle':str(Path(parent).resolve()) if parent else None,
        'hypothesis':hypothesis,'role':'diagnostic','autonomy':'L1: inspect/report/local-state',
        'backend':'existing-a2d-grpc-jsonl-readonly','tasks':tasks,'models':receipts,
        'metric':'ever-contact-lift-5cm-5-action-observations-v1',
        'denominator':'completed + policy_rejected; infrastructure/reset excluded and reported',
        'comparison_axes':['model/data/training provenance','task'],
        'qualification':'No automatic promotion: layout replay, training semantics and prospective acceptance gates not certified',
        'training_auto_launch':False})
    dump(cycle/'state.json',{'phase':'evaluate','state':'registered'})
    return tick(cycle)

def aggregate(rows, plan):
    validate_plan(plan);seen=set();groups=collections.defaultdict(list)
    for r in rows:
        require(r.get('model') in plan['models'], 'Unexpected model')
        ep=r.get('episode'); require(type(ep) is int and 0<=ep<plan['episodes_per_configuration'],'Unexpected episode index')
        key=(r['model'],r.get('mode'),ep);require(key not in seen,'Duplicate episode identity');seen.add(key)
        require(r.get('mode')=='step_gt','Mode changed')
        require(r.get('scene_seed')==plan['scene_seeds'][ep] and r.get('policy_seed')==42,'Seed identity mismatch')
        require(r.get('outcome') in TERMINAL,'Nonterminal outcome in episode log')
        require(type(r.get('success')) is bool,'Success must be Boolean')
        require(type(r.get('max_streak')) is int and r['max_streak']>=0,'Invalid streak')
        require(r['success']==(r['max_streak']>=5),'Saved success/streak contradiction')
        for keynum in ['max_lift','final_lift','elapsed_s']:
            val=r.get(keynum);require(val is None or (type(val) in [int,float] and math.isfinite(val)),f'Nonfinite {keynum}')
        require(type(r.get('executed')) is int and 0<=r['executed']<=300,'Invalid step count')
        if r['outcome']=='completed':require(r['executed']==300,'Completed episode missing actions')
        groups[r['model']].append(r)
    result={}
    for model in plan['models']:
        rs=groups[model];valid=[r for r in rs if r['outcome'] in ['completed','policy_rejected']]
        success=sum(r['success'] for r in valid);fail=collections.Counter()
        for r in valid:
            if not r['success']:
                bucket = ('no_valid_multifinger_contact' if r.get('contacts',0)==0 else
                          'no_simultaneous_contact_lift' if r['max_streak']==0 else 'persistence_under_5')
                fail[bucket] += 1
        result[model]={'attempts':len(rs),'expected':plan['episodes_per_configuration'],'outcomes':dict(collections.Counter(r['outcome'] for r in rs)),
            'valid':len(valid),'invalid':len(rs)-len(valid),'successes':success,
            'success_rate':success/len(valid) if valid else None,'wilson95':wilson(success,len(valid)),
            'failure_counts':dict(fail),'coverage_complete':len(rs)==plan['episodes_per_configuration']}
    return result

def snapshot_rows(path):
    data=Path(path).read_bytes() if Path(path).exists() else b''
    # An append-only writer may be between write and newline; omit that tail explicitly.
    boundary=data.rfind(b'\n')+1;committed=data[:boundary]
    return committed,[json.loads(l) for l in committed.splitlines() if l.strip()],len(data)-boundary

def tick(cycle):
    cycle=Path(cycle).resolve()
    with (cycle/'tick.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (cycle/'PAUSE').exists(): return {'state':'paused'}
        m=read(cycle/'manifest.json');results={};evidence={};problems=[];tails=0
        require(m['loop_code_sha256']==digest(__file__),'Loop implementation changed: create a new cycle')
        snap=cycle/'snapshots'/dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');snap.mkdir(parents=True)
        for task,item in m['tasks'].items():
            src=Path(item['source'])
            require(digest(src/'plan.json')==item['plan_sha256'],'Source plan drift')
            require(digest(src/'evaluate.py')==item['evaluator_sha256'],'Evaluator drift')
            data,rows,tail=snapshot_rows(src/'episodes.jsonl');tails+=tail
            (snap/f'{task}.jsonl').write_bytes(data)
            results[task]=aggregate(rows,item['plan'])
            contract=read(src/'contract.json') if (src/'contract.json').exists() else None
            if rows and contract is None:problems.append(f'{task}: missing runtime contract')
            if contract:
                for key in ['models','max_steps','horizon','policy_seed','scene','metric','expected_hashes']:
                    require(contract.get(key)==item['plan'].get(key),f'Runtime contract mismatch: {task}/{key}')
                if contract.get('model_hashes')!=item['plan']['expected_hashes']:problems.append(f'{task}: runtime model hashes missing/mismatch')
                dump(snap/f'{task}_runtime_contract.json',contract)
            evidence[task]={'source':str(src),'snapshot_sha256':hashlib.sha256(data).hexdigest(),'rows':len(rows),'incomplete_tail_bytes':tail}
        complete=all(g['coverage_complete'] for gs in results.values() for g in gs.values()) and not tails
        if problems:decision={'decision':'retry-eval','phase':'evaluate','reason':problems,'ready_for_training':False}
        elif not complete:decision={'decision':'wait','phase':'evaluate','reason':['Required task/model attempt coverage is incomplete'],'ready_for_training':False}
        else:decision={'decision':'new-experiment','phase':'evaluate','reason':['Diagnostic evaluation complete; causal failure review and layout/replay calibration precede a controlled training change'],'ready_for_training':False}
        decision.update(scientifically_qualified=False,automatic_training=False)
        next_task={'status':'draft' if complete else 'blocked_until_eval_complete','parent_cycle':m['cycle_id'],
            'evidence_snapshot':str(snap),'return_phase':decision['phase'],'hypothesis':None,'one_controlled_change':None,
            'training_job':None,'auto_submit':False,'requirements':['Review per-task failure buckets and representative videos','Resolve action-label/input parity conflicts','Freeze one hypothesis, data/split hashes and comparable optimizer-step budget','Preserve same evaluation contract; use a new cycle ID'],
            'note':'This is a handoff draft, not an authorized or submitted training job.'}
        payload={'loop_version':VERSION,'role':'diagnostic','coverage_complete':complete,'results':results,'evidence':evidence,
                 'limitations':['Seeds are not restored layout IDs','Intervals are descriptive Wilson intervals; shared simulator episodes need not be independent','Policy rejection guards differ','No sealed holdout or automatic promotion rule','Metrics validated from episode summaries, not every physics frame']}
        dump(snap/'aggregate.json',payload);dump(snap/'decision.json',decision);dump(snap/'next_experiment.json',next_task)
        lines=['# A2D 最小评测闭环',f"\nCycle：{m['cycle_id']} · 决策：{decision['decision']} · diagnostic-only",'\n| 场景 | 模型 | 尝试/计划 | 成功/有效 | 无效 | 95% Wilson区间 |','|---|---|---:|---:|---:|---|']
        for task,gs in results.items():
            for model,g in gs.items():
                ci=g['wilson95'];c=f'{ci[0]:.1%}–{ci[1]:.1%}' if ci else '无有效样本'
                lines.append(f"| {task} | {model} | {g['attempts']}/{g['expected']} | {g['successes']}/{g['valid']} | {g['invalid']} | {c} |")
        lines+=['\n成功：曾多指接触＋抬升5cm连续5个动作观测；policy_rejected保留有效分母。区间仅描述性，不是已校准的模型显著性检验。','\n下一步：'+ '；'.join(decision['reason']),'\n下一轮任务单：next_experiment.json（草稿，不会自动训练）。完整快照：'+str(snap)]
        (snap/'REPORT.md').write_text('\n'.join(lines)+'\n')
        dump(cycle/'latest.json',{'snapshot':str(snap),'decision':decision['decision']})
        dump(cycle/'state.json',{'phase':decision['phase'],'state':decision['decision'],'coverage_complete':complete,'snapshot':str(snap)})
        with (cycle/'events.jsonl').open('a') as f:f.write(json.dumps({'event':'evaluation_snapshot','snapshot':str(snap),'decision':decision['decision']})+'\n')
        return {'cycle':str(cycle),'snapshot':str(snap),'decision':decision['decision']}

def pull(host,port,remote,sha,dest):
    require(re.fullmatch(r'[A-Za-z0-9_.@-]+',host) and not host.startswith('-'),'Invalid SSH host')
    require(remote.startswith('/') and '\n' not in remote,'Remote file must be an absolute path')
    require(re.fullmatch('[0-9a-f]{64}',sha),'Expected SHA256 required')
    dest=Path(dest).resolve();require(not dest.exists(),'Destination exists');dest.parent.mkdir(parents=True,exist_ok=True)
    part=dest.with_suffix(dest.suffix+'.partial');require(not part.exists(),'Partial file exists; inspect before retry')
    try:
        with part.open('xb') as f:
            subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p',str(port),host,'cat -- '+shlex.quote(remote)],stdout=f,check=True,timeout=900)
        receipt=bundle_receipt(part,sha);part.replace(dest);receipt['path']=str(dest);dump(dest.with_suffix(dest.suffix+'.receipt.json'),receipt)
    except BaseException:
        part.unlink(missing_ok=True);raise
    return receipt

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('init');a.add_argument('--cycle',required=True);a.add_argument('--task-dir',action='append',required=True);a.add_argument('--hypothesis',required=True);a.add_argument('--parent')
    a=sub.add_parser('tick');a.add_argument('--cycle',required=True)
    a=sub.add_parser('pull');a.add_argument('--host',required=True);a.add_argument('--port',type=int,default=22);a.add_argument('--remote',required=True);a.add_argument('--sha256',required=True);a.add_argument('--dest',required=True)
    a=p.parse_args()
    if a.cmd=='init':out=init_cycle(a.cycle,a.task_dir,a.hypothesis,a.parent)
    elif a.cmd=='tick':out=tick(a.cycle)
    else:out=pull(a.host,a.port,a.remote,a.sha256,a.dest)
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
