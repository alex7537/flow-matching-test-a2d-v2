#!/usr/bin/env python3
"""Read-only evaluator observer; atomic local/Obsidian reports, never controls rollout."""
import argparse, collections, datetime, fcntl, hashlib, json, os, time
from pathlib import Path

def atomic(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(text);tmp.replace(p)

def read_json(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}

def process_token(pid):
    try:
        fields=Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except (OSError,ValueError):return None

def overview(plans):
    lines=['## 本轮目标与参数', '', '| 项目 | 配置 |', '|---|---|']
    def row(label,value):
        clean=str(value).replace('|','／').replace('\n',' ')
        lines.append(f'| {label} | {clean} |')
    for scene,p in plans.items():
        prefix=scene+' · ' if len(plans)>1 else ''
        quota=p.get('target_failures_per_model');count=p.get('episodes_per_configuration','未记录')
        row(prefix+'目标',f"每模型收集 {quota} 次完整失败，最多尝试 {count} 次" if quota else f"比较模型抓取表现，每模型 × 执行模式计划 {count} 次")
        row(prefix+'模型','、'.join(p.get('models',{})))
        display=p.get('server_display','未记录')
        if str(display).startswith('headless'):display='headless'
        row(prefix+'场景',f"{p.get('scene',scene)}；{display}")
        modes=[{'step_gt':'逐步 RPC＋GT（动作逐个下发）','batch16':'合并16动作 RPC'}.get(m,m) for m in p.get('modes',['step_gt'])]
        row(prefix+'执行', '、'.join(modes)+f"；H={p.get('horizon','未记录')}；每轮最多 {p.get('max_steps','未记录')} 个动作")
        seeds=p.get('scene_seeds',[])
        seed_text=f"请求场景 seed {seeds[0]}…{seeds[-1]}（{len(seeds)} 个，详见计划）" if seeds else '场景 seed 未记录'
        row(prefix+'种子',f"模型 seed={p.get('policy_seed','未记录')}；{seed_text}")
        metric=p.get('metric','未记录')
        if metric=='ever lift>=0.05m and thumb + two other fingers >1e-4N for 5 action observations; report diagnostic, not fixed duration':
            metric='拇指＋至少另外两指接触力>1e-4 N，且抬升≥5 cm，连续5个动作后采样；之后掉落不扣分'
        row(prefix+'成功判据',metric)
        cameras=p.get('video_cameras')
        if cameras:
            row(prefix+'视频','＋'.join({'rgb_head':'头部','rgb_right_hand':'右腕'}.get(c,c) for c in cameras)+('；仅完整失败保留；MP4＋GIF' if quota else '；保存策略见计划'))
        row(prefix+'初态', '布局恢复已启用（是否配对须看验收）' if p.get('layout_replay') is True else '未启用布局恢复；相同 seed 不保证实际初态一致' if p.get('layout_replay') is False else '布局恢复状态未记录；相同 seed 不保证初态一致')
    lines+=['', '参数来自各场景 `plan.json`；修改目标或参数时新建批次，历史报告保留原配置。', '']
    return lines

def report(batch, vault, owner_alive=True):
    records=[];warnings=[];evidence={};plans={};statuses={};seen=set()
    for plan_file in sorted(batch.glob('*/plan.json')):
        scene=plan_file.parent.name;plan=read_json(plan_file);plans[scene]=plan
        statuses[scene]=read_json(plan_file.parent/'status.json')
        ledger=plan_file.parent/'episodes.jsonl';raw=ledger.read_bytes() if ledger.exists() else b''
        evidence[scene]={'ledger_sha256':hashlib.sha256(raw).hexdigest(),'plan_sha256':hashlib.sha256(plan_file.read_bytes()).hexdigest(),'ledger':str(ledger)}
        for n,line in enumerate(raw.splitlines(),1):
            try:r=json.loads(line)
            except (ValueError,UnicodeError):warnings.append(f'{scene} 第 {n} 行未完整写入或损坏，暂未计入');continue
            key=(scene,r.get('model'),r.get('mode'),r.get('episode'))
            if key in seen:warnings.append(f'重复终态记录：{key}，不重复计数');continue
            seen.add(key);records.append(dict(r,scene=scene))
    expected={(s,m,mode,e) for s,p in plans.items() for m in p['models'] for mode in p.get('modes',['step_gt']) for e in range(p['episodes_per_configuration'])}
    coverage={(r['scene'],r['model'],r['mode'],r['episode']) for r in records}
    complete=bool(expected) and coverage==expected and not warnings
    queue=read_json(batch/'queue_status.json');qstate=queue.get('state')
    state='已完成' if complete else '异常停止／未完成' if qstate in ['blocked','failed'] or not owner_alive else '暂停' if qstate=='paused' else '进行中'
    if qstate=='paused':state='暂停'
    if qstate=='awaiting_human_review':state='采集结束／等待确认'
    if qstate=='waiting_previous_batch' and owner_alive:state='排队等待前一批次'
    now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    rows=[]
    text=[f'# {batch.name}', '']+overview(plans)+[f'状态：**{state}**｜已记录 {len(records)}/{len(expected)} 次｜更新：{now}', '', '| 场景 | 模型 | 已记录/计划 | 成功/有效 | 成功率 | 无效 | 失败现象：无接触/未达5cm/持续不足 |','|---|---|---:|---:|---:|---:|---|']
    for scene,p in plans.items():
        for model in p['models']:
            rs=[r for r in records if r['scene']==scene and r['model']==model]
            valid=[r for r in rs if r.get('outcome') in ['completed','policy_rejected']]
            success=sum(bool(r.get('success')) for r in valid);failed=[r for r in valid if not r.get('success')]
            cats=collections.Counter('无接触' if not r.get('contacts') else '未达5cm' if (r.get('max_lift') or 0)<.05 else '持续不足' for r in failed)
            rate=f'{success/len(valid):.1%}' if valid else '—';invalid=len(rs)-len(valid)
            text.append(f"| {scene} | {model} | {len(rs)}/{p['episodes_per_configuration']} | {success}/{len(valid)} | {rate} | {invalid} | {cats['无接触']}/{cats['未达5cm']}/{cats['持续不足']} |")
            rows.append(dict(scene=scene,model=model,attempts=len(rs),valid=len(valid),successes=success,invalid=invalid,failure_observations=dict(cats)))
    text+=['','成功按本批次原始记录口径；无效单列。失败分类是粗略观测现象，不代表已证实的因果。未结束回合不并入成功率。']
    for scene,p in plans.items():
        if p.get('target_failures_per_model'):
            text+=['',f"失败视频采集：每模型目标 {p['target_failures_per_model']} 次，计划尝试数为上限；定向采集不能作为无偏成功率比较。"]

        current=statuses[scene].get('current')
        if not complete and current:text.append(f"最近进度：{current.get('model')} / 回合 {current.get('episode')} / {current.get('executed')} 动作（非新增终态）。")
    partial=[]
    for f in batch.glob('*/*/[0-9]*/samples.jsonl'):
        if not (f.parent/'result.json').exists():partial.append(str(f))
    if partial:text+=['',f'存在 {len(partial)} 个尚无 result.json 的动作记录目录；保留为中断／进行中证据，不强行记为失败。']
    if warnings:text+=['','记录校验提示：'+'；'.join(warnings)]
    text+=['',f'原始批次：`{batch}`', '配置与哈希：各场景 `plan.json`、`contract.json`；逐轮证据：`episodes.jsonl`、各回合 `samples.jsonl`。']
    for scene,p in plans.items():
        text+=['']+[f"- {m}：`{path}`；SHA256 `{p.get('expected_hashes',{}).get(m,'未记录')}`" for m,path in p['models'].items()]
    content='\n'.join(text)+'\n';atomic(batch/'AUTO_REPORT.md',content)
    note='自动报告_'+batch.name+'.md';atomic(vault/note,content)
    with (vault/'.auto-report-index.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        atomic(vault/'00_自动测试报告索引.md','# 自动测试报告\n\n每个批次独立更新；完成或中断时保存最终已落盘结果。\n\n'+'\n'.join(f'- [[{p.stem}]]' for p in sorted(vault.glob('自动报告_*.md')))+'\n')
    snapshot=dict(updated_at=now,state=state,owner_alive=owner_alive,complete=complete,recorded=len(records),planned=len(expected),warnings=warnings,rows=rows,evidence=evidence,partial_samples=partial,obsidian_note=str(vault/note))
    atomic(batch/'auto_report_status.json',json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    return snapshot

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch',type=Path,required=True);ap.add_argument('--vault',type=Path,required=True);ap.add_argument('--once',action='store_true');ap.add_argument('--interval',type=float,default=60);args=ap.parse_args()
    if not args.batch.is_dir() or not args.vault.is_dir():raise SystemExit('Batch/vault must exist')
    with (args.batch/'auto_report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pid=int((args.batch/'orchestrator.pid').read_text());token=process_token(pid)
        while True:
            alive=token is not None and process_token(pid)==token
            try:
                result=report(args.batch,args.vault,alive)
                print(json.dumps({'time':result['updated_at'],'state':result['state'],'recorded':result['recorded']},ensure_ascii=False),flush=True)
                if args.once or result['complete'] or not alive or result['state']=='异常停止／未完成':break
            except Exception as e:
                print(f'Report sync failed; retrying: {e!r}',flush=True)
                if args.once:raise
            time.sleep(args.interval)
if __name__=='__main__':main()
