#!/usr/bin/env python3
"""Bounded L1 observer. Reads existing evaluations; never controls robot/training."""
import argparse,fcntl,json,os,time,traceback
from pathlib import Path
from loop import read,dump,tick
p=argparse.ArgumentParser();p.add_argument('--cycle',required=True);p.add_argument('--interval',type=int,default=60);p.add_argument('--max-hours',type=float,default=72);a=p.parse_args()
cycle=Path(a.cycle).resolve();manifest=read(cycle/'manifest.json');start=time.time();old=None;failures=0
with (cycle/'observer.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while time.time()-start<a.max_hours*3600:
  if (cycle/'PAUSE').exists():
   dump(cycle/'observer.json',{'state':'paused','pid':os.getpid(),'time':time.time()});break
  signature=[]
  for item in manifest['tasks'].values():
   for name in ['episodes.jsonl','plan.json','contract.json','evaluate.py']:
    f=Path(item['source'])/name
    signature.append((str(f),f.stat().st_size,f.stat().st_mtime_ns) if f.exists() else (str(f),None,None))
  try:
   if signature!=old:
    result=tick(cycle);old=signature;failures=0
    if read(cycle/'state.json')['coverage_complete']:
     dump(cycle/'observer.json',{'state':'completed','pid':os.getpid(),'time':time.time(),**result});break
   dump(cycle/'observer.json',{'state':'observing','pid':os.getpid(),'time':time.time(),'last_snapshot':read(cycle/'latest.json')})
  except Exception as e:
   failures+=1;dump(cycle/'observer.json',{'state':'blocked' if failures>=3 else 'error','pid':os.getpid(),'error':repr(e),'consecutive_failures':failures,'time':time.time()});traceback.print_exc()
   if failures>=3:break
  time.sleep(max(10,a.interval))
 else:dump(cycle/'observer.json',{'state':'budget_exhausted','time':time.time(),'pid':os.getpid()})
