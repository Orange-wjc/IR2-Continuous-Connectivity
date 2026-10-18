#!/usr/bin/env python3
"""Migrate an existing full checkpoint losslessly, with an address-space cap."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import time
import torch
from finite_contact.memory import load_checkpoint, compact_state, semantic_digest


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(2**20),b''):digest.update(block)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-virtual-gib',type=int,default=22)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Migration requires a new output path')
    limit=args.max_virtual_gib*2**30
    resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
    torch.set_num_threads(1)
    started=time.monotonic()
    print('Streaming legacy checkpoint without the full pickle input buffer',flush=True)
    state=load_checkpoint(args.input)
    before=semantic_digest(state)
    metadata={'episode':state['episode'],'updates':state['learner']['updates'],
              'replay_transitions':len(state.get('replay',[])),
              'success_episodes':len(state.get('success_episodes',[]))}
    print(json.dumps(metadata),flush=True)
    encoding=compact_state(state)
    assert semantic_digest(state)==before,'Logical state changed during compression'
    gc.collect()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.tmp')
    torch.save(state,temporary,pickle_protocol=4)
    del state;gc.collect()
    restored=load_checkpoint(temporary)
    assert semantic_digest(restored)==before,'Serialized roundtrip changed logical state'
    del restored;gc.collect()
    temporary.replace(args.output)
    audit={'status':'passed','source':str(args.input.resolve()),'output':str(args.output.resolve()),
           'source_sha256':sha256(args.input),'output_sha256':sha256(args.output),
           'logical_sha256_before_and_after':before,**metadata,**encoding,
           'source_file_bytes':args.input.stat().st_size,'output_file_bytes':args.output.stat().st_size,
           'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           'virtual_memory_cap_gib':args.max_virtual_gib,'wall_seconds':time.monotonic()-started,
           'unchanged':'All networks, Adam states, RNG states, config, replay order/values, success ordering/values, episode and update counters.'}
    args.output.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
