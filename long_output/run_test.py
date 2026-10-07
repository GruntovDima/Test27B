#!/usr/bin/env python3
"""Start an installed HQ_TEST server, run one quality wave, stop only that server."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

import long_quality as q


def command(model, port):
    return ['vllm','serve',str(model),'--host','127.0.0.1','--port',str(port),
            '--served-model-name','qwen27b','--tensor-parallel-size','4','--max-num-seqs','4',
            '--max-model-len','20480','--max-num-batched-tokens','1280',
            '--gpu-memory-utilization','0.90','--dtype','float16',
            '--mamba-ssm-cache-dtype','float16','--language-model-only','--seed','42',
            '--enable-chunked-prefill','--no-enable-prefix-caching','--async-scheduling',
            '--load-format','safetensors','--compilation-config',
            '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}',
            '--additional-config','{"enable_cpu_binding":false,"ascend_compilation_config":'
            '{"enable_npugraph_ex":false,"fuse_norm_quant":false,"fuse_qknorm_rope":true,'
            '"enable_static_kernel":false,"fuse_muls_add":true}}']


def metadata(model, pack, cmd, revision):
    versions = {name: importlib.metadata.version(name) for name in ('vllm','vllm-ascend','torch-npu')}
    cann = [Path('/usr/local/Ascend/ascend-toolkit/latest/opp/version.info')]
    if os.environ.get('ASCEND_HOME_PATH'):
        cann.insert(0,Path(os.environ['ASCEND_HOME_PATH'])/'opp/version.info')
    version_file = next((path for path in cann if path.is_file()),None)
    if version_file is None:
        raise ValueError('CANN version file unavailable: source CANN set_env.sh first')
    q.verify_tokenizer(pack, model)
    configs = {path.name:q.hashlib.sha256(path.read_bytes()).hexdigest() for path in model.glob('*.json')}
    shards = [dict(name=path.name,bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
              for path in sorted(model.glob('*.safetensors'))]
    if not shards:
        raise ValueError('No safetensors weights in model directory')
    return dict(weights=dict(path=str(model),configs_sha256=configs,shards_stat=shards,
                             full_weight_hashes_verified=False),tokenizer=pack['tokenizer_assets'],
                vllm=versions['vllm'],vllm_ascend=dict(package_version=versions['vllm-ascend'],
                revision_user_supplied=revision,source_sha_verified=False),torch_npu=versions['torch-npu'],
                cann=version_file.read_text(),dtype_quantization='float16; checkpoint quantization unchanged',
                tp=4,devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES','runtime default; user must check devices'),
                prefix_caching=False,server_command=cmd)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path',required=True,type=Path)
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--revision',default=None,help='Actual tested vllm-ascend SHA, user-attested')
    parser.add_argument('--rounds',type=int,default=1)
    parser.add_argument('--out-dir',type=Path,default=None)
    args=parser.parse_args()
    if args.rounds<1:parser.error('--rounds must be positive')
    model=args.model_path.resolve()
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',args.port))==0:
            raise ValueError('Port occupied; existing server will not be reused or stopped')
    if os.environ.get('VLLM_LMHEAD_PRUNE_PACK') or os.environ.get('LMHEAD_PRUNE_PACK'):
        raise ValueError('Pruning must be disabled for this test')
    pack=q.load_pack(q.ROOT/'fixtures/pack.json')
    cmd=command(model,args.port)
    provenance=metadata(model,pack,cmd,args.revision)
    out=args.out_dir or q.ROOT/'runs'/f'{time.strftime("%Y%m%d-%H%M%S")}-{os.getpid()}'
    out.mkdir(parents=True,exist_ok=False)
    q.save(out/'metadata.json',provenance)
    print('Result directory:',out,flush=True)
    print('TP4 / concurrency4, each input8192, natural output target>=8192, max12288. No warmup.',flush=True)
    print('Server source SHA is user-attested, not proven by package version.',flush=True)
    env=os.environ.copy();env['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    env.setdefault('OMP_NUM_THREADS','1')
    server=None
    try:
        with (out/'server.log').open('x') as log:
            server=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            deadline=time.monotonic()+1800
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Server failed ({server.returncode}); inspect {out}/server.log')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health',timeout=3) as response:
                        if response.status==200:break
                except Exception:
                    if time.monotonic()>deadline:raise TimeoutError('Server startup exceeded30minutes')
                    time.sleep(5)
            print('Server ready. Starting four streaming requests.',flush=True)
            return subprocess.call([sys.executable,str(q.ROOT/'long_quality.py'),'run',
                '--model','qwen27b','--tag','HQ_TEST','--base-url',f'http://127.0.0.1:{args.port}',
                '--metadata',str(out/'metadata.json'),'--rounds',str(args.rounds),
                '--progress-dir',str(out/'progress'),'--out',str(out/'run.json')])
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:server.wait(timeout=120)
            except subprocess.TimeoutExpired:
                print('Own server shutdown timed out. Check its PID; no foreign processes/devices reset.',file=sys.stderr)


if __name__=='__main__':
    raise SystemExit(main())
