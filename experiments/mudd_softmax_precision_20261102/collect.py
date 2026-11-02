from pathlib import Path
import argparse,concurrent.futures,json,os,subprocess
import tensorstore as ts
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args();m=json.loads(Path(a.manifest).read_text());root=Path(m['local_DATA']);root.mkdir(parents=True,exist_ok=True)
os.environ['GOOGLE_AUTH_TOKEN_FOR_TESTING']=subprocess.check_output(['/home/mengqy/google-cloud-sdk/bin/gcloud','auth','print-access-token'],text=True).strip()
store=ts.KvStore.open(m['gcs_root']+'/').result()
def fetch(key):
 r=store.read(key).result();assert r.state=='value',key;f=root/key;f.parent.mkdir(exist_ok=True,parents=True);tmp=f.with_suffix(f.suffix+'.tmp');tmp.write_bytes(r.value);tmp.replace(f)
fetch('complete.json');d=json.loads((root/'complete.json').read_text());assert d['cases']==64 and d['source_commit']==m['commit']
keys=[f'old/{step}/{name}' for step in m['steps'] for name in ['complete.json']+[f'sequence_{i:03d}.{ext}' for i in range(32) for ext in ['npz','json']]]
# Remote layout contains old/; local probe and analyzer layout is step/.
def mapped_fetch(key):
 r=store.read(key).result();assert r.state=='value',key;f=root/key.removeprefix('old/');f.parent.mkdir(exist_ok=True,parents=True);tmp=f.with_suffix(f.suffix+'.tmp');tmp.write_bytes(r.value);tmp.replace(f)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(mapped_fetch,keys))
print('SOFTMAX_64_CASES_COLLECTED',flush=True)
