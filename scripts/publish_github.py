"""Publish this audited project/release using existing Git Credential Manager auth.

Explicit user authorization is required before invoking. No credentials are logged or saved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import httpx

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',default='superembedding')
    ap.add_argument('--tag',default='v0.1.0-fashion2400')
    args=ap.parse_args()
    credential=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',
        text=True,capture_output=True,check=True,env={**os.environ,'GIT_TERMINAL_PROMPT':'0','GCM_INTERACTIVE':'never'})
    fields=dict(line.split('=',1) for line in credential.stdout.splitlines() if '=' in line)
    client=httpx.Client(headers={'Authorization':'Bearer '+fields['password'],
                                 'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'},timeout=120)
    api='https://api.github.com'
    response=client.get(api+'/user');response.raise_for_status();owner=response.json()['login']
    repo=f'{owner}/{args.repo}'
    response=client.get(api+'/repos/'+repo)
    if response.status_code==404:
        response=client.post(api+'/user/repos',json={'name':args.repo,'private':False,'auto_init':False,
            'description':'Reusable attribute embeddings from OmniJev: reproducible architecture comparisons and training on an 8GB GPU.'})
        response.raise_for_status()
    else:
        response.raise_for_status()
        # Do not modify an unrelated repository accidentally.
        existing=subprocess.run(['git','remote','get-url','origin'],cwd=ROOT,text=True,capture_output=True)
        if existing.returncode or existing.stdout.strip().removesuffix('.git')!=f'https://github.com/{repo}':
            raise RuntimeError('Repository already exists; inspect destination before publishing')
    remotes=subprocess.check_output(['git','remote'],cwd=ROOT,text=True).split()
    if 'origin' not in remotes: subprocess.run(['git','remote','add','origin',f'https://github.com/{repo}.git'],cwd=ROOT,check=True)
    subprocess.run(['git','push','-u','origin','main'],cwd=ROOT,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    response=client.get(f'{api}/repos/{repo}/releases/tags/{args.tag}')
    if response.status_code==404:
        # GitHub's by-tag endpoint can omit drafts. Reuse an existing draft on retry.
        listed=client.get(f'{api}/repos/{repo}/releases',params={'per_page':100})
        listed.raise_for_status()
        matches=[r for r in listed.json() if r['tag_name']==args.tag]
        if len(matches)>1: raise RuntimeError('Multiple matching releases; inspect before uploading')
        if matches:
            response=client.get(f'{api}/repos/{repo}/releases/{matches[0]["id"]}')
        else:
            response=client.post(f'{api}/repos/{repo}/releases',json={'tag_name':args.tag,'target_commitish':head,
                'name':'Fashion2400 reproducible local experiments','draft':True,'prerelease':True,
                'body':(ROOT/'RELEASE_NOTES.md').read_text(encoding='utf-8')})
    response.raise_for_status();release=response.json()
    expected=json.loads((ROOT/'release_assets/SHA256SUMS.json').read_text())
    assets=client.get(f'{api}/repos/{repo}/releases/{release["id"]}/assets');assets.raise_for_status()
    present={a['name']:a for a in assets.json()}
    verified=[]
    for entry in expected+[{'file':'SHA256SUMS.json'}]:
        path=ROOT/'release_assets'/entry['file']
        digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        if entry.get('sha256',digest)!=digest: raise ValueError('Archive checksum changed')
        if path.name in present:
            asset=present[path.name]
            if asset.get('digest')!='sha256:'+digest or asset['size']!=path.stat().st_size:
                raise RuntimeError('Existing asset differs; refusing overwrite')
        else:
            url=release['upload_url'].split('{')[0]
            with path.open('rb') as content:
                response=client.post(url,params={'name':path.name},content=iter(lambda:content.read(1024*1024),b''),
                    headers={'Content-Type':'application/octet-stream','Content-Length':str(path.stat().st_size)},timeout=1800)
            response.raise_for_status();asset=response.json()
            if asset['size']!=path.stat().st_size: raise RuntimeError('Uploaded size mismatch')
            if asset.get('digest') and asset['digest']!='sha256:'+digest: raise RuntimeError('Uploaded checksum mismatch')
        verified.append({'name':path.name,'size':asset['size'],'sha256':digest,'url':asset['browser_download_url']})
        print(json.dumps({'uploaded':path.name,'bytes':asset['size']}),flush=True)
    response=client.patch(f'{api}/repos/{repo}/releases/{release["id"]}',json={'draft':False})
    response.raise_for_status()
    result={'repository':f'https://github.com/{repo}','release':response.json()['html_url'],'commit':head,'assets':verified}
    (ROOT/'runs/github_publication.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
