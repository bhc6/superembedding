"""Download official research archives with HTTP resume and verified extraction."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import tarfile
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'cub': ('https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz', '97eceeb196236b17998738112f37df78'),
    'dtd': ('https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz', 'fff73e5086ae6bdbea199a49dfb8a4c1'),
}

def download(name):
    url, md5 = SOURCES[name]
    root = ROOT/'data'/'official'; root.mkdir(parents=True, exist_ok=True)
    target = root/url.rsplit('/', 1)[-1]
    part = target.with_suffix(target.suffix+'.part')
    if not target.exists():
        for attempt in range(8):
            try:
                offset = part.stat().st_size if part.exists() else 0
                with httpx.stream('GET', url, headers={'Range': f'bytes={offset}-'} if offset else {},
                                  follow_redirects=True, timeout=120) as r:
                    r.raise_for_status()
                    append = offset and r.status_code == 206
                    if append and not r.headers.get('content-range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Incorrect Content-Range')
                    size = offset if append else 0; tick=time.monotonic()
                    with part.open('ab' if append else 'wb') as f:
                        for chunk in r.iter_bytes(1024*1024):
                            f.write(chunk); size += len(chunk)
                            if time.monotonic()-tick > 20:
                                print(f'{name}: downloaded {size/2**20:.1f} MiB', flush=True); tick=time.monotonic()
                part.replace(target)
                break
            except (httpx.HTTPError, OSError) as e:
                print(f'{name}: attempt {attempt+1}: {e}', flush=True)
                if attempt == 7: raise
                time.sleep(3)
    actual_md5 = hashlib.file_digest(target.open('rb'), 'md5').hexdigest()
    if md5 and actual_md5 != md5: raise ValueError(f'{name}: official MD5 mismatch')
    sha = hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()
    marker = root/f'{name}_verified.json'
    if not marker.exists():
        with tarfile.open(target) as archive:
            archive.extractall(root, filter='data')
        marker.write_text(json.dumps({'url':url,'sha256':sha,'md5':actual_md5,'size':target.stat().st_size},indent=2))
    print(f'{name}: verified and extracted, SHA256 {sha}', flush=True)

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, SOURCES))
