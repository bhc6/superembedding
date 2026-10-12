"""Download complete official MMEB ScienceQA task without the 7 GB image ZIP."""
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
import httpx
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from superembedding.checkpoint import atomic_json

REV='2f069730be515ea60778413777816b53e2d2a697'
BASE=f'https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/{REV}/'

class RangeReader(io.RawIOBase):
    def __init__(self,client,url):
        self.client=client
        r=client.head(url);r.raise_for_status();self.url=str(r.url)
        self.size=int(r.headers['content-length']);self.pos=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n==0:return b''
        if n>64*2**20:raise RuntimeError('Unexpected ZIP range exceeds 64 MiB')
        end=self.pos+n-1
        for attempt in range(6):
            try:
                # stream prevents accidental full-archive buffering if server ignores Range.
                with self.client.stream('GET',self.url,headers={'Range':f'bytes={self.pos}-{end}'}) as r:
                    if r.status_code!=206 or r.headers.get('content-range')!=f'bytes {self.pos}-{end}/{self.size}':
                        raise RuntimeError('Server failed exact byte-range check')
                    data=r.read()
                if len(data)!=n:raise RuntimeError('Truncated range')
                self.pos+=n;return data
            except httpx.HTTPError:
                if attempt==5:raise RuntimeError('ZIP network failure, URL redacted') from None
                time.sleep(2**attempt)


def main():
    root=Path('C:/Users/123/.cache/superembedding/v5/mmeb_scienceqa');root.mkdir(parents=True,exist_ok=True)
    with httpx.Client(timeout=90,follow_redirects=True) as c:
        p=root/'test.parquet'
        if not p.exists():
            r=c.get(BASE+'ScienceQA/test-00000-of-00001.parquet');r.raise_for_status();p.write_bytes(r.content)
        rows=pq.read_table(p).to_pylist()
        remote=RangeReader(c,BASE+'images.zip')
        with zipfile.ZipFile(remote) as archive:
            names=archive.namelist();lookup={n.split('ScienceQA/',1)[-1]:n for n in names if 'ScienceQA/' in n and not n.endswith('/')}
            def download(row):
                destination=root/row['qry_img_path'];destination.parent.mkdir(exist_ok=True,parents=True)
                if not destination.exists():
                    key=row['qry_img_path'].split('ScienceQA/',1)[-1]
                    info=archive.getinfo(lookup[key])
                    reader=RangeReader.__new__(RangeReader)
                    reader.client=c;reader.url=remote.url;reader.size=remote.size;reader.pos=info.header_offset
                    block=reader.read(info.compress_size+4096)
                    if block[:4]!=b'PK\x03\x04':raise RuntimeError('Invalid local ZIP header')
                    name_length,extra_length=struct.unpack_from('<HH',block,26)
                    offset=30+name_length+extra_length
                    if offset>4096:raise RuntimeError('Unexpectedly large ZIP header')
                    compressed=block[offset:offset+info.compress_size]
                    data=zlib.decompress(compressed,-15) if info.compress_type==8 else compressed if info.compress_type==0 else None
                    if data is None or len(data)!=info.file_size or zlib.crc32(data)!=info.CRC:
                        raise RuntimeError('ZIP integrity check failed')
                    temp=destination.with_suffix('.tmp');temp.write_bytes(data);temp.replace(destination)
            with ThreadPoolExecutor(max_workers=8) as pool:
                for i,_ in enumerate(pool.map(download,rows),1):
                    if i%50==0:print(json.dumps({'scienceqa_images':i,'total':len(rows)}),flush=True)
    atomic_json({'dataset':'TIGER-Lab/MMEB-eval','revision':REV,'config':'ScienceQA','split':'test',
                 'queries':len(rows),'parquet_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                 'scope':'Complete official task; original per-query candidate lists and positive-at-index-0 preserved.'},root/'metadata.json')

if __name__=='__main__':main()
