import hashlib
import json
from pathlib import Path
import pytest
import torch
from PIL import Image
from superembedding.baselines import image_bytes,validate_vectors,load_packed
from superembedding.checkpoint import atomic_save,fingerprint
from scripts.extract_baselines import api_batch

def test_decoded_image_hash_and_corruption(tmp_path):
    image=Image.new('RGB',(7,9),'blue');image.save(tmp_path/'image.png')
    record={'id':'one','image':'image.png','image_sha256':hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()}
    assert image_bytes(tmp_path,record)==(tmp_path/'image.png').read_bytes()
    Image.new('RGB',(7,9),'red').save(tmp_path/'image.png')
    with pytest.raises(ValueError,match='hash changed'):image_bytes(tmp_path,record)

def test_packed_global_directions_and_tamper(tmp_path):
    records=[{'id':'one','image_sha256':'abc'}]
    prov={'basis':{'directions':['color','shape']},'spec':{'width':3}}
    p={'ids':['one'],'hashes':['abc'],'provenance_fingerprint':fingerprint(prov),'vectors':torch.tensor([[1.,2.,3.]])}
    atomic_save(p,tmp_path/'features.pt')
    u,z=load_packed(tmp_path,records,prov)
    assert u.shape==(1,2,3) and torch.equal(u[:,0],u[:,1]) and torch.equal(z,u[:,0])
    p['hashes']=['changed'];atomic_save(p,tmp_path/'features.pt')
    with pytest.raises(ValueError,match='record mismatch'):load_packed(tmp_path,records,prov)

@pytest.mark.parametrize('v',[torch.zeros(2,3),torch.full((2,3),float('nan')),torch.ones(1,3)])
def test_invalid_embeddings(v):
    with pytest.raises(ValueError):validate_vectors(v,2,3)

def test_api_reorders_indices_and_does_not_log_credentials(tmp_path,monkeypatch):
    import scripts.extract_baselines as ex
    image=Image.new('RGB',(7,9),'blue');image.save(tmp_path/'image.png')
    digest=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
    rows=[{'id':str(i),'image':'image.png','image_sha256':digest} for i in range(2)]
    class Response:
        status_code=200
        def json(self):return {'output':{'embeddings':[{'index':1,'embedding':[3.,4.]},{'index':0,'embedding':[1.,2.]}]},'usage':{'input_tokens':4}}
    class Client:
        def post(self,url,json):
            assert json['parameters']['instruct']=='color'
            return Response()
    v,info=api_batch(Client(),{'model':'mock','width':2,'parameters':{}},tmp_path,rows,'color')
    assert torch.equal(v,torch.tensor([[1.,2.],[3.,4.]]))
    assert 'base64' not in json.dumps(info) and info['usage']['input_tokens']==4
