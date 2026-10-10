"""Exact branch topology, batched only across images with identical prefix layout.

Each image retains its own recurrent/KV state; each option receives a copy of
that image's state. Different images and different options never attend together.
"""
import torch

def prepare_image(encoder,image):
    from mso import branch as BR
    mso=encoder._mso; mso.dev=next(encoder.backbone.parameters()).device
    questions={'attributes':{'type':'choice','instructions':encoder.basis.instruction,
                             'criteria':{a:None for a in encoder.basis.directions}}}
    keys,opts,ids,spans,pix=mso._encode_many(str(image),None,questions,['attributes'])
    if tuple(keys[0])!=encoder.basis.directions: raise ValueError('Attribute order changed')
    opens,closes=spans[0]; length=max(1,opens[0]-1)
    base=dict(pix); base['input_ids']=torch.tensor(ids[0],device=mso.dev)[None]
    base['attention_mask']=torch.ones_like(base['input_ids'])
    penc=BR.prefix_inputs(base,length)
    rows=BR.rows_from_ids(ids[0],length,opens,closes,k=len(opts[0]))
    key=(tuple(ids[0][:length]),tuple(tuple(r) for r in rows),
         tuple(pix['image_grid_thw'].flatten().tolist()))
    return {'prefix':penc,'rows':rows,'key':key}

@torch.no_grad()
def encode_prepared(encoder,items):
    from mso import branch as BR
    if not items or any(i['key']!=items[0]['key'] for i in items):
        raise ValueError('Batched images must have identical token/grid layouts')
    mso=encoder._mso; dev=mso.dev; backbone=encoder.backbone
    batch=len(items); rows=items[0]['rows']; k=len(rows)
    penc={key:torch.cat([i['prefix'][key] for i in items],dim=0).to(dev) for key in items[0]['prefix']}
    prefix_pos=BR.prefix_positions(mso.model,penc)
    length=penc['input_ids'].shape[1]
    out=backbone(**penc,position_ids=prefix_pos,use_cache=True,logits_to_keep=1)
    cache=BR.expand_cache(out.past_key_values,k)
    p0=int(prefix_pos.max())+1 if prefix_pos is not None else length
    allrows=rows*batch; n=max(map(len,rows)); count=len(allrows)
    ids=torch.full((count,n),mso.pad_id,device=dev,dtype=torch.long)
    attention=torch.zeros((count,n),device=dev,dtype=torch.long)
    for i,row in enumerate(allrows):
        ids[i,:len(row)]=torch.tensor(row,device=dev); attention[i,:len(row)]=1
    full_attention=torch.cat([torch.ones((count,length),device=dev,dtype=torch.long),attention],dim=1)
    pos=torch.arange(p0,p0+n,device=dev)[None,None,:].expand(3,count,n).contiguous()
    result=backbone(input_ids=ids,attention_mask=full_attention,past_key_values=cache,use_cache=True,
                    output_hidden_states=True,cache_position=torch.arange(length,length+n,device=dev),
                    position_ids=pos,logits_to_keep=1)
    h=result.hidden_states[-1]; marks=[BR._marks(r,mso.open_ids,mso.close_ids) for r in allrows]
    ar=torch.arange(count,device=dev)
    u=h[ar,torch.tensor([m[1] for m in marks],device=dev)].float().reshape(batch,k,-1)
    zq=h[ar,torch.tensor([max(0,m[0]-1) for m in marks],device=dev)].float().reshape(batch,k,-1)[:,0]
    return u,zq
