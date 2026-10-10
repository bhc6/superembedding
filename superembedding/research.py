"""Multivalued observed-label retrieval and strictly training-only sampling."""
from collections import defaultdict
import itertools
import torch
from torch.nn import functional as F

def label_sets(records, keys):
    return [[frozenset(r['labels'][k]) for k in keys] for r in records]

def relation(labels, dims, device='cpu'):
    n=len(labels); relevant=torch.ones(n,n,dtype=torch.bool,device=device)
    observed=torch.ones(n,dtype=torch.bool,device=device)
    for k in dims:
        values=sorted(set().union(*(r[k] for r in labels)))
        mapping={v:i for i,v in enumerate(values)}
        encoded=torch.zeros(n,len(values),device=device)
        for i,r in enumerate(labels):
            if r[k]: encoded[i,[mapping[v] for v in r[k]]]=1
        relevant &= (encoded@encoded.T)>0
        observed &= encoded.any(dim=1)
    valid=observed[:,None]&observed[None,:]
    valid.fill_diagonal_(False); relevant &= valid
    return relevant,valid

@torch.no_grad()
def rank_metrics(vectors, positive, valid, chunk=256):
    device=vectors.device; n=len(vectors); rows=[]
    for start in range(0,n,chunk):
        stop=min(n,start+chunk)
        scores=vectors[start:stop].float()@vectors.float().T
        mask=valid[start:stop].to(device); positives=positive[start:stop].to(device)
        order=scores.masked_fill(~mask,-torch.inf).argsort(dim=1,descending=True,stable=True)
        hits=positives.gather(1,order)
        counts=positives.sum(1); eligible=counts>0
        precision=hits.cumsum(1)/torch.arange(1,n+1,device=device)
        ap=(precision*hits).sum(1)/counts.clamp_min(1)
        rows.append(torch.stack([ap,hits[:,0].float(),hits[:,:min(5,n)].any(1).float(),eligible.float()],1).cpu())
    values=torch.cat(rows); eligible=values[:,3].bool()
    result={'eligible_queries':int(eligible.sum()),'total_queries':n,
            'observed_queries':int(valid.any(1).sum()),
            'mAP':None,'R@1':None,'R@5':None}
    if eligible.any():
        for i,key in enumerate(('mAP','R@1','R@5')): result[key]=values[eligible,i].mean().item()
    return result

def prepare_relations(labels, keys, device='cpu'):
    dims=[(i,) for i in range(len(keys))]
    dims+=list(itertools.combinations(range(len(keys)),2))
    dims+=[tuple(range(len(keys)))]
    return {d:relation(labels,d,device) for d in dict.fromkeys(dims)}

def random_ranking_expectation(positive,valid):
    """Analytic AP/R@K expectation for a uniform random order of valid candidates."""
    n=valid.sum(1).double().cpu();m=positive.sum(1).double().cpu();eligible=m>0
    harmonic=torch.cat([torch.zeros(1,dtype=torch.float64),
        (1/torch.arange(1,int(n.max())+1,dtype=torch.float64)).cumsum(0)])
    h=harmonic[n.long()]
    ap=(h+(m-1)/(n-1).clamp_min(1)*(n-h))/n.clamp_min(1)
    miss=torch.ones_like(n)
    for rank in range(5):
        factor=(1-m/(n-rank).clamp_min(1)).clamp(0,1)
        miss*=torch.where(n>rank,factor,torch.ones_like(factor))
    return {'mAP':ap[eligible].mean().item() if eligible.any() else None,
            'R@1':(m/n.clamp_min(1))[eligible].mean().item() if eligible.any() else None,
            'R@5':(1-miss)[eligible].mean().item() if eligible.any() else None,
            'eligible_queries':int(eligible.sum()),'total_queries':len(n)}

def evaluate(slots,vectors,keys,relations):
    attrs={k:rank_metrics(slots[:,i],*relations[(i,)]) for i,k in enumerate(keys)}
    macro={}
    for metric in ('mAP','R@1','R@5'):
        vals=[v[metric] for v in attrs.values() if v[metric] is not None]
        macro[metric]=sum(vals)/len(vals) if vals else None
    attrs['macro']=macro
    pairs={}
    for dims in itertools.combinations(range(len(keys)),2):
        v=F.normalize(slots[:,dims].flatten(1),dim=-1)
        pairs['+'.join(keys[i] for i in dims)]=rank_metrics(v,*relations[dims])
    return {'attributes':attrs,'pairs':pairs,
            'joint':rank_metrics(vectors,*relations[tuple(range(len(keys)))])}

class ObservedTriplets:
    """Compact inverted indices; missing labels never become negatives."""
    def __init__(self,labels,indices,joint=False):
        self.labels=labels; self.indices=tuple(indices); self.pools=[]
        dims=[tuple(range(len(labels[0])))] if joint else [(i,) for i in range(len(labels[0]))]
        for direction in dims:
            byvalue=defaultdict(set)
            for i in self.indices:
                if all(labels[i][k] for k in direction):
                    for value in itertools.product(*(sorted(labels[i][k]) for k in direction)):
                        byvalue[value].add(i)
            positives={}; observed=sorted(set().union(*byvalue.values())) if byvalue else []
            for i in observed:
                candidates=set().union(*(byvalue[v] for v in itertools.product(*(sorted(labels[i][k]) for k in direction))))-{i}
                if candidates and len(candidates)<len(observed)-1: positives[i]=tuple(sorted(candidates))
            if not positives: raise ValueError(f'No observed triplets for dimensions {direction}')
            self.pools.append((direction,tuple(positives),positives,observed))

    def sample(self,batch,rng):
        ids=[]; attrs=[]
        for _ in range(batch):
            k=rng.randrange(len(self.pools)); dims,anchors,positive,observed=self.pools[k]
            a=rng.choice(anchors); p=rng.choice(positive[a])
            def matches(n): return all(self.labels[a][d]&self.labels[n][d] for d in dims)
            for attempt in range(100):
                n=rng.choice(observed)
                if not matches(n): break
            else: n=rng.choice([n for n in observed if not matches(n)])
            ids.append((a,p,n)); attrs.append(k)
        return torch.tensor(ids),torch.tensor(attrs)
