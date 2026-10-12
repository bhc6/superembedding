"""Original CC0 synthetic paired evidence diagnostic. No external images/fonts."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageDraw,ImageFont
from superembedding.checkpoint import atomic_json

COLORS={'red':'#cf3434','blue':'#275ed0','green':'#268144','orange':'#d97817'}
SHAPES=['circle','square','triangle']

def draw_scene(path,scene,variant):
    rng=random.Random(scene['seed']+(70000 if variant=='irrelevant' else 0))
    im=Image.new('RGB',(448,320),'#f5f3ee' if variant!='irrelevant' else '#eaf3fa')
    d=ImageDraw.Draw(im);font=ImageFont.load_default(size=18)
    x,y=scene['position'];color=COLORS[scene['color']];shape=scene['shape']
    def shape_at(kind,x,y,c):
        if kind=='square':d.rectangle((x-25,y-25,x+25,y+25),fill=c)
        elif kind=='circle':d.ellipse((x-25,y-25,x+25,y+25),fill=c)
        else:d.polygon([(x,y-28),(x-28,y+25),(x+28,y+25)],fill=c)
    d.text((x-35,y-65),'TARGET A',fill='black',font=font)
    shape_at(shape,x,y,color)
    d.text((x-32,y+39),'ID '+str(scene['number']),fill='black',font=font)
    # B is an irrelevant object, kept spatially separated from A.
    bx=350 if x<224 else 90;by=160
    d.text((bx-20,by-62),'B',fill='black',font=font)
    shape_at(rng.choice(SHAPES),bx,by,rng.choice(list(COLORS.values())))
    for _ in range(8):
        px=rng.randrange(15,430);py=rng.choice([18,292])
        d.line((px,py,px+8,py+3),fill=rng.choice(list(COLORS.values())),width=2)
    im.save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='data/evidence_diagnostic_v1')
    ap.add_argument('--scenes',type=int,default=64);args=ap.parse_args()
    root=Path(args.output);(root/'images').mkdir(parents=True,exist_ok=True)
    cases=[];rng=random.Random(20261011)
    for i in range(args.scenes):
        scene={'seed':i,'color':rng.choice(list(COLORS)),'shape':rng.choice(SHAPES),
               'number':rng.randrange(10,99),'position':rng.choice([(105,145),(330,145)])}
        base=f'images/{i:04d}_anchor.png';irrelevant=f'images/{i:04d}_irrelevant.png'
        draw_scene(root/base,scene,'anchor');draw_scene(root/irrelevant,scene,'irrelevant')
        for attribute in ['color','shape','number','position']:
            changed=dict(scene)
            if attribute=='color':changed[attribute]=rng.choice([x for x in COLORS if x!=scene[attribute]])
            elif attribute=='shape':changed[attribute]=rng.choice([x for x in SHAPES if x!=scene[attribute]])
            elif attribute=='number':changed[attribute]=(scene[attribute]+37-10)%89+10
            else:changed[attribute]=(330,145) if scene['position'][0]<224 else (105,145)
            relevant=f'images/{i:04d}_{attribute}.png';draw_scene(root/relevant,changed,'anchor')
            value=('left' if scene['position'][0]<224 else 'right') if attribute=='position' else scene[attribute]
            query={'color':f'Target A is {value}.','shape':f'Target A is a {value}.',
                   'number':f'The ID printed below Target A is {value}.',
                   'position':f'Target A is on the {value} side of the image.'}[attribute]
            cases.append({'id':f'{i:04d}/{attribute}','scene_group':i,'attribute':attribute,
                          'anchor':base,'relevant_change':relevant,'irrelevant_change':irrelevant,
                          'query':query,'anchor_factors':scene,'changed_factors':changed})
    (root/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cases),encoding='utf-8')
    (root/'LICENSE').write_text('Original synthetic scene images and annotations: CC0 1.0 Universal.\nhttps://creativecommons.org/publicdomain/zero/1.0/\nNo external images; Pillow bundled default font. Generator code follows repository license.\n')
    atomic_json({'version':1,'seed':20261011,'cases':len(cases),'scene_groups':args.scenes,
                 'images':args.scenes*6,'license':'CC0-1.0','split':'diagnostic only; never train on these cases',
                 'purpose':'Sensitivity to target changes versus nuisance changes, reusable candidate encoding',
                 'limitations':['Synthetic, single renderer; does not establish real-world generalization or novelty.',
                                'Position intervention swaps target/distractor locations; analyze separately from appearance/text interventions.',
                                'Target direction labels require testing, not assumed correspondence to learned slots.'],
                 'records_sha256':hashlib.sha256((root/'records.jsonl').read_bytes()).hexdigest(),
                 'files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'images').glob('*.png'))}},root/'metadata.json')
    print(json.dumps({'cases':len(cases),'images':args.scenes*6,'output':str(root.resolve())}),flush=True)

if __name__=='__main__':main()
