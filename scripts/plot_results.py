"""Scientific figure from recorded metrics; error bars are seed SD, not CI."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    entries=[]
    for family in ['omnijev','base','flair']:
        r=json.loads((ROOT/f'runs/fashion2400/results_{family}.json').read_text())
        for name,values in r['architectures'].items():
            entries.append((f'{family}\n{name}',values['summary']))
    x=np.arange(len(entries));fig,ax=plt.subplots(figsize=(13,5.5),layout='constrained')
    for offset,key,label,color in [(-.2,'attribute_mAP','Single attribute','#227C9D'),(.2,'joint_mAP','All three attributes','#E07A5F')]:
        values=[e[1][key]['mean']*100 for e in entries]
        errors=[e[1][key]['std']*100 for e in entries]
        ax.bar(x+offset,values,.38,yerr=errors,label=label,color=color,capsize=3)
    ax.set_xticks(x,[e[0] for e in entries],rotation=25,ha='right',fontsize=9)
    ax.set_ylabel('Held-out image-to-image retrieval mAP (%)')
    ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.set_title('Fashion2400: 1,462 train / 469 validation / 469 test; mean ± seed SD (n=3)')
    ax.legend(frameon=False)
    out=ROOT/'artifacts/figures';out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'architecture_comparison.png',dpi=180)
    fig.savefig(out/'architecture_comparison.pdf')
    plt.close(fig)


if __name__=='__main__':main()
