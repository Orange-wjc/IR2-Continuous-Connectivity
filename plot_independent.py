"""Figures generated only from completed and independently validated results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reproduction_results/independent_20261006'


def main():
    report=json.loads((OUT/'comparison.json').read_text());assert report['complete']
    assert json.loads((OUT/'validation.json').read_text())['all_pass']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    conditions=['low','medium','high'];methods=['ir2','frontier_rendezvous'];colors=['#0072B2','#D55E00']
    fig,ax=plt.subplots(figsize=(6.5,3.8),layout='constrained')
    for j,method in enumerate(methods):
        values=[report['per_condition'][c][method]['successes']/30*100 for c in conditions]
        bars=ax.bar(np.arange(3)+(j-.5)*.32,values,width=.32,color=colors[j],label=['Corrected IR2','Frontier / rendezvous'][j])
        for bar,c in zip(bars,conditions):
            r=report['per_condition'][c][method];ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"{r['successes']}/30",ha='center',fontsize=9)
    ax.set_xticks(range(3),['Low (0)','Medium (6.5)','High (13)']);ax.set_ylabel('Exploration success (%)');ax.set_xlabel('Fixed attenuation condition: X_g = K');ax.set_ylim(0,112);ax.legend(loc='upper center',ncols=2,frameon=False);ax.axhline(80,color='#666666',ls=':',lw=.8)
    fig.savefig(OUT/'success_comparison.png',dpi=250);fig.savefig(OUT/'success_comparison.pdf');plt.close(fig)
    protocol=json.loads((OUT/'protocol.json').read_text());layout=protocol['layouts'][0]
    with Image.open(ROOT/layout['map_path']) as im:free=np.asarray(im.convert('L'))>150
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    robot_colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
    for row,method in enumerate(methods):
        for col,condition in enumerate(conditions):
            data=json.loads((OUT/'raw'/f'00_{condition}_{method}.json').read_text());r=data['row'];ax=axes[row,col]
            ax.imshow(free,cmap='gray',vmin=0,vmax=1)
            for i,path in enumerate(data['paths']):
                points=np.asarray(path);ax.plot(points[:,0],points[:,1],color=robot_colors[i],lw=.65,alpha=.85);ax.scatter(*points[0],color=robot_colors[i],s=9,marker='o');ax.scatter(*points[-1],color=robot_colors[i],s=20,marker='x')
            ax.set_title(f"{method} / {condition}\n{'SUCCESS' if r['success'] else 'FAIL'}; {r['steps']} rounds; min coverage {r['min_explored']*100:.1f}%",fontsize=9);ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Predeclared first layout: '+Path(layout['map_path']).name+'; ground truth used only for visualization',fontsize=11)
    fig.savefig(OUT/'first_layout_trajectories.png',dpi=200);fig.savefig(OUT/'first_layout_trajectories.pdf');plt.close(fig)
    print('Saved success comparison and first-layout trajectories (PNG/PDF)')


if __name__=='__main__':main()
