"""Create method comparisons and figures; called by part_b_heat_flux.py."""
from pathlib import Path
import argparse,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results')
R=parser.parse_args().results;F=R/'figures';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':10})
def rows(name):return list(csv.DictReader((R/name).open()))
nodes=rows('nodal_flux_per_k.csv');all_methods=rows('seven_method_nodal_comparison.csv')
base=np.array([[float(r['qx_per_k']),float(r['qy_per_k'])] for r in nodes])
details=[]
for name in dict.fromkeys(r['method'] for r in all_methods):
    a=np.array([[float(r['qx_per_k']),float(r['qy_per_k'])] for r in all_methods if r['method']==name])
    delta=np.linalg.norm(a-base,axis=1)
    relative=100*delta/np.linalg.norm(base,axis=1)
    angles=np.degrees(np.arctan2(a[:,1],a[:,0])-np.arctan2(base[:,1],base[:,0]))
    angles=(angles+180)%360-180
    signs=bool(np.all(np.sign(a)==np.sign(base)))
    details.append([name,float(np.max(delta)),float(np.max(relative)),float(np.max(abs(angles))),signs])
with (R/'method_summary.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f);w.writerow(['method','max_vector_difference','max_relative_percent','max_direction_degrees','same_component_signs']);w.writerows(details)
faces=rows('heater_faces_per_k.csv')
open_faces=[r for r in faces if float(r['endpoint_sample_not_finite_limit'])==0]
mid=next(r for r in faces if float(r['y'])==2.5)
zones=[]
for lower,upper,label in [(2.25,2.75,'Central half'),(2.125,2.875,'Central three quarters'),(2.015625,2.984375,'All non-endpoint samples')]:
    subset=[r for r in open_faces if lower<=float(r['y'])<=upper]
    errors=[max(100*abs(float(r[side+'_first'])-float(r[side+'_second']))/abs(float(r[side+'_second'])) for r in subset) for side in ['left','right']]
    zones.append([label,lower,upper,*errors])
with (R/'heater_region_comparison.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f);w.writerow(['region','y_min','y_max','left_max_relative_percent','right_max_relative_percent']);w.writerows(zones)
with (R/'method_differences_by_node.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f);w.writerow(['method','x','y','relative_vector_percent','direction_degrees'])
    for name in dict.fromkeys(r['method'] for r in all_methods):
        aa=np.array([[float(r['qx_per_k']),float(r['qy_per_k'])] for r in all_methods if r['method']==name])
        for i,node in enumerate(nodes):
            angle=(np.degrees(np.arctan2(aa[i,1],aa[i,0])-np.arctan2(base[i,1],base[i,0]))+180)%360-180
            w.writerow([name,node['x'],node['y'],100*np.linalg.norm(aa[i]-base[i])/np.linalg.norm(base[i]),angle])
sample=[next(r for r in faces if float(r['y'])==y) for y in [2.125,2.25,2.5,2.75,2.875]]
with (R/'heater_face_samples.csv').open('w',newline='',encoding='utf-8') as f:
    keys=['y','left_first','left_second','right_first','right_second'];w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader();w.writerows(sample)
fig,axes=plt.subplots(1,2,figsize=(8.3,3.15))
for ax,side in zip(axes,['left','right']):
    y=[float(r['y']) for r in open_faces]
    for method,label,style in [('first','Two-point (first order)','--'),('second','Three-point (second order)','-')]:
        ax.plot(y,[float(r[side+'_'+method]) for r in open_faces],style,label=label,lw=1.5)
    ax.set(xlabel='Position along heater y',ylabel=r'Outward heat flux $q/k$',title=side.capitalize()+' heater face')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(F/'heater_method_comparison.png',dpi=300);plt.close(fig)
# All methods use the same field; C2 has zero difference and is omitted.
nonzero=[r for r in details if r[0]!='Central 2']
fig,axes=plt.subplots(1,2,figsize=(8.3,3.0))
for ax,index,label in [(axes[0],2,'Maximum relative vector difference (%)'),(axes[1],3,'Maximum direction difference (degrees)')]:
    ax.bar(np.arange(6),[r[index] for r in nonzero],color=['#597da0']*4+['#448a80']*2)
    ax.set_xticks(np.arange(6),[r[0] for r in nonzero],rotation=25,fontsize=8)
    ax.set_yscale('log');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    ax.set_title('Heat-flux values' if index==2 else 'Heat-flow directions')
fig.tight_layout();fig.savefig(F/'method_selection.png',dpi=300);plt.close(fig)
assert all(r[4] for r in details)
assert np.isclose(details[-1][2],details[-2][2])
print('Generated method comparison data and figures in',R.resolve())
