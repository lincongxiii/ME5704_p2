"""Reproduce Part B data and all three figures: python part_b_heat_flux.py.

Flux coefficients q/k are in K per coordinate-length unit. The internal line
x=1.5, 2<=y<=3 is fixed at 350 K; the outer edges are fixed at 200 K.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONFIRMED = np.array([226.6487,231.5347,216.0715,271.8111,279.7359,
                      229.1359,266.4886,272.7308,224.2339])
POINTS = [(x,y) for y in (1,2,3) for x in (1,2,3)]
ROOT = Path(__file__).resolve().parent
SCHEMES = {
    'Forward 1': ([0,1],[-1,1],1),
    'Backward 1': ([-1,0],[-1,1],1),
    'Forward 2': ([0,1,2],[-1.5,2,-.5],2),
    'Backward 2': ([-2,-1,0],[.5,-2,1.5],2),
    'Central 2': ([-1,1],[-.5,.5],2),
    'Central 4': ([-2,-1,1,2],[1/12,-2/3,2/3,-1/12],4),
    'Richardson': ([-1,-.5,.5,1],[1/6,-4/3,4/3,-1/6],4)}

def solve(h):
    n=round(4/h)
    if not np.isclose(n*h,4) or not np.isclose(1.5/h,round(1.5/h)) or not np.isclose(1/h,round(1/h)):
        raise ValueError('Grid must align with domain, target points and heater endpoints')
    axis=np.arange(n+1)*h; X,Y=np.meshgrid(axis,axis)
    outer=(X==0)|(X==4)|(Y==0)|(Y==4)
    source=np.isclose(X,1.5)&(Y>=2)&(Y<=3)
    fixed=outer|source
    T=np.full(X.shape,200.)
    T[source]=350.
    unknown=np.argwhere(~fixed); ids=np.full(X.shape,-1,int)
    ids[~fixed]=np.arange(len(unknown))
    A=lil_matrix((len(unknown),len(unknown))); b=np.zeros(len(unknown))
    for row,(j,i) in enumerate(unknown):
        A[row,row]=4
        for dj,di in [(0,1),(0,-1),(1,0),(-1,0)]:
            jj,ii=j+dj,i+di
            if fixed[jj,ii]: b[row]+=T[jj,ii]
            else: A[row,ids[jj,ii]]=-1
    A=A.tocsr(); u=spsolve(A,b); T[~fixed]=u
    residual=float(np.max(np.abs(A@u-b))/(np.max(np.asarray(abs(A).sum(axis=1)))*np.max(abs(u))+np.max(abs(b))))
    return axis,T,source,residual,len(u)

def nodal(T,h,order=2):
    rows=[]
    for x,y in POINTS:
        i,j=round(x/h),round(y/h)
        radius=1 if order==2 else 2
        # A stencil may touch the heater as its boundary, but cannot cross it.
        if 2<=y<=3 and (x-radius*h<1.5<x+radius*h):
            raise ValueError('Stencil crosses internal source; refine grid')
        if order==2:
            dx=(T[j,i+1]-T[j,i-1])/(2*h); dy=(T[j+1,i]-T[j-1,i])/(2*h)
        else:
            dx=(T[j,i-2]-8*T[j,i-1]+8*T[j,i+1]-T[j,i+2])/(12*h)
            dy=(T[j-2,i]-8*T[j-1,i]+8*T[j+1,i]-T[j+2,i])/(12*h)
        rows.append([x,y,T[j,i],-dx,-dy,np.hypot(dx,dy)])
    return np.array(rows)

def power(T,source):
    """Discrete reaction: each source-to-free edge once, including endpoint edges.

    k*b*(Ts-Tu)*h/h is edge heat rate. Return Q/(k*b), b=thickness.
    Opposite fluxes on free-free edges cancel in the summed equations.
    """
    total=0.
    for j,i in np.argwhere(source):
        for dj,di in [(0,1),(0,-1),(1,0),(-1,0)]:
            if not source[j+dj,i+di]: total+=T[j,i]-T[j+dj,i+di]
    outer=float(np.sum(T[1,1:-1]-T[0,1:-1])+np.sum(T[-2,1:-1]-T[-1,1:-1])+
                np.sum(T[1:-1,1]-T[1:-1,0])+np.sum(T[1:-1,-2]-T[1:-1,-1]))
    return total,outer

def faces(T,h):
    i=round(1.5/h); rows=[]
    for j in range(round(2/h),round(3/h)+1):
        left1=(350-T[j,i-1])/h; right1=(350-T[j,i+1])/h
        left2=(3*350-4*T[j,i-1]+T[j,i-2])/(2*h)
        right2=(3*350-4*T[j,i+1]+T[j,i+2])/(2*h)
        rows.append([j*h,left1,right1,left2,right2,int(j in (round(2/h),round(3/h)))])
    return np.array(rows)

def csvwrite(path,header,rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=ROOT/'results')
    args=parser.parse_args();out=args.outdir; (out/'figures').mkdir(parents=True,exist_ok=True)
    checks={};studies=[];grids=[]
    for h in [.5,.25,.125,.0625,.03125,.015625]:
        a,T,s,r,n=solve(h);p,outer=power(T,s);fs=faces(T,h)
        v=nodal(T,h);grids.append((a,T,s,v,fs))
        assert np.min(T)>=200-1e-8 and np.max(T)<=350+1e-8
        assert np.max(abs(T[s]-350))==0 and r<1e-12
        assert abs(p-outer)/p<1e-9
        mid=fs[len(fs)//2]
        studies.append([h,n,r,p,outer,abs(p-outer)/p,mid[3],mid[4]])
    a,T,s,v,fs=grids[-1];h=a[1]
    mismatch=float(np.max(abs(v[:,2]-CONFIRMED)))
    assert mismatch<=5.1e-5,('Part A temperature mismatch',mismatch)
    fourth=nodal(T,h,4)
    scheme_diff=float(np.max(np.linalg.norm(v[:,3:5]-fourth[:,3:5],axis=1)))
    csvwrite(out/'nodal_flux_per_k.csv',['x','y','T_K','qx_per_k','qy_per_k','magnitude_per_k'],v)
    csvwrite(out/'nodal_flux_central4_per_k.csv',['x','y','T_K','qx_per_k','qy_per_k','magnitude_per_k'],fourth)
    csvwrite(out/'heater_faces_per_k.csv',['y','left_first','right_first','left_second','right_second','endpoint_sample_not_finite_limit'],fs)
    csvwrite(out/'grid_study.csv',['h','unknowns','normalized_residual','Q_source_per_kb','Q_outer_per_kb','relative_balance','left_mid_second','right_mid_second'],studies)
    comparison=[]
    for name,(offsets,weights,p) in SCHEMES.items():
        # D_R = [4 D(h) - D(2h)]/3 on the same temperature field.
        step=2*h if name=='Richardson' else h
        off=np.rint(np.array(offsets)*step/h).astype(int)
        for x,y in POINTS:
            i,j=round(x/h),round(y/h)
            dx=float(np.dot(weights,T[j,i+off])/step)
            dy=float(np.dot(weights,T[j+off,i])/step)
            comparison.append([name,x,y,-dx,-dy])
    csvwrite(out/'seven_method_nodal_comparison.csv',['method','x','y','qx_per_k','qy_per_k'],comparison)
    csvwrite(out/'target_grid_study.csv',['h','x','y','T_K','qx_per_k','qy_per_k'],[[a[1],*r[:5]] for a,T,s,v,fs0 in grids for r in v])
    np.savez_compressed(out/'temperature_full_finest.npz',x=a,y=a,T=T,source=s)
    change=float(np.max(np.linalg.norm(v[:,3:5]-grids[-2][3][:,3:5],axis=1)))
    checks.update({'part_a_max_rounding_difference_K':mismatch,'nodal_C2_C4_max_vector_difference':scheme_diff,
                   'nodal_last_grid_max_vector_change':change,
                   'mesh_change_to_formula_change_ratio':change/scheme_diff,
                   'temperature_min_K':float(T.min()),'temperature_max_K':float(T.max()),
                   'heater_max_temperature_error_K':float(np.max(abs(T[s]-350))),
                   'max_normalized_residual':max(row[2] for row in studies),
                   'max_relative_heat_balance':max(row[5] for row in studies),
                   'status':'PASS'})
    (out/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    X,Y=np.meshgrid(a,a)
    fig,ax=plt.subplots(figsize=(6.4,5.4));cf=ax.contourf(X,Y,T,levels=np.linspace(200,350,31),cmap='inferno')
    ax.contour(X,Y,T,levels=np.arange(210,350,10),colors='white',linewidths=.35,alpha=.5)
    ax.plot([1.5,1.5],[2,3],color='cyan',lw=3,label='Internal heat source (350 K)')
    ax.quiver(v[:,0],v[:,1],v[:,3],v[:,4],color='cyan',angles='xy',scale_units='xy',scale=180,width=.006)
    ax.quiverkey(ax.collections[-1],.8,1.04,50,'50 K per length unit',labelpos='W')
    ax.set(xlabel='x',ylabel='y',xlim=(0,4),ylim=(0,4));ax.set_aspect('equal');ax.legend(loc='lower left',fontsize=8)
    fig.colorbar(cf,ax=ax,label='Temperature (K)');fig.tight_layout();fig.savefig(out/'figures/temperature_and_flux.png',dpi=230);plt.close(fig)
    print(json.dumps(checks,indent=2))
    subprocess.run([sys.executable,str(ROOT/'plot_results.py'),'--results',str(out.resolve())],check=True)

if __name__=='__main__':main()
