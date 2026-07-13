#!/usr/bin/env python3
import os, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); LOG=os.path.join(HERE,'..','log')

def load(path):
    r=[[float(v) for v in l.replace(' ','').strip().split(';')] for l in open(path) if not l.startswith('#') and l.strip()]
    return np.array(r)

def lnorm(x,y):
    N=len(x); dx=np.array([x[(i+1)%N]-x[(i-1)%N] for i in range(N)]); dy=np.array([y[(i+1)%N]-y[(i-1)%N] for i in range(N)])
    t=np.stack([dx,dy],1); t/=np.linalg.norm(t,axis=1,keepdims=True); return np.stack([-t[:,1],t[:,0]],1)

kmax=np.tan(0.4189)/0.33
c=load('src/csv_data/test_worldv5.csv'); cx,cy,cwr,cwl,ck=c[:,1],c[:,2],c[:,7],c[:,8],c[:,4]
n=lnorm(cx,cy); rwx,rwy=cx-cwr*n[:,0],cy-cwr*n[:,1]; lwx,lwy=cx+cwl*n[:,0],cy+cwl*n[:,1]

def stats(name,a):
    k=np.abs(a[:,4]); v=a[:,5]
    print(f"{name:20s} pts={len(a):3d}  max|kappa|={k.max():.3f}  undrivable={int((k>kmax).sum()):3d}  vx[{v.min():.2f},{v.max():.2f}] mean={v.mean():.2f}")

print(f"steering-limit curvature = {kmax:.3f}")
stats('centerline', c)
opt=load('src/csv_data/test_worldv5_optimize.csv'); stats('optimize (yours)', opt)
mc=load('src/csv_data/test_worldv5_mincurv.csv'); stats('mincurv (mine)', mc)

fig,ax=plt.subplots(figsize=(15,9))
ax.plot(np.append(rwx,rwx[0]),np.append(rwy,rwy[0]),'k-',lw=1.5)
ax.plot(np.append(lwx,lwx[0]),np.append(lwy,lwy[0]),'k-',lw=1.5,label='track walls')
ax.plot(cx,cy,'--',color='gray',lw=1.0,label='centerline')
ax.plot(opt[:,1],opt[:,2],'-',color='tab:green',lw=1.8,label='optimize (yours)')
ax.plot(mc[:,1],mc[:,2],'-',color='tab:orange',lw=1.4,label='mincurv (mine)')
ax.set_aspect('equal'); ax.legend(fontsize=11); ax.grid(alpha=0.3)
ax.set_title('test_worldv5: your optimize line vs min-curvature line')
os.makedirs(LOG,exist_ok=True); out=os.path.join(LOG,'line_all.png')
plt.savefig(out,dpi=90,bbox_inches='tight'); print('saved',os.path.normpath(out))
