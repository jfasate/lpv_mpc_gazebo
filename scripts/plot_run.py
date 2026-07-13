#!/usr/bin/env python3
import os, sys, csv, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); LOG=os.path.join(HERE,'..','log')
logf=sys.argv[1] if len(sys.argv)>1 else 'run.csv'
path=os.path.join(LOG,logf)
rows=list(csv.DictReader(open(path)))
def col(k): 
    out=[]
    for r in rows:
        try: out.append(float(r[k]))
        except: out.append(np.nan)
    return np.array(out)
X,Y=col('X'),col('Y'); rX,rY=col('ref_X'),col('ref_Y'); it=col('iter'); status=[r['status'] for r in rows]
run=np.array([s!='startup' for s in status])
# reference line (optimize)
def load(p):
    r=[[float(v) for v in l.replace(' ','').strip().split(';')] for l in open(p) if not l.startswith('#') and l.strip()]
    return np.array(r)
opt=load('src/csv_data/test_worldv5_optimize.csv')
cen=load('src/csv_data/test_worldv5.csv')
# walls from centerline
cx,cy,wr,wl=cen[:,1],cen[:,2],cen[:,7],cen[:,8]
N=len(cx); dx=np.array([cx[(i+1)%N]-cx[(i-1)%N] for i in range(N)]);dy=np.array([cy[(i+1)%N]-cy[(i-1)%N] for i in range(N)])
t=np.stack([dx,dy],1);t/=np.linalg.norm(t,axis=1,keepdims=True);nx,ny=-t[:,1],t[:,0]
rwx,rwy=cx-wr*nx,cy-wr*ny; lwx,lwy=cx+wl*nx,cy+wl*ny

fig,ax=plt.subplots(figsize=(15,9))
ax.plot(np.append(rwx,rwx[0]),np.append(rwy,rwy[0]),'k-',lw=1); ax.plot(np.append(lwx,lwx[0]),np.append(lwy,lwy[0]),'k-',lw=1,label='walls')
ax.plot(opt[:,1],opt[:,2],'-',color='green',lw=1.2,label='reference (optimize)')
# car path colored by iteration
m=run & ~np.isnan(X)
sc=ax.scatter(X[m],Y[m],c=it[m],cmap='autumn',s=8,zorder=5,label='car path (time)')
ax.plot(X[m][0],Y[m][0],'b*',ms=20,zorder=6,label='car START')
# draw a few reference-chase lines (car -> ref) every 40 ticks
idx=np.where(m)[0][::40]
for i in idx:
    ax.plot([X[i],rX[i]],[Y[i],rY[i]],'-',color='purple',lw=0.5,alpha=0.6)
plt.colorbar(sc,label='iteration'); ax.set_aspect('equal'); ax.legend(fontsize=10); ax.grid(alpha=0.3)
ax.set_title(f'{logf}: car path (dots) vs reference line (green); purple = car->ref it was chasing')
out=os.path.join(LOG,'run_path.png'); plt.savefig(out,dpi=90,bbox_inches='tight'); print('saved',os.path.normpath(out))
print("car X range", round(np.nanmin(X[m]),2), round(np.nanmax(X[m]),2), " Y range", round(np.nanmin(Y[m]),2), round(np.nanmax(Y[m]),2))
print("opt X range", round(opt[:,1].min(),2), round(opt[:,1].max(),2), " Y range", round(opt[:,2].min(),2), round(opt[:,2].max(),2))
