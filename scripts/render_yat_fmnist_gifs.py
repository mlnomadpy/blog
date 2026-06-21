#!/usr/bin/env python3
"""Render the two training GIFs for the "Your Neuron Is a Picture" JAX companion.

1. yat-vs-relu-progression.gif  — three banks (Yat from images, Yat from noise,
   ReLU filters) training on Fashion-MNIST side by side, with train-loss curves.
   All three classify; only the image-seeded Yat is legible.
2. prototype-trajectories.gif   — a UMAP fit on Fashion-MNIST with the prototypes
   tracked in that learned space: image-seeded prototypes settle into the class
   clusters, noise-seeded ones wander the gaps and never join the data.

Real JAX/Flax NNX training + UMAP fit/transform.  Run:
    python3 scripts/render_yat_fmnist_gifs.py
"""
from pathlib import Path as _Path
import tempfile as _tf
_ROOT = _Path(__file__).resolve().parents[1]
import warnings; warnings.filterwarnings('ignore')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image
from pathlib import Path
OUT=_ROOT/'public'; CACHE=_Path(_tf.gettempdir())/'yat_fmnist_cache.npz'
BG,INK,MUTED,BORDER,ACCENT,BLUE='#fbfaf6','#181818','#666a70','#ded9cb','#b3661b','#4a7fb3'
CLASS_COL=['#b3661b','#4a7fb3','#3a8f5e','#9a4f9c','#c2553a','#5a7d3a','#2f8f8f','#a06a2a','#7a5fc0','#c0892a']
CLASSES=["T-shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Boot"]
K=48; LR=1.2e-2; EPOCHS=40; B=256; SUB=20000
if not CACHE.exists():
    import jax, jax.numpy as jnp, optax, torchvision, umap
    from flax import nnx
    tr=torchvision.datasets.FashionMNIST('/tmp/fmnist',train=True,download=True); te=torchvision.datasets.FashionMNIST('/tmp/fmnist',train=False,download=True)
    X=tr.data.numpy().reshape(-1,784).astype('float32')/255.0; y=tr.targets.numpy().astype('int32')
    Xte=(te.data.numpy().reshape(-1,784).astype('float32')/255.0)[:5000]; yte=te.targets.numpy().astype('int32')[:5000]
    idx=np.random.RandomState(0).permutation(len(X))[:SUB]; X,y=X[idx],y[idx]; Xj,yj,Xtej,ytej=map(jnp.asarray,(X,y,Xte,yte))
    mu=X.mean(0); sd=X.std(0)
    Wimg=X[np.random.RandomState(1).permutation(len(X))[:K]].copy(); Wn=np.clip(mu+sd*np.random.RandomState(1).randn(K,784),0,1).astype('float32')
    class YL(nnx.Module):
        def __init__(s,*,rngs,Wi): s.W=nnx.Param(jnp.asarray(Wi)); s.lb=nnx.Param(jnp.full((),jnp.log(jnp.expm1(1.0)))); s.le=nnx.Param(jnp.full((),jnp.log(jnp.expm1(1.0))))
        def __call__(s,x):
            b=jax.nn.softplus(s.lb.value); e=jax.nn.softplus(s.le.value); dot=x@s.W.value.T
            return (dot+b)**2/(jnp.sum(x**2,-1,keepdims=True)+jnp.sum(s.W.value**2,-1)-2*dot+e)
    class YM(nnx.Module):
        def __init__(s,*,rngs,Wi): s.yat=YL(rngs=rngs,Wi=Wi); s.ro=nnx.Linear(K,10,rngs=rngs)
        def __call__(s,x): return s.ro(s.yat(x))
    class RM(nnx.Module):
        def __init__(s,*,rngs): s.l1=nnx.Linear(784,K,rngs=rngs); s.l2=nnx.Linear(K,10,rngs=rngs)
        def __call__(s,x): return s.l2(jax.nn.relu(s.l1(x)))
    ymi=YM(rngs=nnx.Rngs(0),Wi=Wimg); ymn=YM(rngs=nnx.Rngs(0),Wi=Wn); rm=RM(rngs=nnx.Rngs(0))
    oi=nnx.Optimizer(ymi,optax.adam(LR),wrt=nnx.Param); on=nnx.Optimizer(ymn,optax.adam(LR),wrt=nnx.Param); orr=nnx.Optimizer(rm,optax.adam(LR),wrt=nnx.Param)
    @nnx.jit
    def st(m,o,xb,yb):
        def lf(m): return optax.softmax_cross_entropy_with_integer_labels(m(xb),yb).mean()
        l,g=nnx.value_and_grad(lf)(m); o.update(m,g); return l
    n=len(X); total=EPOCHS*(n//B); snap=sorted(set([0]+[int(round(v)) for v in np.geomspace(1,total,32)]))
    ordr=np.random.RandomState(99).permutation(n); A,Bs,C=[],[],[]; lI,lN,lR=[],[],[]
    for t in range(total+1):
        if t in snap: A.append(np.asarray(ymi.yat.W.value).copy()); Bs.append(np.asarray(ymn.yat.W.value).copy()); C.append(np.asarray(rm.l1.kernel.value).T.copy())
        if t<total:
            i=(t*B)%(n-B); j=ordr[i:i+B]; xb,yb=Xj[jnp.asarray(j)],yj[jnp.asarray(j)]
            lI.append(float(st(ymi,oi,xb,yb))); lN.append(float(st(ymn,on,xb,yb))); lR.append(float(st(rm,orr,xb,yb)))
    accs=np.array([float((jnp.argmax(ymi(Xtej),-1)==ytej).mean()),float((jnp.argmax(ymn(Xtej),-1)==ytej).mean()),float((jnp.argmax(rm(Xtej),-1)==ytej).mean())])
    vote=np.asarray(ymi.ro.kernel.value).argmax(1); vote_n=np.asarray(ymn.ro.kernel.value).argmax(1)
    samp=np.random.RandomState(5).permutation(len(X))[:3000]; reducer=umap.UMAP(n_neighbors=15,min_dist=0.25,random_state=42).fit(X[samp])
    D=reducer.embedding_; ys=y[samp]; P=np.stack([reducer.transform(W) for W in A]); Pn=np.stack([reducer.transform(W) for W in Bs])
    np.savez(CACHE,A=np.array(A),Bs=np.array(Bs),C=np.array(C),snap=np.array(snap),total=total,accs=accs,vote=vote,lI=np.array(lI),lN=np.array(lN),lR=np.array(lR),D=D,ys=ys,P=P,Pn=Pn,vote_n=vote_n)
    print(f"cached  yat-img {accs[0]*100:.1f}  yat-noise {accs[1]*100:.1f}  relu {accs[2]*100:.1f}  final-loss yat {np.mean(lI[-20:]):.3f}")
z=np.load(CACHE); A,Bs,C,snap=z['A'],z['Bs'],z['C'],z['snap']; total=int(z['total']); accs=z['accs']; vote=z['vote']
lI,lN,lR=z['lI'],z['lN'],z['lR']; D,ys,P=z['D'],z['ys'],z['P']; Pn=z['Pn']; vote_n=z['vote_n']; S=len(snap); ai,an,ar=accs
def sm(a,w=15): 
    k=np.ones(w)/w; return np.convolve(a,k,mode='valid')
sI,sN,sR=sm(lI),sm(lN),sm(lR)
def mont(Wm,cols=6):
    rows=(K+cols-1)//cols; cv=np.zeros((rows*28,cols*28))
    for k in range(K):
        t=Wm[k].reshape(28,28); t=(t-t.min())/(np.ptp(t)+1e-9); r,c=divmod(k,cols); cv[r*28:r*28+28,c*28:c*28+28]=t
    return cv
def frame(s):
    fig=plt.figure(figsize=(9.6,5.6),dpi=96,facecolor=BG)
    fig.text(0.5,0.965,"A kernel is a picture only when its prototype lives on the data",ha='center',color=INK,fontsize=14.5,weight='bold')
    fig.text(0.5,0.928,f"step {snap[s]} / {total}  —  all three classify; only the left is readable",ha='center',color=MUTED,fontsize=9.5,family='monospace')
    for x0,title,Wm,acc,col in [(0.03,"Yat, from images",A[s],f"{ai*100:.0f}%",ACCENT),(0.355,"Yat, from noise",Bs[s],f"{an*100:.0f}%",BLUE),(0.68,"ReLU (directions)",C[s],f"{ar*100:.0f}%",MUTED)]:
        ax=fig.add_axes([x0,0.34,0.30,0.54]); ax.imshow(mont(Wm),cmap='gray',vmin=0,vmax=1); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title}  {acc}",color=INK,fontsize=10,weight='bold',pad=4)
        for sp in ax.spines.values(): sp.set_color(BORDER)
    ax=fig.add_axes([0.08,0.08,0.87,0.20]); st=snap[s]
    for ser,col,lab in [(sI,ACCENT,'Yat (images)'),(sN,BLUE,'Yat (noise)'),(sR,MUTED,'ReLU')]:
        m=min(st,len(ser)); ax.plot(np.arange(m),ser[:m],color=col,lw=1.6,label=lab)
        if m>0: ax.plot([m-1],[ser[m-1]],'o',color=col,ms=4)
    ax.set_xlim(0,total); ax.set_ylim(0.02,2.6); ax.set_yscale('log')
    ax.set_xlabel('training step',color=MUTED,fontsize=9); ax.set_ylabel('train loss',color=MUTED,fontsize=9)
    ax.tick_params(colors=MUTED,labelsize=8)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.legend(loc='upper right',fontsize=8,frameon=False,labelcolor=INK,ncol=3)
    fig.canvas.draw(); a=np.asarray(fig.canvas.buffer_rgba()).copy(); plt.close(fig); return a
fr=[frame(s) for s in range(S)]
Image.fromarray(fr[-1]).save(OUT/"yat-vs-relu-progression-preview.png")
imageio.mimsave(OUT/"yat-vs-relu-progression.gif",[fr[0]]*3+fr+[fr[-1]]*10,duration=0.36,loop=0,palettesize=48,subrectangles=True)
print("progression gif done")
# trajectory (two panels: image-init vs noise-init)
xmn,xmx,ymn,ymx=D[:,0].min(),D[:,0].max(),D[:,1].min(),D[:,1].max(); mx,my=(xmx-xmn)*0.05,(ymx-ymn)*0.05
def panel(ax,Pp,vv,title):
    for c in range(10):
        m=ys==c; ax.scatter(D[m,0],D[m,1],s=7,color=CLASS_COL[c],alpha=0.30,linewidths=0,zorder=1)
    for k in range(Pp.shape[1]):
        tr=Pp[:S2+1,k]; ax.plot(tr[:,0],tr[:,1],color=CLASS_COL[vv[k]%10],alpha=0.5,lw=1.0,zorder=3)
        ax.scatter([Pp[S2,k,0]],[Pp[S2,k,1]],s=42,color=CLASS_COL[vv[k]%10],edgecolor='white',linewidths=1.0,zorder=6)
    ax.set_xlim(xmn-mx,xmx+mx); ax.set_ylim(ymn-my,ymx+my); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.set_title(title,color=INK,fontsize=12,weight='bold',pad=5)
def f2(S2v):
    global S2; S2=S2v
    fig=plt.figure(figsize=(11.2,6.4),dpi=92,facecolor=BG)
    fig.text(0.5,0.95,"Prototypes only join the data if you seed them there",ha='center',color=INK,fontsize=15.5,weight='bold')
    fig.text(0.5,0.908,"UMAP fit on Fashion-MNIST; 48 prototypes tracked in that learned space as they train",ha='center',color=MUTED,fontsize=9.5)
    panel(fig.add_axes([0.02,0.07,0.46,0.80]),P,vote,"seeded from images: settle into the classes")
    panel(fig.add_axes([0.52,0.07,0.46,0.80]),Pn,vote_n,"seeded from noise: wander the gaps")
    fig.text(0.99,0.02,f"step {snap[S2]}",ha='right',color=MUTED,fontsize=10,family='monospace')
    fig.canvas.draw(); a=np.asarray(fig.canvas.buffer_rgba()).copy(); plt.close(fig); return a
fr2=[f2(s) for s in range(S)]
Image.fromarray(fr2[-1]).save(OUT/"prototype-trajectories-preview.png")
imageio.mimsave(OUT/"prototype-trajectories.gif",[fr2[0]]*3+fr2+[fr2[-1]]*18,duration=0.34,loop=0,palettesize=128,subrectangles=True)
print("two-panel trajectory gif done")
