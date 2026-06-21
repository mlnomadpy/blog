#!/usr/bin/env python3
"""Render notrain-confusion.gif for the "Your Neuron Is a Picture" JAX companion.

As labeled k-means-centroid prototypes are added per class (with a one-hot
readout and ZERO training), the test-set confusion matrix sharpens from muddy to
a clean diagonal and accuracy climbs 42 -> 68 percent. Reads the exported
centroid sprite (public/yat-fmnist/centroids.png). Real JAX kernel computation.
    python3 scripts/render_yat_notrain_gif.py
"""
from pathlib import Path as _Path
_ROOT=_Path(__file__).resolve().parents[1]
import warnings; warnings.filterwarnings('ignore')
import numpy as np, jax, jax.numpy as jnp, torchvision
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image
from pathlib import Path
OUT=_ROOT/'public'; BG,INK,MUTED,BORDER,ACCENT='#fbfaf6','#181818','#666a70','#ded9cb','#b3661b'
CLASS_COL=['#b3661b','#4a7fb3','#3a8f5e','#9a4f9c','#c2553a','#5a7d3a','#2f8f8f','#a06a2a','#7a5fc0','#c0892a']
CLASSES=["T-shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Boot"]
PMAX=20; b=0.5; eps=0.05
# decode centroids from the exported sprite (10 rows=classes x 20 cols, class-major)
spr=np.asarray(Image.open(_ROOT/'public'/'yat-fmnist'/'centroids.png')).astype('float32')/255.0
C=np.zeros((10*PMAX,784),'float32'); Lab=np.zeros(10*PMAX,int)
for c in range(10):
    for j in range(PMAX):
        C[c*PMAX+j]=spr[c*28:c*28+28, j*28:j*28+28].reshape(784); Lab[c*PMAX+j]=c
te=torchvision.datasets.FashionMNIST('/tmp/fmnist',train=False,download=True)
Xte=jnp.asarray(te.data.numpy().reshape(-1,784).astype('float32')/255.0); yte=te.targets.numpy().astype('int32')
def confusion_and_acc(npc):
    sel=np.concatenate([np.where(Lab==c)[0][:npc] for c in range(10)])
    W=jnp.asarray(C[sel]); Wn=jnp.sum(W**2,1); M=jnp.asarray(np.eye(10)[Lab[sel]])
    @jax.jit
    def f(Xb):
        dot=Xb@W.T; d2=jnp.sum(Xb**2,1,keepdims=True)+Wn-2*dot; return ((dot+b)**2/(d2+eps))@M
    pred=[]
    for i in range(0,len(yte),2000): pred.append(np.asarray(jnp.argmax(f(Xte[i:i+2000]),-1)))
    pred=np.concatenate(pred)
    CM=np.zeros((10,10))
    for t,p in zip(yte,pred): CM[t,p]+=1
    CM=CM/CM.sum(1,keepdims=True)
    return CM, float((pred==yte).mean())
data=[confusion_and_acc(n) for n in range(1,PMAX+1)]
print("acc by nper:",[round(a*100,1) for _,a in data])
def frame(idx):
    npc=idx+1; CM,acc=data[idx]
    fig=plt.figure(figsize=(10,5.0),dpi=96,facecolor=BG)
    fig.text(0.5,0.95,"Solving Fashion-MNIST with no training, one prototype at a time",ha='center',color=INK,fontsize=14.5,weight='bold')
    fig.text(0.5,0.905,"prototypes are labeled pictures; the readout just votes; 0 gradient steps",ha='center',color=MUTED,fontsize=10)
    # left: prototype bank
    axb=fig.add_axes([0.03,0.08,0.44,0.76])
    bank=np.ones((10*30, npc*30))
    for c in range(10):
        for j in range(npc):
            t=np.clip(C[c*PMAX+j].reshape(28,28),0,1); bank[c*30:c*30+28, j*30:j*30+28]=1-t  # invert for ink-on-light
    axb.imshow(bank,cmap='gray',vmin=0,vmax=1,aspect='auto'); axb.set_xticks([]); axb.set_yticks([])
    for c in range(10):
        axb.text(-0.6, c*30+14, CLASSES[c], ha='right', va='center', color=CLASS_COL[c], fontsize=8, transform=axb.transData)
    axb.set_title(f"{npc} prototype{'s' if npc>1 else ''} per class  ({npc*10} labeled pictures)",color=INK,fontsize=10,weight='bold',pad=4)
    for sp in axb.spines.values(): sp.set_color(BORDER)
    # right: confusion matrix
    axc=fig.add_axes([0.58,0.10,0.36,0.62])
    axc.imshow(CM,cmap='Oranges',vmin=0,vmax=1)
    axc.set_xticks(range(10)); axc.set_yticks(range(10))
    axc.set_xticklabels([c[:4] for c in CLASSES],rotation=90,fontsize=6.5,color=MUTED); axc.set_yticklabels([c[:4] for c in CLASSES],fontsize=6.5,color=MUTED)
    axc.set_xlabel('predicted',color=MUTED,fontsize=9); axc.set_ylabel('true',color=MUTED,fontsize=9)
    for sp in axc.spines.values(): sp.set_color(BORDER)
    fig.text(0.76,0.80,f"{acc*100:.0f}%",ha='center',color=ACCENT,fontsize=30,weight='bold')
    fig.text(0.76,0.755,"test accuracy",ha='center',color=MUTED,fontsize=9)
    fig.canvas.draw(); a=np.asarray(fig.canvas.buffer_rgba()).copy(); plt.close(fig); return a
fr=[frame(i) for i in range(PMAX)]
Image.fromarray(fr[-1]).save(OUT/"notrain-confusion-preview.png")
imageio.mimsave(OUT/"notrain-confusion.gif",[fr[0]]*4+fr+[fr[-1]]*14,duration=0.42,loop=0,palettesize=96,subrectangles=True)
print("wrote notrain-confusion.gif", round((_ROOT/'public'/'notrain-confusion.gif').stat().st_size/1e6,2),"MB")
