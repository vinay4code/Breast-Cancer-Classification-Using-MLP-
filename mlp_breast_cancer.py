import numpy as np, json
from sklearn.datasets import load_breast_cancer
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = load_breast_cancer()
X = data.data.astype(float)
y = (1 - data.target).reshape(-1,1).astype(float)   # 1 = malignant (positive), 0 = benign

def split(X,y,test_frac,seed=42):
    rng=np.random.RandomState(seed); tr=[];te=[]
    for c in (0,1):
        idx=np.where(y[:,0]==c)[0]; rng.shuffle(idx)
        n=int(round(len(idx)*test_frac)); te+=list(idx[:n]); tr+=list(idx[n:])
    rng.shuffle(tr); rng.shuffle(te)
    return X[tr],X[te],y[tr],y[te]

def relu(z): return np.maximum(0,z)
def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))
def bce(y,p):
    p=np.clip(p,1e-12,1-1e-12); return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))

def init(h,seed=0):
    r=np.random.RandomState(seed)
    return dict(W1=r.randn(30,h)*np.sqrt(2/30), b1=np.zeros((1,h)),
                W2=r.randn(h,1)*np.sqrt(2/h), b2=np.zeros((1,1)))

def forward(P,X):
    Z1=X@P["W1"]+P["b1"]; A1=relu(Z1); Z2=A1@P["W2"]+P["b2"]; A2=sigmoid(Z2)
    return Z1,A1,A2

def train(Xtr,ytr,h,lr=0.1,epochs=1000):
    P=init(h); P0={k:v.copy() for k,v in P.items()}; m=len(Xtr); losses=[]
    for e in range(epochs):
        Z1,A1,A2=forward(P,Xtr); losses.append(bce(ytr,A2))
        dZ2=(A2-ytr)/m                      # dL/dZ2 for sigmoid+BCE
        dW2=A1.T@dZ2; db2=dZ2.sum(0,keepdims=True)
        dA1=dZ2@P["W2"].T; dZ1=dA1*(Z1>0)
        dW1=Xtr.T@dZ1; db1=dZ1.sum(0,keepdims=True)
        P["W1"]-=lr*dW1; P["b1"]-=lr*db1; P["W2"]-=lr*dW2; P["b2"]-=lr*db2
    return P0,P,losses

def metrics(y,p):
    yp=(p>=0.5).astype(int).ravel(); yt=y.ravel().astype(int)
    tp=int(((yp==1)&(yt==1)).sum()); tn=int(((yp==0)&(yt==0)).sum())
    fp=int(((yp==1)&(yt==0)).sum()); fn=int(((yp==0)&(yt==1)).sum())
    acc=(tp+tn)/len(yt); pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0
    f1=2*pr*rc/(pr+rc) if pr+rc else 0
    return dict(acc=acc,prec=pr,rec=rc,f1=f1,tn=tn,fp=fp,fn=fn,tp=tp)

res=[]; store={}
for tf,name in ((0.30,"70:30"),(0.20,"80:20")):
    Xtr,Xte,ytr,yte=split(X,y,tf)
    mu,sd=Xtr.mean(0),Xtr.std(0)
    Xtr_s=(Xtr-mu)/sd; Xte_s=(Xte-mu)/sd
    for h in (8,15,25):
        P0,P,L=train(Xtr_s,ytr,h)
        mt=metrics(ytr,forward(P,Xtr_s)[2]); me=metrics(yte,forward(P,Xte_s)[2])
        res.append(dict(split=name,h=h,ntr=len(Xtr),nte=len(Xte),loss0=L[0],lossN=L[-1],train_acc=mt["acc"],**me))
        store[f"{name}_{h}"]=dict(P0=P0,P=P,L=L)
        print(name,h,len(Xtr),len(Xte),round(L[0],4),round(L[-1],4),round(mt["acc"],4),{k:round(v,4) if isinstance(v,float) else v for k,v in me.items()})
json.dump(res,open("res.json","w"))
np.savez("weights.npz",**{f"{k}_{n}_{w}":v[n][w] for k,v in store.items() for n in("P0","P") for w in v[n]})
np.savez("losses.npz",**{k:np.array(v["L"]) for k,v in store.items()})

# plots
fig,ax=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
for a,name in zip(ax,("70:30","80:20")):
    for h in (8,15,25): a.plot(store[f"{name}_{h}"]["L"],label=f"30-{h}-1")
    a.set_title(f"Training loss ({name} split)"); a.set_xlabel("Epoch"); a.grid(alpha=.3); a.legend()
ax[0].set_ylabel("BCE loss"); plt.tight_layout(); plt.savefig("loss.png",dpi=170); plt.close()
fig,ax=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
for a,name in zip(ax,("70:30","80:20")):
    for h in (8,15,25): a.plot(store[f"{name}_{h}"]["L"],label=f"30-{h}-1")
    a.set_yscale("log"); a.set_title(f"Loss, log scale ({name})"); a.set_xlabel("Epoch"); a.grid(alpha=.3,which="both"); a.legend()
plt.tight_layout(); plt.savefig("loss_log.png",dpi=170); plt.close()
# confusion matrices
fig,axs=plt.subplots(2,3,figsize=(9,6))
for i,name in enumerate(("70:30","80:20")):
    for j,h in enumerate((8,15,25)):
        r=[q for q in res if q["split"]==name and q["h"]==h][0]
        cm=np.array([[r["tn"],r["fp"]],[r["fn"],r["tp"]]]); a=axs[i,j]
        a.imshow(cm,cmap="Blues")
        for (u,v),val in np.ndenumerate(cm): a.text(v,u,val,ha="center",va="center",color="white" if val>cm.max()/2 else "black",fontsize=13)
        a.set_xticks([0,1]);a.set_yticks([0,1]);a.set_xticklabels(["Benign","Malig."]);a.set_yticklabels(["Benign","Malig."])
        a.set_title(f"30-{h}-1, {name}",fontsize=10)
        if j==0:a.set_ylabel("Actual")
        if i==1:a.set_xlabel("Predicted")
plt.tight_layout(); plt.savefig("cm.png",dpi=170); plt.close()
# weight change plot (30-15-1, 80:20 W1 init vs final)
s=store["80:20_15"]
fig,ax=plt.subplots(1,3,figsize=(11,3.4))
for a,M,t in zip(ax,(s["P0"]["W1"],s["P"]["W1"],s["P"]["W1"]-s["P0"]["W1"]),("W1 initial","W1 updated","Change (updated - initial)")):
    im=a.imshow(M,aspect="auto",cmap="coolwarm",vmin=-abs(M).max(),vmax=abs(M).max()); a.set_title(t,fontsize=10); a.set_xlabel("Hidden neuron"); a.set_ylabel("Input feature"); plt.colorbar(im,ax=a)
plt.tight_layout(); plt.savefig("w1heat.png",dpi=170); plt.close()
print(data.feature_names[:3], (y==1).sum(), (y==0).sum())
