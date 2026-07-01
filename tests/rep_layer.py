"""rep_layer.py — nail the layer-localization finding + prep the KSTE-MD confirm.

E1b per-layer exact->para recall + gap; E1c greedy layer-subset; E2b per-layer
query->episode recall (the actual selector task, layer L on both sides);
E3b per-layer episode self-recall (ceiling). Dumps the best-layer representation
vectors to /tmp/spgate/*.i32 (int32, scaled) for the C KSTE-MD confirm.

Usage: python rep_layer.py <exact_dir> <para_dir>
"""
import os, sys, glob, struct
import numpy as np
np.seterr(all="ignore")

def load(p):
    b=open(p,"rb").read(); ng,d=struct.unpack("<II",b[:8]); return ng,np.frombuffer(b[8:],"<f4").astype(np.float64)
def dq(d):
    o={};
    for p in glob.glob(os.path.join(d,"q_*.bin")):
        c=int(os.path.basename(p)[2:-4]); ng,a=load(p); o[c]=a.reshape(ng,-1,512)
    return [o[c] for c in sorted(o)]
def dk(d):
    o={}
    for p in glob.glob(os.path.join(d,"k_*.bin")):
        c=int(os.path.basename(p)[2:-4]); ng,a=load(p); o[c]=a.reshape(ng,-1,512)
    return [o[c] for c in sorted(o)]
def cm(A,B):
    A=A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-30); B=B/(np.linalg.norm(B,axis=1,keepdims=True)+1e-30); return A@B.T
def r1(s): n=s.shape[0]; return float(np.mean(np.argmax(s,axis=1)==np.arange(n)))
def gp(s): n=s.shape[0]; d=np.diag(s); o=s[~np.eye(n,dtype=bool)]; return d.mean(),o.mean(),d.mean()-o.mean()

ex_d,pa_d=sys.argv[1],sys.argv[2]
EX,PA,K=dq(ex_d),dq(pa_d),dk(ex_d); n=min(len(EX),len(PA),len(K)); EX,PA,K=EX[:n],PA[:n],K[:n]
NG=EX[0].shape[0]
print(f"=== REP-LAYER n={n} NG={NG} Q{EX[0].shape} K{K[0].shape} ===\n")

print("[E1b] per global-layer: exact_i -> paraphrase_i (mean over heads -> [512])")
print("      L   recall@1   same-cos diff-cos  gap")
perL=[]
for L in range(NG):
    A=np.stack([EX[i][L].mean(0) for i in range(n)]); B=np.stack([PA[i][L].mean(0) for i in range(n)])
    s=cm(A,B); sm,dm,g=gp(s); perL.append((L,r1(s),g)); print(f"      {L}   {100*r1(s):6.1f}%   {sm:.4f}  {dm:.4f}  {g:+.4f}")
bestL=max(perL,key=lambda t:t[1])[0]
print(f"      -> best layer L={bestL}")

print("\n[E1c] greedy layer-subset (concat mean-heads of chosen layers), exact->para recall@1")
chosen=[]; cur=0.0
remaining=list(range(NG))
for _ in range(NG):
    bestadd=None
    for L in remaining:
        cols=chosen+[L]
        A=np.concatenate([np.stack([EX[i][c].mean(0) for i in range(n)]) for c in cols],axis=1)
        B=np.concatenate([np.stack([PA[i][c].mean(0) for i in range(n)]) for c in cols],axis=1)
        r=r1(cm(A,B))
        if bestadd is None or r>bestadd[1]: bestadd=(L,r)
    if bestadd[1]<=cur+1e-9 and chosen: break
    chosen.append(bestadd[0]); remaining.remove(bestadd[0]); cur=bestadd[1]
    print(f"      + L{bestadd[0]} -> layers{chosen} recall@1={100*cur:.1f}%")

print("\n[E2b] ACTUAL SELECTOR task: query_i(layer L, mean-heads) -> episode_i(layer L, mean-pos)")
print("      L   recall@1")
for L in range(NG):
    Q=np.stack([EX[i][L].mean(0) for i in range(n)]); Kk=np.stack([K[i][L].mean(0) for i in range(n)])
    print(f"      {L}   {100*r1(cm(Q,Kk)):6.1f}%")

print("\n[E3b] episode self-recall per layer (half-pos split) — content-bearing ceiling")
print("      L   self-recall@1")
for L in range(NG):
    Ka=[];Kb=[]
    for v in K:
        np_=v.shape[1]; h=max(1,np_//2); Ka.append(v[L,:h,:].mean(0)); Kb.append(v[L,h:,:].mean(0) if np_>1 else v[L].mean(0))
    print(f"      {L}   {100*r1(cm(np.stack(Ka),np.stack(Kb))):6.1f}%")

# dump best-layer mean-heads vectors (exact, para) as int32 for the C KSTE-MD confirm
def dump(vecs, L, path):
    M=np.stack([vecs[i][L].mean(0) for i in range(n)])          # [n,512]
    q=np.clip(M*1e6,-2.147e9,2.147e9).astype(np.int32)
    with open(path,"wb") as f:
        f.write(struct.pack("<II",n,512)); f.write(q.tobytes())
outdir=os.environ.get("SPGATE","/tmp/spgate")
dump(EX,bestL,os.path.join(outdir,"L_exact.i32")); dump(PA,bestL,os.path.join(outdir,"L_para.i32"))
print(f"\nDUMPED best-layer L={bestL} exact/para int32 -> {outdir}/L_exact.i32, L_para.i32  (for C KSTE-MD confirm)")
