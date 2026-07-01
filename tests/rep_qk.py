"""rep_qk.py — port of recall.rs qk_relevance; does restricting it to L5 do Q->episode recall?

The live selector ranks episodes by qk_relevance(query global-Q, episode global-K): for each
(global layer, head, position) score = q_head . K[pos] (raw*1/sqrt(HD), or cosine if SP_B3_QK_COSINE),
summarised as top-m mean. It sums over ALL global layers. The L5 finding says the signal is in L5 and
averaging dilutes. Test recall@1 (incoming query i -> its episode K_i) for layer subsets x {raw,cosine}.

Usage: python rep_qk.py <exact_dir> <para_dir>
"""
import os, sys, glob, struct
import numpy as np
np.seterr(all="ignore")
HD, G_NH = 512, 16
def load(p):
    b=open(p,"rb").read(); ng,d=struct.unpack("<II",b[:8]); return ng,np.frombuffer(b[8:],"<f4").astype(np.float64)
def dq(d):
    o={}
    for p in glob.glob(os.path.join(d,"q_*.bin")):
        c=int(os.path.basename(p)[2:-4]); ng,a=load(p); o[c]=a.reshape(ng,G_NH,HD)
    return [o[c] for c in sorted(o)]
def dk(d):
    o={}
    for p in glob.glob(os.path.join(d,"k_*.bin")):
        c=int(os.path.basename(p)[2:-4]); ng,a=load(p); o[c]=a.reshape(ng,-1,HD)
    return [o[c] for c in sorted(o)]

def qk_topm(q, k, layers, m=8, cosine=True):
    """q [ng,G_NH,HD], k [ng,npos,HD] -> top-m mean score over (layer in layers, head, pos)."""
    Q = q[layers]                        # [nL,16,HD]
    K = k[layers]                        # [nL,npos,HD]
    if cosine:
        Q = Q/(np.linalg.norm(Q,axis=2,keepdims=True)+1e-30)
        K = K/(np.linalg.norm(K,axis=2,keepdims=True)+1e-30)
    # scores[l,h,p] = Q[l,h].K[l,p]
    s = np.einsum("lhd,lpd->lhp", Q, K)  # [nL,16,npos]
    flat = np.sort(s.ravel())[::-1]
    mm = min(m, flat.size)
    return float(flat[:mm].mean())

def recall(Qs, Ks, layers, m, cosine):
    n=len(Qs); hit=0
    for i in range(n):
        best=-1; bv=-1e18
        for j in range(n):
            v=qk_topm(Qs[i], Ks[j], layers, m, cosine)
            if v>bv: bv=v; best=j
        if best==i: hit+=1
    return 100.0*hit/n

ex,pa = sys.argv[1], sys.argv[2]
EX,PA,K = dq(ex),dq(pa),dk(ex); n=min(len(EX),len(PA),len(K)); EX,PA,K=EX[:n],PA[:n],K[:n]
NG=EX[0].shape[0]
print(f"=== REP-QK  n={n} NG={NG}  qk_relevance recall@1 (query -> its episode K), floor {100/n:.1f}% ===\n")
allL=list(range(NG))
configs=[("all-layers raw",   allL,      False),
         ("all-layers cosine",allL,      True),
         ("L5 raw",           [5],       False),
         ("L5 cosine",        [5],       True),
         ("L5+L6 cosine",     [5,6],     True),
         ("L4-6 cosine",      [4,5,6],   True)]
print(f"{'config':20s}  exact->K   para->K   (m=8)")
for name,ls,cos in configs:
    re_=recall(EX,K,ls,8,cos); rp_=recall(PA,K,ls,8,cos)
    print(f"{name:20s}  {re_:6.1f}%   {rp_:6.1f}%")
print("\n(if L5/L5+L6 cosine >> all-layers => restrict the live qk_relevance to L5 = the drop-in change)")
