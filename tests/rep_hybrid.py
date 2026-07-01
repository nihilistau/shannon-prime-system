"""rep_hybrid.py — does the L5 method-change close the faithfulness paraphrase gap?

Current deployed selector = Jaccard (token overlap): 100% on exact queries, ~floor on
paraphrases. The L5 finding: layer-5 query embedding recalls paraphrases at ~85% (cosine).
This tests the DEPLOYABLE change: Jaccard + L5-cosine hybrid, on the real 61 fact-conflicts.

Recall task: incoming query -> its own fact i, over 61 facts.
  - EXACT incoming  (query text ~ fact wording): Jaccard should ace it.
  - PARA  incoming  (anti-lexical paraphrase):    Jaccard collapses; L5 rescues.
Keys: each fact stored with (fact_text for Jaccard, exact-query L5 embedding for cosine).

Usage: python rep_hybrid.py <exact_dir> <para_dir> <facts.json>
"""
import os, sys, glob, struct, json, re
import numpy as np
np.seterr(all="ignore")
L5 = 5

def load(p):
    b=open(p,"rb").read(); ng,d=struct.unpack("<II",b[:8]); return ng,np.frombuffer(b[8:],"<f4").astype(np.float64)
def dq(d):
    o={}
    for p in glob.glob(os.path.join(d,"q_*.bin")):
        c=int(os.path.basename(p)[2:-4]); ng,a=load(p); o[c]=a.reshape(ng,-1,512)
    return [o[c] for c in sorted(o)]
def l5emb(vecs):
    return np.stack([v[L5].mean(0) for v in vecs])   # [n,512]
def toks(s): return set(re.findall(r"[a-z0-9]+", s.lower()))
def jac(a,b):
    A,B=toks(a),toks(b); return len(A&B)/max(1,len(A|B))
def cnorm(M):
    return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-30)
def r1(sim): n=sim.shape[0]; return float(np.mean(np.argmax(sim,axis=1)==np.arange(n)))
def zrows(M):
    return (M-M.mean(1,keepdims=True))/(M.std(1,keepdims=True)+1e-9)

ex_d,pa_d,fj=sys.argv[1],sys.argv[2],sys.argv[3]
EX,PA=dq(ex_d),dq(pa_d); F=json.load(open(fj,encoding="utf-8"))
n=min(len(EX),len(PA),len(F)); EX,PA,F=EX[:n],PA[:n],F[:n]
Ex5,Pa5=cnorm(l5emb(EX)),cnorm(l5emb(PA))
print(f"=== REP-HYBRID  n={n}  (Jaccard vs L5-cosine vs hybrid) ===\n")

# similarity matrices: rows = incoming query, cols = stored fact key
def jac_mat(incoming_field):
    return np.array([[jac(F[i][incoming_field], F[j]["fact"]) for j in range(n)] for i in range(n)])
# L5 cosine: incoming query embedding vs stored EXACT-query key embedding
L5_exact_key = Ex5
def l5_mat(incoming_emb):
    return incoming_emb @ L5_exact_key.T

for tag, field, emb in [("EXACT incoming", "q", Ex5), ("PARA incoming", "para", Pa5)]:
    J = jac_mat(field); Lc = l5_mat(emb)
    H = zrows(J) + zrows(Lc)                          # simple z-score sum hybrid
    print(f"[{tag}]  Jaccard={100*r1(J):5.1f}%   L5-cosine={100*r1(Lc):5.1f}%   HYBRID={100*r1(H):5.1f}%")

print("\n(current deployed = Jaccard column. The method change adds the L5 column;")
print(" HYBRID should hold ~100% on exact AND rescue paraphrase where Jaccard collapses.)")
