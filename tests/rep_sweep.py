"""rep_sweep.py — which INPUT REPRESENTATION carries fact-identity signal?

The premise (operator, 2026-07-01): KSTE-MD amplifies input separability, so the
job is to find a representation that SEPARATES facts, then feed the pipeline THAT.
Uses the on-disk gemma-4-12B captures:
  qdump/q_*.bin      = exact-query  last-token global-Q  [ng, G_NH, HD]
  qdump_para/q_*.bin = paraphrase-query last-token global-Q  [ng, G_NH, HD]
  qdump/k_*.bin      = episode global-K  [ng, npos, HD]   (MULTI-token -> content-bearing)

Three experiments (cosine-based; fast):
  E1  query-pooling sweep: exact_i -> its paraphrase_i recall@1 + same/diff gap.
      (can last-token global-Q be rescued by any pooling/normalization? prior: no.)
  E2  episode-pooling recall: exact-query_i -> episode K_i, various K poolings.
  E3  CEILING probe: split each episode K's positions in half, pool each half,
      self-recall Ka_i -> Kb_j. Does a MULTI-TOKEN content-bearing rep carry fact
      identity? If yes -> capturing a pooled QUERY will pay off (the engine change).

Usage: python rep_sweep.py <exact_dir> <para_dir>
"""
import os, sys, glob, struct
import numpy as np
np.seterr(all="ignore")

def load(path):
    b = open(path, "rb").read()
    ng, d = struct.unpack("<II", b[:8])
    a = np.frombuffer(b[8:], "<f4").astype(np.float64)
    return ng, d, a

def load_dir_q(d):
    out = {}
    for p in glob.glob(os.path.join(d, "q_*.bin")):
        cid = int(os.path.basename(p)[2:-4]); ng, dd, a = load(p)
        out[cid] = a.reshape(ng, -1, 512)   # [ng, G_NH, HD]
    return [out[c] for c in sorted(out)]

def load_dir_k(d):
    out = {}
    for p in glob.glob(os.path.join(d, "k_*.bin")):
        cid = int(os.path.basename(p)[2:-4]); ng, dd, a = load(p)
        out[cid] = a.reshape(ng, -1, 512)   # [ng, npos, HD]
    return [out[c] for c in sorted(out)]

def cos_mat(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-30)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-30)
    return An @ Bn.T

def recall1(sim):
    n = sim.shape[0]
    return float(np.mean(np.argmax(sim, axis=1) == np.arange(n)))

def gap(sim):
    n = sim.shape[0]; d = np.diag(sim)
    off = sim[~np.eye(n, dtype=bool)]
    return float(d.mean()), float(off.mean()), float(d.mean() - off.mean())

def stack(vecs, fn):
    return np.stack([fn(v).ravel() for v in vecs], axis=0)

def main():
    ex_d, pa_d = sys.argv[1], sys.argv[2]
    EX, PA, K = load_dir_q(ex_d), load_dir_q(pa_d), load_dir_k(ex_d)
    n = min(len(EX), len(PA), len(K)); EX, PA, K = EX[:n], PA[:n], K[:n]
    print(f"=== REP-SWEEP  n={n}  Q shape {EX[0].shape}  K shape {K[0].shape} ===\n")

    # ---- query pooling variants (input: [ng,G_NH,HD]) ----
    def flat(v):        return v
    def mean_heads(v):  return v.mean(axis=1)
    def mean_layers(v): return v.mean(axis=0)
    def mean_both(v):   return v.mean(axis=(0, 1))
    def l2_mean_both(v):
        vn = v / (np.linalg.norm(v, axis=2, keepdims=True) + 1e-30)
        return vn.mean(axis=(0, 1))
    def last_layer(v):  return v[-1].mean(axis=0)
    variants = [("flat65536(baseline)", flat), ("mean_heads", mean_heads),
                ("mean_layers", mean_layers), ("mean_both", mean_both),
                ("l2norm_then_mean_both", l2_mean_both), ("last_layer_mean_heads", last_layer)]

    print("[E1] QUERY pooling — exact_i -> its paraphrase_i (recall@1 over %d; floor %.1f%%)" % (n, 100.0/n))
    print("     variant                     recall@1   same-cos  diff-cos   gap")
    for name, fn in variants:
        A, B = stack(EX, fn), stack(PA, fn)
        s = cos_mat(A, B); sm, dm, g = gap(s)
        print(f"     {name:26s}  {100*recall1(s):6.1f}%   {sm:7.4f}  {dm:7.4f}  {g:+.4f}")
    # per-layer best (mean over heads within one global layer)
    best = (-1, -1.0)
    for L in range(EX[0].shape[0]):
        A = np.stack([EX[i][L].mean(0) for i in range(n)]); B = np.stack([PA[i][L].mean(0) for i in range(n)])
        r = recall1(cos_mat(A, B))
        if r > best[1]: best = (L, r)
    print(f"     best single global-layer: L={best[0]}  recall@1={100*best[1]:.1f}%")

    # ---- E2: query -> episode recall, episode pooled various ways ----
    print("\n[E2] EPISODE pooling — exact-query_i -> episode_i (recall@1; query=mean_heads->[ng,HD] flat)")
    Q2 = stack(EX, lambda v: v.mean(axis=1))          # [n, ng*HD]
    def k_mean(v):  return v.mean(axis=1)             # mean over npos -> [ng,HD]
    def k_max(v):   return v.max(axis=1)
    def k_last(v):  return v[:, -1, :]
    def k_meanpos_meanlayer(v): return v.mean(axis=(0, 1))  # [HD]
    for name, fn in [("K mean_pos", k_mean), ("K max_pos", k_max), ("K last_pos", k_last)]:
        Kr = stack(K, fn)
        if Kr.shape[1] == Q2.shape[1]:
            print(f"     {name:16s}  recall@1={100*recall1(cos_mat(Q2, Kr)):6.1f}%")
    # query mean_both vs K mean_both (both [HD])
    Qb = stack(EX, lambda v: v.mean(axis=(0, 1))); Kb = stack(K, k_meanpos_meanlayer)
    print(f"     mean_both Q vs mean_both K:   recall@1={100*recall1(cos_mat(Qb, Kb)):6.1f}%")

    # ---- E3: CEILING probe — does a MULTI-TOKEN content rep carry fact identity? ----
    print("\n[E3] CEILING — split each episode K's positions in half, pool each, self-recall Ka_i -> Kb_j")
    Ka, Kb = [], []
    for v in K:
        npos = v.shape[1]; h = max(1, npos // 2)
        Ka.append(v[:, :h, :].mean(axis=1).ravel())
        Kb.append(v[:, h:, :].mean(axis=1).ravel() if npos > 1 else v[:, :, :].mean(axis=1).ravel())
    Ka, Kb = np.stack(Ka), np.stack(Kb)
    s = cos_mat(Ka, Kb); sm, dm, g = gap(s)
    print(f"     mean-pooled half-episode self-recall@1 = {100*recall1(s):.1f}%   same-cos {sm:.4f}  diff-cos {dm:.4f}  gap {g:+.4f}")
    # also L2-normalized per (layer,pos) before pooling
    Ka2, Kb2 = [], []
    for v in K:
        vn = v / (np.linalg.norm(v, axis=2, keepdims=True) + 1e-30)
        npos = vn.shape[1]; h = max(1, npos // 2)
        Ka2.append(vn[:, :h, :].mean(axis=1).ravel()); Kb2.append(vn[:, h:, :].mean(axis=1).ravel() if npos > 1 else vn.mean(axis=1).ravel())
    Ka2, Kb2 = np.stack(Ka2), np.stack(Kb2)
    s2 = cos_mat(Ka2, Kb2); sm2, dm2, g2 = gap(s2)
    print(f"     L2-normalized variant:                 self-recall@1 = {100*recall1(s2):.1f}%   gap {g2:+.4f}")

    print("\nINTERPRETATION:")
    print("  E1 high -> last-token query is salvageable by pooling (surprise).")
    print("  E3 high -> a MULTI-TOKEN pooled rep DOES carry fact identity => capturing a pooled")
    print("            QUERY will pay off (justifies the engine change). E3 low -> deeper problem.")

if __name__ == "__main__":
    main()
