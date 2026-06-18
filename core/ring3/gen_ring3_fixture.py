#!/usr/bin/env python3
# gen_ring3_fixture.py — emit the T_RING3_NATIVE fixture: the canonical 40
# NIGHTSHIFT seeds (2 real-episode + 38 synthetic, identical to
# g_r3_nightshift.main) and a set of bind/unbind/superpose reference vectors
# computed through the NATIVE sp_pr_mul (via ok_bind, the Python reference the C
# core is being gated against). All ±1 carriers use the canonical splitmix64
# (ok_bind.carrier/idvec, now unified with core/ring3 sp_r3_carrier/idvec).
#
# The C gate (T_RING3_NATIVE) loads this file and asserts:
#  (a) C sp_r3_bind/unbind/superpose == these vectors, BIT-IDENTICAL (int64),
#  (b) C sp_r3_nightshift on these seeds reproduces the Python verdict ([32,8]).
#
# Fixture format (all little-endian):
#   magic   u32  = 0x52334E54 ('R3NT')
#   version u32  = 1
#   n_seeds u32        seeds   i64[n_seeds]
#   n_cases u32
#   per case: D u32 ; seed_a i64 ; seed_b i64 ;
#             bind   i64[D]   (= ok.bind(carrier(a), idvec(a)))
#             unbind i64[D]   (= ok.unbind(bind, carrier(a)))
#             super2 i64[D]   (= bind(a) + bind(b), two-episode superposition)
import os, sys, struct
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
ENG=os.path.normpath(os.path.join(HERE,"..","..","..","shannon-prime-system-engine"))
R3=os.path.join(ENG,"tools","ring3")
sys.path.insert(0, R3)
os.environ.setdefault("SP_R3_LIB",
    os.path.normpath(os.path.join(HERE,"..","..","build","libsp.dll")))
import ok_bind as ok
import g_r3_nightshift as ns   # reuse its seed derivation verbatim

def canonical_seeds():
    realK={"ep_toy":(f"{ENG}/_p33_ep",16),"ep_wiki":(f"{ENG}/_c2_ep_wiki",84)}
    seeds=[]
    for n,(d,p) in realK.items():
        seeds.append(int(ns.ep_sig_seed(d,p)) & ((1<<64)-1))
    rng=np.random.default_rng(31337)
    for i in range(38):
        seeds.append(int(rng.integers(1,2**62)))
    return seeds

def main():
    out_path = os.path.join(HERE, "T_RING3_NATIVE_fixture.bin")
    seeds = canonical_seeds()
    # bind/unbind/superpose reference vectors over a couple of (D, seed) cases.
    cases=[]
    for D in (1024,128,256,512):
        a = seeds[0]; b = seeds[1]
        ca = ok.carrier(a, D); ia = ok.idvec(a, D)
        cb = ok.carrier(b, D); ib = ok.idvec(b, D)
        M  = ok.bind(ca, ia)
        est= ok.unbind(M, ca)
        s2 = ok.bind(ca, ia) + ok.bind(cb, ib)
        cases.append((D,a,b,np.asarray(M,dtype=np.int64),
                      np.asarray(est,dtype=np.int64),np.asarray(s2,dtype=np.int64)))
    with open(out_path,"wb") as f:
        f.write(struct.pack("<II",0x52334E54,1))
        f.write(struct.pack("<I",len(seeds)))
        for s in seeds: f.write(struct.pack("<q", s if s < (1<<63) else s-(1<<64)))
        f.write(struct.pack("<I",len(cases)))
        for (D,a,b,M,est,s2) in cases:
            f.write(struct.pack("<I",D))
            f.write(struct.pack("<q", a if a<(1<<63) else a-(1<<64)))
            f.write(struct.pack("<q", b if b<(1<<63) else b-(1<<64)))
            M.tofile(f); est.tofile(f); s2.tofile(f)
    print(f"[fixture] wrote {out_path}: {len(seeds)} seeds, {len(cases)} bind/unbind cases")
    print(f"[fixture] seeds[0..4] = {[hex(s) for s in seeds[:5]]}")
    return 0

if __name__=="__main__":
    sys.exit(main())
