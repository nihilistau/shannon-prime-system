# shannon-prime-system

Clean from-scratch math core for the [shannon-prime-lattice](../shannon-prime-lattice) architecture.

Provides (when complete):

- **KSTE encoder** — deterministic, Frobenius-invariant map from K-vectors in $\mathbb{R}^d$ to packed 64-byte trees in $T_{60,3}$. VHT2 + Möbius reorder + anchor / residual.
- **Dominance order $\preceq_d$ + Friedman sieve** — Kruskal-Friedman homeomorphic embedding on packed trees; signature-prefilter; dedup cache.
- **ARM (Algebraic Resonance Memory)** — HRR binding/unbinding in the CRT cyclotomic ring $\mathbb{Z}_q[x]/(x^N+1)$. Bounded-state cross-node aggregation primitive.
- **CRT NTT primitives** — dual-prime forward/inverse NTT with Barrett reduction and CRT recombination.
- **Position-as-Arithmetic helpers** — prime-factorization mapping for sequence and network coordinates.

## Not a fork

This is **not** a fork or extension of the older `shannon-prime/` repo. It is a clean rebuild containing only the primitives used by `shannon-prime-lattice`. The math comes from the same papers, but every line of code is written fresh in this project.

The phasing in `../shannon-prime-lattice/papers/PPT-LAT-Roadmap.md` defines what gets built when.

## Status

**Phase 1 — math core, in progress.** Each primitive is a self-contained module under `core/<module>/` with its own tests; the EXISTS-guarded root build picks up whatever modules exist. See `CONVENTIONS.md` for the module template and `../shannon-prime-lattice/papers/PPT-LAT-Roadmap.md` §7 for the per-subphase contracts.

| Subphase | Module | Tests | Tier-1 (Win MinGW-gcc) |
|----------|--------|-------|------------------------|
| 1A | `core/ok_arith` — O_K arithmetic over Q(√-163) | T_OK_1..6 | ✅ green |
| 1B | `core/ntt_crt` — dual-prime CRT-NTT (N ∈ {128,256,512}) | T_NTT_1..5 | in progress |
| 1C | `core/poly_ring` — R_q polynomial-ring attention | T_PR_1..4 | pending (needs 1B) |
| 1D | `core/vht2` — VHT2 + Möbius + 63-byte Spinor block | T_VHT_1..6 | ✅ green |
| 1E | `core/frobenius` — Frobenius lift for Q8 weights | T_FRO_1..3 (T_FRO_4 → Phase 2) | in progress |
| 1F | `core/kste` — KSTE encoder + Tier-0/Tier-1 dominance | T_KSTE_1..5 | ✅ green |

Platform-gate policy is staged (roadmap §3.7): Tier-1 = Windows MinGW-gcc (closes in-session), Tier-2 = Linux gcc via CI (`.github/workflows/ci.yml`), Tier-3 = Windows MSVC (follow-up wave).

## Build

MinGW gcc 15.2 + Ninja:

```bash
# whole repo
cmake -B build -G Ninja -DCMAKE_C_COMPILER=gcc
cmake --build build
ctest --test-dir build --output-on-failure

# a single module (fast iteration)
cmake -B core/<m>/build -S core/<m> -G Ninja -DCMAKE_C_COMPILER=gcc
ctest --test-dir core/<m>/build --output-on-failure
```

Run once with `-DSP_UBSAN=ON` before closing a module (auto-falls back to `-fsanitize-undefined-trap-on-error` where libubsan is unavailable, e.g. MinGW-Builds).

## License

AGPL-3.0-or-later. See `LICENSE`. Commercial licensing available — contact the copyright holder.
