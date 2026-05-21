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

**Phase 0 — empty.** Build environment set up, README + LICENSE + .gitignore in place. Phase 1 lands the KSTE encoder with tests T1.1–T1.5.

## Build (placeholder)

```bash
cmake -B build -G Ninja
cmake --build build
ctest --test-dir build
```

CMake stubs are in place; no source files yet.

## License

AGPL-3.0-or-later. See `LICENSE`. Commercial licensing available — contact the copyright holder.
