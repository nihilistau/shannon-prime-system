# STATUS — shannon-prime-system

**Date:** 2026-08-29  
**Class:** `STANDING` — exact-integer math core + frozen L1 C ABI. **Not absorbed by Kairos.**

Unique work that lives here:

- `O_K = Z[(1+√−163)/2]` arithmetic
- Dual-prime CRT-NTT (`ntt_crt.h`), Frobenius codec, exact islands (RMS/softmax/GELU/CORDIC-RoPE)
- ARM two-ring KV, C2 router, Ring-3 VSA
- Frozen L1 C ABI (`include/sp/sp_l1.h`)

Consumed by [shannon-prime-system-engine](https://github.com/nihilistau/shannon-prime-system-engine) as `lib/shannon-prime-system`.
Canon / status: [shannon-prime-lattice](https://github.com/nihilistau/shannon-prime-lattice).

Kairos is a Python companion harness. It does not own this ABI or these kernels.
