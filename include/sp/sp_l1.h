/* sp_l1.h -- the frozen Layer-1 C ABI of libshannonprime (the math core): the
 * session + forward + arch-query surface of PPT-LAT-L1-ABI-v0 (Appendix A). The
 * model-handle/loader half (opaque sp_model, sp_model_load/unload, the mmap
 * accessors, sp_model_to_qwen3) lives alongside in sp/sp_model.h; the status enum +
 * sp_last_error live in sp/sp_status.h. The inference code these declare is written
 * here in the math core and run by every consumer -- the four backends and any L2
 * binding -- through this one ABI; nothing duplicates the forward.
 *
 * The boundary this header freezes is the seam between L1 (the math core,
 * libshannonprime -- C/CUDA/Vulkan/HVX) and any L2 binding (Rust first). Load-bearing
 * contract principles, each of which a function signature below encodes:
 *
 *  - Caller-allocates on the hot path. The only L1-owned memory crossing the FFI is
 *    the opaque sp_model / sp_session handles, each with a matched destroyer. Every
 *    logits buffer is caller-allocated and sized from sp_arch_info.vocab_size; L1
 *    mallocs nothing L2 has to free. There is no sp_alloc/sp_free pair.
 *  - The model is immutable after load (many sessions per model; the L2 wrapper is
 *    Send + Sync). The session is the single-thread mutable state -- KV cache now,
 *    ARM + sieve banks later, plus arch scratch -- Send but NOT Sync (no two threads
 *    inside a forward call on one session; L2 enforces it with &mut self).
 *  - Cancellation is inverted to an L2-owned atomic flag. L1 never allocates a cancel
 *    object (which would open a use-after-free window when a session is destroyed on
 *    one thread while a stale cancel handle is poked on another). L2 owns the atomic;
 *    L1 only reads it (relaxed) at layer boundaries and unwinds to the last completed
 *    boundary returning SP_ECANCEL, leaving session state consistent.
 *  - The forward is two functions because prefill (compute-bound, a chunk of tokens)
 *    and decode (bandwidth-bound, one token) have asymmetric cost shapes that L2 must
 *    interleave across requests.
 *  - Clone / rewind are speculative-decoding primitives from day one: clone forks an
 *    independent session (deep-copy KV/ARM/sieve), rewind rolls back accepted tokens
 *    (ARM writes are journaled per-token so rewind is precise).
 *  - Determinism is fixed at session-create (serial reductions, single stream). The
 *    deterministic mode is what the T_FRO_4 bit-exact gate -- and the first parity
 *    gate for this ABI (sp_prefill_chunk == the reference forward's last-position
 *    logits) -- runs against. Toggling mid-session is forbidden; reduction order and
 *    stream topology bake into kernel selection at create time.
 *
 * Errors are sp_status (sp/sp_status.h); the discrete-algebra failure modes (Spinor /
 * VHT2 / Frobenius / NTT / ARM / sieve) have named codes so a tripped theorem
 * invariant maps to a status rather than collapsing to SP_EBADSTATE. Every failing
 * call also sets the thread-local sp_last_error() detail string.
 */
#ifndef SP_L1_H
#define SP_L1_H

#include <stdint.h>
#include <stddef.h>
#include "sp/sp_status.h"   /* sp_status, sp_last_error */
#include "sp/sp_model.h"    /* opaque sp_model + sp_model_load/unload (the model half) */

#ifdef __cplusplus
extern "C" {
#endif

/* Working-precision enum (2-L1.FP16). Stored as a fixed-width uint32_t in
 * sp_arch_info.preferred_precision and sp_session_config.precision_override (NOT as a
 * raw `enum` field — enum width is implementation-defined and these cross the
 * .sp-model wire / FFI boundary). */
enum sp_precision {
    SP_PRECISION_UNSPECIFIED = 0,
    SP_PRECISION_F32         = 1,
    SP_PRECISION_FP16        = 2,
    SP_PRECISION_QF32        = 3,
    /* SP_PRECISION_FP8     = 4   reserved for a future sub-phase (no kernel yet) */
};

/* ── §2 arch info -- caller-stack-allocated, L1 fills it in place ──
 * A public projection of the math core's model config. L2 reads this once at load to
 * size the logits buffer (vocab_size * sizeof(float)) and never re-queries. Rust
 * bindings should treat the out-parameter as &mut MaybeUninit<sp_arch_info>.
 *
 * Growth discipline: new fields are APPENDED in the reserved arch_struct tail
 * (PPT-LAT-SP-MODEL-v0 §3, arch_struct_capacity = 256). The loader (sp_model_load.c)
 * memcpy's min(arch_struct_size, sizeof(sp_arch_info)) bytes (sp_model_arch.c), so an
 * old .sp-model (smaller arch_struct_size) leaves the appended fields ZERO — that is
 * the "unspecified" sentinel for each. NOT an ABI version bump. sizeof MUST stay
 * <= 256 (compile-asserted in sp_model_arch.c). */
typedef struct {
    uint32_t arch_id;          /* sp_arch_id wire value (QWEN3=2, GEMMA3=3, ...)        */
    uint32_t vocab_size;       /* logits width; sizes the caller's logits buffer       */
    uint32_t hidden_dim;       /* embedding length (n_embd)                            */
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;       /* GQA grouping                                         */
    uint32_t head_dim;
    uint32_t max_context;
    uint32_t swa_window;       /* sliding-window size; 0 = full attention              */
    float    rope_freq_base;   /* global RoPE base                                     */
    uint8_t  ffn_variant;      /* 0 = SwiGLU (qwen3), 1 = GeGLU (gemma3)               */
    uint8_t  norm_variant;     /* 0 = pre-norm only, 1 = sandwich (gemma3 post-norms)  */
    uint8_t  tied_embeddings;  /* LM head reuses the token-embedding matrix            */
    uint8_t  has_qk_norm;      /* per-head Q/K RMSNorm present                         */

    /* ── appended 2-L1.FP16 (reserved arch_struct tail; 0 = unspecified) ── */
    uint32_t n_ff;                /* feed-forward width; 0 -> bridge derives from ffn_gate shape */
    float    rms_eps;             /* RMSNorm epsilon; 0.0 -> bridge defaults 1e-6 + load warning  */
    uint32_t preferred_precision; /* sp_precision; 0 -> SP_PRECISION_UNSPECIFIED (session falls back) */
} sp_arch_info;

/* Populate *out from the loaded model's header arch_struct. Caller-stack-allocated.
 * SP_OK on success; SP_EBADARG on a NULL handle/out. */
sp_status sp_model_arch(const sp_model *m, sp_arch_info *out);

/* sp_session_config.flags bits. Default 0 = persistent f32 KV (no compression).
 * 2-L1.PARITY inline KV-cache codec (the §4.9 frozen Spinor layout, sp/spinor_block.h): */
#define SP_KV_SPINOR     (1u << 0)   /* persistent COMPRESSED KV: VHT2+Mobius 63-byte Spinor blocks, decoded inline on read */
#define SP_KV_SPINOR_REF (1u << 1)   /* parity reference: f32 cache + in-place Spinor roundtrip (decode-from-block == this, by codec identity) */

/* ── §6 session config -- immutable for the session's lifetime ── */
typedef struct {
    size_t   max_context;     /* hard cap on sequence position; 0 = arch default       */
    int      deterministic;   /* nonzero: serial reductions, single stream (bit-exact) */
    uint32_t arm_bank_kb;     /* ARM HRR bank size; 0 = arch default (Phase 9+)         */
    uint32_t sieve_capacity;  /* Friedman-sieve frontier cap; 0 = arch default (Phase 5+) */
    uint32_t flags;           /* reserved bitfield (e.g. SP_VERIFY_TENSORS)            */
    uint32_t precision_override; /* sp_precision (2-L1.FP16). Working-precision precedence:
                                  *   override (non-zero) > arch_info.preferred_precision
                                  *   (non-zero) > SP_PRECISION_F32. 0 = defer to the model. */
} sp_session_config;

/* ── §1 opaque session handle -- per-thread KV (+ later ARM/sieve) + arch scratch ── */
typedef struct sp_session sp_session;

/* Create a session over an immutable model. `cancel_flag` is an L2-owned atomic the
 * session reads (relaxed) at layer boundaries: 0 = continue, nonzero = cancel. Its
 * storage MUST stay valid until sp_session_destroy returns (L2 holds it in an Arc).
 * Pass NULL for no cancellation. SP_OK / SP_ENOMEM / SP_EBADARG / SP_ECONTEXT_FULL. */
sp_status sp_session_create(const sp_model *m, const sp_session_config *cfg,
                            volatile int *cancel_flag, sp_session **out_session);
void      sp_session_destroy(sp_session *s);

/* Resolved working precision for this session (2-L1.FP16): the precedence result of
 * cfg->precision_override > arch_info.preferred_precision > SP_PRECISION_F32, fixed at
 * create. Backend dispatch reads this to select the fp16 vs f32 vs qf32 kernel path;
 * the math-core reference forward ignores it (it stays f32, the bit-exact anchor).
 * Returns an sp_precision value; SP_PRECISION_UNSPECIFIED on a NULL handle. */
uint32_t  sp_session_precision(const sp_session *s);

/* ── §3 two-function forward -- logits buffers are caller-allocated ──
 * prefill consumes n_tokens, advances position by n_tokens, writes ONLY the last
 * token's logits. decode consumes one token, advances by one, writes that token's
 * logits. logits_capacity < vocab_size returns SP_EBADARG. On a tripped cancel flag
 * the call unwinds to the last completed layer and returns SP_ECANCEL. */
sp_status sp_prefill_chunk(sp_session *s, const int32_t *tokens, size_t n_tokens,
                           float *logits_last, size_t logits_capacity);
sp_status sp_decode_step  (sp_session *s, int32_t token,
                           float *logits, size_t logits_capacity);

/* ── §4 session manipulation -- speculative-decoding-shaped ──
 * clone deep-copies KV/ARM/sieve into an independent session (the spec-decode fork);
 * rewind rolls back n_tokens of accepted state (the reject primitive; ARM writes are
 * journaled so rewind is exact); position reports the current sequence index. */
sp_status sp_session_clone   (const sp_session *s, volatile int *cancel_flag,
                              sp_session **out);
sp_status sp_session_rewind  (sp_session *s, size_t n_tokens);
sp_status sp_session_position(const sp_session *s, size_t *pos_out);

/* ── §5 compute backend registration (Sprint NTT.5b) ──
 *
 * The Bluestein wrapper in math-core (sp_pr_bluestein_*) and any future backend-
 * aware math-core kernel may route its inner NTT calls through a caller-supplied
 * dispatch function rather than the host ntt_crt path. The dispatcher is supplied
 * by L2 (the daemon's FastRPC trampoline; or any other compute backend) and the
 * session stores the (opaque handle, forward fn, inverse fn) triple. Backend-
 * awareness is OPT-IN: existing host code paths keep running unchanged when no
 * backend is registered.
 *
 * Operator + Gemini pre-authorized this L1 ABI extension on 2026-05-30; see
 * tools/sp_compute_skel/docs/PLAN-NTT-5b.md for the architectural review.
 *
 * Stability: handle is OPAQUE — L1 stores the pointer as-is and re-emits it
 * verbatim to the dispatcher; L1 never dereferences it. Lifetime: caller must
 * keep the backing object alive at least until sp_session_destroy (or until
 * unregistration via NULL pointers) returns. */

/* Opaque compute backend handle. L1 never dereferences it. */
typedef struct sp_compute_backend_handle sp_compute_backend_handle;

/* Dispatch function for one prime, one direction.
 *
 *   handle: backend-supplied opaque pointer (re-emitted verbatim from registration)
 *   q_idx : 0 for q_1 = 1073738753, 1 for q_2 = 1073732609
 *   N     : transform length (math-core ntt_init admissible: 128, 256, 512)
 *   in    : N × u32 LE input residues in [0, q)
 *   out   : N × u32 LE output residues in [0, q)
 *
 * Forward and inverse have identical signatures. The math-core wrapper handles
 * CRT recombination across the two per-prime invocations; the dispatcher only
 * services one prime per call.
 *
 * Returns 0 on success, -1 on error. */
typedef int (*sp_compute_ntt_dispatch_fn)(
    void *handle, int q_idx, int N,
    const uint32_t *in, uint32_t *out);

/* Register a compute backend for this session. After registration, math-core
 * kernels that opt into backend dispatch (currently: sp_pr_bluestein_*) route
 * inner NTT calls through forward/inverse instead of the host ntt_crt path.
 *
 * Pass NULL handle + NULL forward + NULL inverse to unregister.
 *
 * Lifetime of `handle`: caller-owned; must remain valid until either:
 *   - sp_session_register_compute_backend is called again to replace/unregister,
 *   - or sp_session_destroy returns.
 *
 * Thread-safety: caller-serialized with all other &mut sp_session operations
 * (i.e. holding the L2 Mutex<SpSession> guard). NOT safe to call concurrently
 * with a forward call on the same session.
 *
 * Returns SP_OK on success; SP_EBADARG on a NULL session pointer. */
sp_status sp_session_register_compute_backend(
    sp_session *s,
    void *handle,
    sp_compute_ntt_dispatch_fn forward,
    sp_compute_ntt_dispatch_fn inverse);

/* Read-back accessors so the math-core wrappers (which take const sp_session *)
 * can pull the registered backend without exposing struct sp_session internals.
 * NULL fields mean "no backend registered" — caller-side fallback to host path. */
void *sp_session_compute_backend_handle (const sp_session *s);
sp_compute_ntt_dispatch_fn sp_session_compute_backend_forward(const sp_session *s);
sp_compute_ntt_dispatch_fn sp_session_compute_backend_inverse(const sp_session *s);

#ifdef __cplusplus
}
#endif
#endif /* SP_L1_H */
