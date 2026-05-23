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

/* ── §2 arch info -- caller-stack-allocated, L1 fills it in place ──
 * A public projection of the math core's model config. L2 reads this once at load to
 * size the logits buffer (vocab_size * sizeof(float)) and never re-queries. Rust
 * bindings should treat the out-parameter as &mut MaybeUninit<sp_arch_info>. */
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
} sp_arch_info;

/* Populate *out from the loaded model's header arch_struct. Caller-stack-allocated.
 * SP_OK on success; SP_EBADARG on a NULL handle/out. */
sp_status sp_model_arch(const sp_model *m, sp_arch_info *out);

/* ── §6 session config -- immutable for the session's lifetime ── */
typedef struct {
    size_t   max_context;     /* hard cap on sequence position; 0 = arch default       */
    int      deterministic;   /* nonzero: serial reductions, single stream (bit-exact) */
    uint32_t arm_bank_kb;     /* ARM HRR bank size; 0 = arch default (Phase 9+)         */
    uint32_t sieve_capacity;  /* Friedman-sieve frontier cap; 0 = arch default (Phase 5+) */
    uint32_t flags;           /* reserved bitfield (e.g. SP_VERIFY_TENSORS)            */
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

#ifdef __cplusplus
}
#endif
#endif /* SP_L1_H */
