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

    /* ── appended Phase 3-G4 (Gemma4; reserved arch_struct tail; 0 = unspecified) ──
     * For arch_id == GEMMA4 the base head_dim/n_heads/n_kv_heads/rope_freq_base/
     * swa_window hold the GLOBAL-layer geometry; these carry the SWA-layer geometry
     * and the per-layer-input (AltUp) + shared-KV + softcap shape. All zero on
     * non-Gemma4 models. sizeof(sp_arch_info) MUST stay <= 256. */
    uint32_t g4_hd_swa;           /* SWA head_dim (256)   */
    uint32_t g4_nh_swa;           /* SWA n_head (8)       */
    uint32_t g4_nkv_swa;          /* SWA n_head_kv (2)    */
    float    g4_rope_base_swa;    /* SWA RoPE base (1e4)  */
    uint32_t g4_n_embd_per_layer; /* per-layer-input width (256); 0 = no AltUp path */
    uint32_t g4_n_kv_from_start;  /* layers [0,this) own KV; rest reuse (shared-KV)  */
    float    g4_logit_softcap;    /* final-logit softcap (30); 0 = none              */
    uint32_t g4_swa_period;       /* SWA/global period (6); global when L%period==period-1 */

    /* ── appended Phase 3-MoE+GDN (qwen35moe / Qwen3.6; reserved tail; 0 = unspecified) ──
     * For arch_id == QWEN36 the base head_dim/n_heads/n_kv_heads hold the FULL-ATTN
     * geometry; these carry the GDN (gated delta-net) geometry + MoE params + IMRoPE
     * sections. All zero on non-qwen35moe models. sizeof(sp_arch_info) MUST stay <= 256. */
    uint32_t q36_full_attn_interval; /* full-attn iff (L+1)%this==0 (4)  */
    uint32_t q36_n_expert;           /* routed experts (256)             */
    uint32_t q36_n_expert_used;      /* top-k (8)                        */
    uint32_t q36_n_ff_exp;           /* per-expert FFN dim (512)         */
    uint32_t q36_n_ff_shexp;         /* shared-expert FFN dim (512)      */
    float    q36_expert_weights_scale;/* scale on renormed top-k weights (1.0) */
    uint32_t q36_gdn_conv_k;         /* causal conv kernel (4)           */
    uint32_t q36_gdn_state;          /* GDN head_dim S (128)             */
    uint32_t q36_gdn_n_k_heads;      /* k/q heads H_k (16)               */
    uint32_t q36_gdn_n_v_heads;      /* v heads H_v / dt_rank (32)       */
    uint32_t q36_gdn_inner;          /* d_inner (4096)                   */
    int32_t  q36_rope_sections[4];   /* IMRoPE sections [11,11,10,0]     */
    uint32_t q36_rope_dim;           /* rope.dimension_count (64)        */
    float    q36_rope_base;          /* rope.freq_base (1e7)             */
    uint32_t q36_nextn_predict_layers;/* trailing NextN/MTP blocks (0)   */

    /* ── appended Phase 5-DG (diffusion-gemma; reserved tail; 0 = unspecified) ──
     * For arch_id == DIFFUSION_GEMMA the backbone geometry is carried in the base +
     * g4_* + q36_n_expert(_used)/q36_n_ff_exp fields (the gemma4 MoE backbone); these
     * carry the diffusion-specific surface. canvas_length is REQUIRED (the forward
     * splits [prompt|canvas] on it). The eb_* entropy-bound sampler params are 0 when
     * absent from the GGUF (this conversion wave omits them; sampler falls back to
     * defaults at N4). All zero on non-diffusion models. sizeof(sp_arch_info) <= 256. */
    uint32_t dg_canvas_length;        /* diffusion.canvas_length (256); REQUIRED      */
    uint32_t dg_eb_max_steps;         /* diffusion.eb_max_steps; 0 = unspecified      */
    float    dg_eb_t_min;             /* diffusion.eb_t_min                            */
    float    dg_eb_t_max;             /* diffusion.eb_t_max                            */
    float    dg_eb_entropy_bound;     /* diffusion.eb_entropy_bound (MI accept cutoff) */
    float    dg_eb_stability_threshold;/* diffusion.eb_stability_threshold            */
    float    dg_eb_confidence_threshold;/* diffusion.eb_confidence_threshold          */
    uint32_t dg_shift_logits;         /* diffusion.shift_logits (canvas models: 0)    */
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

/* ── §6 forward backend registration (Sprint WIRE-HEX) ──
 *
 * Distinct from the §5 NTT compute-backend hook (which only services inner NTT
 * calls inside the Bluestein wrapper): this hook replaces the WHOLE prefill
 * forward call with a caller-supplied dispatcher. Targets the engine's
 * per-backend full-forward entry points (gemma3_forward_hexagon,
 * gemma3_forward_cuda, gemma3_forward_vulkan) which run the entire transformer
 * forward on accelerator silicon and return the full [n_tok * n_vocab] logits.
 *
 * Background — the 6-month gap this hook closes: the engine's ppl.c routes
 * SP_BACKEND={cuda,vulkan,hexagon} to per-backend full-forward functions
 * (src/forward/ppl.c:27-47), but sp_daemon's prefill/decode go through
 * sp_session which always runs math-core's REFERENCE forward
 * (lib/shannon-prime-system/core/forward/forward.c, identified as "pure-f32
 * scalar" in core/forward_dispatch/forward_dispatch.c:1-15). Registration
 * lets L2 plug the production forward in without engine code duplication.
 *
 * Scope: prefill_chunk only. decode_step (persistent KV) is NOT hooked in
 * this sprint — backend forwards like gemma3_forward_hexagon re-run the full
 * forward over the accumulated history per call (the engine's ppl-style
 * usage). Hooking decode would require either re-running the full history
 * per token (devastating to perf) or a per-backend persistent-KV variant
 * (different sprint). The fallback for decode is the math-core reference
 * path, identical to today's behavior.
 *
 * Stability: handle is OPAQUE; L1 stores and re-emits it verbatim. qm_opaque
 * is the borrowed sp_session-owned qwen3_model pointer (see
 * sp_session_qwen3_model below) — L1 emits its internal `qm` to the
 * dispatcher; dispatcher casts to `const qwen3_model *` and calls into the
 * engine's backend (which is what `gemma3_forward_hexagon` already takes).
 * Lifetime: caller must keep handle alive at least until sp_session_destroy
 * (or until unregistration via NULL fn) returns. */

/* Full-forward dispatcher signature.
 *
 *   handle    : backend-supplied opaque pointer (re-emitted verbatim)
 *   qm_opaque : session's borrowed qwen3_model pointer (treat as const qwen3_model *
 *               on the engine side; L1 stores it as void *).
 *   tokens    : n_tok int32 token IDs (the full accumulated history)
 *   n_tok     : number of tokens
 *   logits    : caller-allocated [n_tok * n_vocab] f32 output
 *
 * Returns 0 on success, non-zero on error (sp_prefill_chunk maps non-zero
 * to SP_EBADSTATE). The dispatcher MUST write all n_tok positions' logits
 * (sp_prefill_chunk extracts the LAST position for the caller). */
typedef int (*sp_forward_dispatch_fn)(
    void *handle, const void *qm_opaque,
    const int32_t *tokens, int n_tok, float *logits);

/* Register a forward backend for this session. After registration,
 * sp_prefill_chunk dispatches the full forward through `fn` instead of the
 * math-core reference. sp_decode_step is unaffected (persistent KV path).
 *
 * Pass NULL handle + NULL fn to unregister.
 *
 * Lifetime of `handle`: caller-owned; must remain valid until either:
 *   - sp_session_register_forward_backend is called again to replace/unregister,
 *   - or sp_session_destroy returns.
 *
 * Thread-safety: caller-serialized with all &mut sp_session operations
 * (i.e. holding the L2 Mutex<SpSession> guard). NOT safe to call
 * concurrently with a forward call on the same session.
 *
 * Returns SP_OK on success; SP_EBADARG on a NULL session pointer. */
sp_status sp_session_register_forward_backend(
    sp_session *s,
    void *handle,
    sp_forward_dispatch_fn fn);

/* Read-back accessors. NULL fn means "no forward backend registered" —
 * caller-side fallback to math-core reference path. */
void *sp_session_forward_backend_handle(const sp_session *s);
sp_forward_dispatch_fn sp_session_forward_backend_fn(const sp_session *s);

/* Borrow accessor: the session's internal qwen3_model pointer. Used by L2
 * trampolines that need to pass the model handle to a forward dispatcher
 * (e.g. the engine's gemma3_forward_hexagon takes const qwen3_model *).
 * NULL on a NULL session; never NULL on a successfully-created session.
 * The pointer is borrowed from the model handle and remains valid for the
 * session's lifetime; caller MUST NOT free it. */
const void *sp_session_qwen3_model(const sp_session *s);

/* ── §6b persistent-KV decode backend registration (Sprint WIRE-CUDA-DECODE) ──
 *
 * Distinct from the §6 forward-backend hook, which is PREFILL-ONLY: it re-runs
 * the WHOLE forward over the accumulated history per call (the engine's ppl-style
 * usage). Token-by-token DECODE through that hook is two failures — O(N²) cost,
 * and for a packed (OK_Q4B) model the tied full-vocab head is materialized only
 * inside the engine's DECODE path, so driving decode through the prefill entry
 * trips its head guard. This §6b verb binds a STATEFUL, session-resident KV
 * handle + a step-wise dispatch table (a vtable of lifecycle ops over one
 * device-resident cache), mirroring the engine's frozen gemma4_kv_* C ABI.
 *
 * When a kvdecode backend is registered, sp_decode_step routes the single-token
 * forward through dt->decode_step on the session-resident handle instead of the
 * math-core reference KV path (else current behaviour — the field is calloc-zero
 * at create, so an unregistered session is byte-compatible with every existing
 * consumer). The backend owns the device-resident cache across calls — that
 * persistence is the whole point (the recompute tax the crossbar deletes).
 *
 * Lifetime: the handle + dt table are caller-owned; L1 stores the pointers as-is
 * and re-emits the handle verbatim to the dispatch fns (never dereferenced by
 * L1). They MUST stay valid until either re-registration (NULL dt unregisters)
 * or sp_session_destroy returns. Thread-safety: caller-serialized with all &mut
 * sp_session ops (the L2 Mutex<SpSession> guard); NOT safe concurrent with a
 * decode on the session. */

/* Opaque per-session KV-decode handle. On the CUDA backend this wraps an
 * engine sp_g4_kv* (a resident KV cache). Owned by the backend; L1 never
 * dereferences it — it only re-emits the pointer to the dispatch fns. */
typedef struct sp_kvdecode_handle sp_kvdecode_handle;

/* Step-wise persistent-KV decode dispatch table. Each fn returns 0 on success,
 * non-zero on error (sp_last_error carries detail). The backend owns the
 * device-resident cache across calls.
 *
 *   open    (qm_opaque, pmax, &handle)  : alloc resident KV (dpos=0). qm_opaque
 *                                         is the session's borrowed qwen3_model*.
 *   prefill (handle, tokens, n_tok)     : ingest history, store K/V, dpos+=n.
 *   decode_step (handle, token, logits) : forward ONE token at the live dpos,
 *                                         write full-vocab logits [vocab] for the
 *                                         NEXT position, advance dpos.
 *   rewind  (handle, n)                 : O(1) cold-evict dpos -= n.
 *   position(handle)                    : current dpos (>=0), or -1 on NULL.
 *   close   (handle)                    : free the resident cache (NULL-safe).
 *
 * sp_decode_step uses only decode_step (the others are the lifecycle the L2
 * registrant drives directly: open at register, prefill/rewind/position as
 * needed, close at unregister/destroy). */
typedef struct sp_kvdecode_dispatch_fn {
    int  (*open)       (const void *qm_opaque, int pmax, sp_kvdecode_handle **out);
    int  (*prefill)    (sp_kvdecode_handle *h, const int32_t *tokens, int n_tok);
    int  (*decode_step)(sp_kvdecode_handle *h, int32_t token, float *logits);
    int  (*rewind)     (sp_kvdecode_handle *h, int n);
    int  (*position)   (const sp_kvdecode_handle *h);
    void (*close)      (sp_kvdecode_handle *h);
} sp_kvdecode_dispatch_fn;

/* Register a persistent-KV decode backend for this session. After registration,
 * sp_decode_step routes through dt->decode_step on `handle` instead of the
 * math-core reference decode. `handle` is the backend-opaque KV handle (created
 * by the caller via dt->open before this call). Pass NULL dt to unregister
 * (the session reverts to the reference KV path; L1 does NOT call dt->close —
 * the caller owns the handle's lifetime).
 *
 * Returns SP_OK; SP_EBADARG on a NULL session. */
sp_status sp_session_register_kvdecode_backend(
    sp_session *s,
    sp_kvdecode_handle *handle,
    const sp_kvdecode_dispatch_fn *dt);

/* Read-back accessors (NULL dt => no kvdecode backend; fall back to reference). */
sp_kvdecode_handle *sp_session_kvdecode_backend_handle(const sp_session *s);
const sp_kvdecode_dispatch_fn *sp_session_kvdecode_backend_dt(const sp_session *s);

/* ── §6c per-session byte-exact ("auditable mode") knob on the kvdecode backend ──
 * CONTRACT-CHAT-FULLSTACK B1 (lattice papers/CONTRACT-CHAT-FULLSTACK.md §3).
 *
 * This is NOT a new frozen L1 verb on `sp_session` — it is a BACKEND-INTERNAL
 * runtime knob on the resident KV-decode handle (the `sp_g4_kv*` behind the §6b
 * dispatch table). It is registered HERE (append-only, per the contract's ABI
 * rule: any new session-level knob registers in this header FIRST) but lives as
 * an ENGINE backend symbol, not a math-core function, because it toggles a
 * device-side flag specific to the CUDA gemma4 decode (`d_bx_flag` +
 * k_attn_decode_win_bx) and the math-core has no such state.
 *
 * Engine symbol (sp_engine/cuda_backend.h):
 *     int gemma4_kv_byteexact_set(sp_g4_kv *s, int on);
 * Daemon glue (sp_daemon_cuda_glue.c):
 *     int sp_daemon_cuda_kvdecode_byteexact(void *handle, int on);
 *
 * Semantics: on!=0 routes the resident decode through the exact-integer islands
 * (RMSNorm/RoPE/GELU — d_bx_flag) + the dual-prime CRT-NTT exact-integer
 * attention (k_attn_decode_win_bx) => run-to-run bit-identical (the AUDITABILITY
 * / cross-machine-determinism axis). on==0 restores the float decode path =
 * byte-identical null floor. Per-request callable under the resident-cache Mutex
 * (the chat path sets on=1 at request start, on=0 at end). When a future fully
 * generic kvdecode backend needs this, it is promoted to a dispatch-table row
 * here (§6b struct grows append-only); for now the single CUDA backend owns it.
 *
 * NULL floor: a handle on which it is never called (default sp_g4_kv.bx_on=0) is
 * byte-identical to the Stage-A float decode — no frozen surface renumbered. */

/* ── §6d XBAR ring + episode-replay knobs on the kvdecode backend ──
 * CONTRACT-CHAT-FULLSTACK B2 (lattice papers/CONTRACT-CHAT-FULLSTACK.md §3-B2).
 *
 * Like §6c these are NOT new frozen `sp_session` verbs — they are BACKEND-INTERNAL
 * runtime knobs on the resident KV-decode handle (the `sp_g4_kv*` behind the §6b
 * dispatch table), registered HERE (append-only, per the contract's ABI rule) but
 * living as ENGINE backend symbols because they touch device-side CUDA-gemma4
 * decode state the math-core does not model.
 *
 * (a) SWA W-slot RING (the O(1)-context KV win). The resident cache's SWA-owner
 *     layers (the period-6 non-global layers) can be allocated as a `Wring=min(W,P)`
 *     ring instead of a full Pmax cache; the decode writes slot=pos%Wring and
 *     attends in POSITION order via k_attn_decode_ring (fp reduction byte-identical
 *     to the full-window decode ⇒ a null-floor parity, KAI-1c). This is selected at
 *     gemma4_kv_open time (the ring + its undo-journal are allocation-shaped), so it
 *     is a DAEMON-STARTUP knob, not per-request: the daemon sets it before opening
 *     the resident cache. Globals stay full-cache on the resident path (the compact
 *     global slab + learned-LSH sparse global recall — SP_ARM_SLAB / SP_ARM_LSH,
 *     CONTRACT-XBAR-P3 Phase C — is wired only on the one-shot gemma4_decode_cuda,
 *     NOT yet on the resident gemma4_kv_* decode; porting it is a tracked follow-on).
 *     Engine: gemma4_kv_open reads SP_G4_KV_RING_W / SP_G4_KV_JMAX (cuda_forward.cu);
 *     daemon: SP_DAEMON_KVDECODE_RING_W / SP_DAEMON_KVDECODE_JMAX set that env at
 *     startup. NULL floor: ring_W=0 (unset) = the full-cache Stage-A/B1 decode.
 *
 * (b) Episode REPLAY into a live turn (SP_REPLAY recall, C2 #222). Recall a stored
 *     episode's owner K/V directly into the resident cache at [dpos,dpos+npos) and
 *     advance dpos — so a prior memory rolls into the current chat turn; reject is
 *     the O(1) gemma4_kv_rewind(npos) inverse (ring-aware journal). PER-REQUEST: the
 *     chat path calls it under the cache Mutex before decode when the request names
 *     an episode dir. Engine symbol:
 *         int gemma4_kv_replay(sp_g4_kv *s, const char *epdir, int npos, int zero);
 *     Daemon glue (sp_daemon_cuda_glue.c):
 *         int sp_daemon_cuda_kvdecode_replay(void *handle, const char *epdir,
 *                                            int npos, int zero);
 *     NULL floor: a request that names no episode never calls it (byte-identical to
 *     the B1/Stage-A path). When a future generic kvdecode backend needs replay it is
 *     promoted to a dispatch-table row here (§6b struct grows append-only).
 *
 * Both default-off ⇒ a daemon with neither set is byte-identical to the B1 decode —
 * no frozen surface renumbered. */

/* ── §6e single latent entry seam on the kvdecode backend ──
 * CONTRACT-CHAT-FULLSTACK B5 (lattice papers/CONTRACT-CHAT-FULLSTACK.md §6).
 *
 * The operator's single-entry-point architecture: the model has ONE true input — a
 * stream of continuous latent vectors in its residual space — and EVERY modality
 * enters through the one residual seam (gemma4_kv_inject* → the model mints K/V
 * natively, RoPE + per-layer variance correct by construction). Three SOURCES feed
 * that one entry: TEXT (BPE ids → embd[id]*sqrt(E) → seam), AUDIO (EAR/KAI-3
 * projector frames → seam), and MEMORY (decoded episode residuals → seam; episodic
 * K/V recall stays the §6d-b replay seam).
 *
 * Like §6c/§6d these are NOT new frozen `sp_session` verbs — they are BACKEND-INTERNAL
 * runtime knobs on the resident KV-decode handle (the `sp_g4_kv*` behind the §6b
 * dispatch table), registered HERE (append-only, per the contract's ABI rule) but
 * living as ENGINE backend symbols because they touch device-side CUDA-gemma4 state
 * the math-core does not model.
 *
 * (a) TEXT via the seam. Engine symbol (sp_engine/cuda_backend.h):
 *         int gemma4_kv_inject_tokens(sp_g4_kv *s, const int32_t *toks, int n);
 *     Daemon glue (sp_daemon_cuda_glue.c):
 *         int sp_daemon_cuda_kvdecode_inject_tokens(void *handle, const int32_t *toks, int n);
 *     Per token id, stages embd[id]*sqrt(E) into the inject buffer (the SAME arithmetic
 *     the stock embed-at step runs) and steps the real id ⇒ the residual entering layer 0
 *     is BIT-IDENTICAL to gemma4_kv_prefill(&id,1). PARITY by construction: text-via-seam
 *     == text-via-prefill. PER-REQUEST: the chat path calls it under the cache Mutex when
 *     `single_entry` is set, in place of prefill for the prompt head.
 *
 * (b) GENERIC residual-frame channel (audio/memory). Engine symbol (already in the engine):
 *         int gemma4_kv_inject_seq(sp_g4_kv *s, const float *embs, int n_frames, int ph_token);
 *     Daemon glue (sp_daemon_cuda_glue.c):
 *         int sp_daemon_cuda_kvdecode_inject_frames(void *handle, const float *embs,
 *                                                   int n_frames, int ph_token);
 *     Injects n_frames raw E-float residual vectors at consecutive positions, each minted
 *     at ph_token — the seam AUDIO (EAR/KAI-3) and MEMORY (decoded episode residuals) feed.
 *
 * NULL floor: a request that sets neither single_entry nor inject_frames never calls
 * either ⇒ byte-identical to the §6d/B1 prefill decode. No frozen surface renumbered.
 * When a future fully generic kvdecode backend needs these, they are promoted to
 * dispatch-table rows here (§6b struct grows append-only); for now the CUDA backend owns
 * them. */

#ifdef __cplusplus
}
#endif
#endif /* SP_L1_H */
