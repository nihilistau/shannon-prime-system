/* xbar_episode.h — XBAR P3.0: the episode / owner-map manifest.
 *
 * CONTRACT-XBAR-P3 §2 (stage P3.0) + §P3.0-manifest-schema (normative) + §3
 * (the geometry law / prefix-sum addressing). RATIFIED (operator, 2026-06-11).
 *
 * An XBAR episode is `{ K store, V store, manifest }` (the C1-lite / NIGHTSHIFT
 * format, extended for the jagged gemma-4 cache). This module is the MANIFEST
 * half only: the data structure + the owner-indirect byte-offset law + a
 * round-trippable serializer. It carries NO model and touches NO decode (the
 * curator_replay.c standalone discipline); the gate G-P3-0 grades it in
 * isolation. P3.1+ wires it into gemma4_decode_cuda.
 *
 * WHY A MANIFEST. The Exec (gemma-4) cache is jagged per layer CLASS:
 *   - 12B  : every layer OWNS its K/V; KVD is CONSTANT 512 (global 1x512 ==
 *            SWA 2x256; gemma4.c:16,79 — audit finding #1). The episode byte
 *            layout therefore degenerates to the C1-lite ((L*P)+pos)*512*4.
 *   - E2B  : kvfs=15 owners / 20 sharers, and the owner rows are GENUINELY
 *            jagged — global owners write 512-wide rows, SWA owners 256-wide
 *            (cuda_forward.cu:1615-1617). Addressing an episode then needs the
 *            prefix-sum-over-owners law + owner indirection for sharer layers.
 *
 * THE NORMATIVE ADDRESSING LAW (CONTRACT-XBAR-P3 §3):
 *   off[L]          = Σ_{L' < L, owns_kv(L')} P · kvd_{L'} · 4    (owners only)
 *   block(L, pos)   = off[own[L]] + pos · kvd_{own[L]} · 4
 * where own[L] = L for owners, and for sharers = kvfs-1 (global) / kvfs-2 (SWA)
 * — the gemma4.c:198 reuse rule (a global sharer reuses the global owner kvfs-1,
 * an SWA sharer the SWA owner kvfs-2; same class ⇒ kvd_{own[L]} == kvd_L). On
 * the 12B (all owners, kvd ≡ 512) this is exactly ((L*P)+pos)*512*4, byte for
 * byte — P3.0's uniform-null.
 *
 * NOTE: projk (the router sidecar) is NOT part of the episode manifest — it is
 * recoverable by re-projecting the stored K (curator_replay.c / C1L.0a), and its
 * heterogeneous per-class element layout is the SEPARATE sp_arm_geom_layout()
 * concern (arm.h:147-168). This manifest addresses the K/V STORE bytes only.
 */
#ifndef SP_XBAR_EPISODE_H
#define SP_XBAR_EPISODE_H

#include <stdint.h>
#include <stddef.h>
#include "sp/arm.h"   /* SP_ARM_PROJ_SEED (macro only — no link dependency) */

#ifdef __cplusplus
extern "C" {
#endif

#define SP_XBAR_EP_MAGIC   "XBAREP01"   /* 8 bytes, not NUL-terminated on disk */
#define SP_XBAR_EP_VERSION 1u
#define SP_XBAR_SHA_LEN    32           /* artifact sha (sha256 raw bytes) */

#define SP_XBAR_CLASS_SWA    0
#define SP_XBAR_CLASS_GLOBAL 1

/* Per-layer geometry the CALLER supplies (read off the loaded gemma4 config).
 * The builder derives owns_kv / own[L] / kvd / off[L] from these + (kvfs,period).
 * `cls` must be consistent with the period rule global = (L % period == period-1)
 * — the builder does not recompute it, so a fixture can drive any geometry. */
typedef struct {
    uint8_t cls;               /* SP_XBAR_CLASS_GLOBAL / _SWA */
    int32_t nh;                /* query heads (capture metadata) */
    int32_t nkv;               /* kv heads in this class */
    int32_t hd;                /* head_dim in this class */
    int32_t window;            /* -1 global / sliding-window width */
    float   rope_base;         /* 1e6 global / 1e4 swa (capture metadata) */
    uint8_t has_freq_factors;  /* proportional rope_freqs present (global only) */
    uint8_t vless;             /* V = raw-K projection, no attn_v (12B globals) */
} sp_xbar_layer_geom;

/* Per-layer record carried IN the manifest (schema = CONTRACT-XBAR-P3 §P3.0). */
typedef struct {
    uint8_t  cls;
    uint8_t  owns_kv;          /* L < kvfs */
    uint8_t  vless;
    uint8_t  has_freq_factors;
    int32_t  nh, nkv, hd, kvd; /* kvd = nkv*hd */
    int32_t  window;
    int32_t  own;              /* own[L] */
    float    rope_base;
    uint64_t off;              /* byte offset of own[L]'s block in the store */
} sp_xbar_layer;

typedef struct {
    uint32_t version;
    uint8_t  artifact_sha[SP_XBAR_SHA_LEN];
    uint64_t proj_seed;        /* SP_ARM_PROJ_SEED at build time */
    int32_t  NL, P, period, kvfs, r;
    sp_xbar_layer *layers;     /* [NL], owned by the manifest (malloc'd) */
    uint64_t store_bytes;      /* total per-stream store size = Σ_owners P*kvd*4 */
} sp_xbar_manifest;

/* Build a manifest from per-layer geometry. Allocates mf->layers[NL]. Computes
 * owns_kv, own[L] (gemma4.c:198 reuse rule), kvd, off[L] (prefix-sum over
 * owners), and store_bytes. Returns 0 on success, non-zero on bad args
 * (kvfs<1 or a sharer whose computed owner is not a valid in-range owner — the
 * gemma4.c:202 bounds guard, surfaced here instead of an OOB). */
int sp_xbar_manifest_build(sp_xbar_manifest *mf,
                           int NL, int P, int period, int kvfs, int r,
                           uint64_t proj_seed,
                           const uint8_t artifact_sha[SP_XBAR_SHA_LEN],
                           const sp_xbar_layer_geom *geom);

/* Byte offset of layer L's K/V block for position pos:
 *     off[own[L]] + pos · kvd_{own[L]} · 4   (== layers[L].off + pos*kvd*4,
 * since a sharer shares its owner's class ⇒ identical kvd). */
uint64_t sp_xbar_block_off(const sp_xbar_manifest *mf, int L, int pos);

/* Serialized size in bytes for a fully-built manifest. */
size_t sp_xbar_manifest_serial_size(const sp_xbar_manifest *mf);

/* Serialize to buf (must hold >= serial_size). Fixed-width little-endian, no
 * struct padding written. Returns bytes written, or 0 if cap is too small. */
size_t sp_xbar_manifest_serialize(const sp_xbar_manifest *mf,
                                  uint8_t *buf, size_t cap);

/* Deserialize from buf[0,len). Allocates mf->layers. Returns 0 on success,
 * non-zero on magic/version mismatch or truncation. */
int sp_xbar_manifest_deserialize(sp_xbar_manifest *mf,
                                 const uint8_t *buf, size_t len);

/* Free mf->layers (idempotent; zeroes the pointer). */
void sp_xbar_manifest_free(sp_xbar_manifest *mf);

#ifdef __cplusplus
}
#endif
#endif /* SP_XBAR_EPISODE_H */
