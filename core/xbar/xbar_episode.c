/* xbar_episode.c — XBAR P3.0 episode/owner-map manifest (CONTRACT-XBAR-P3 §2/§3).
 *
 * Header + serializer only — no model, no decode (the curator_replay.c
 * standalone discipline). The owner-indirect prefix-sum addressing law lives in
 * sp_xbar_manifest_build / sp_xbar_block_off; the serializer is fixed-width
 * little-endian so a round-trip is byte-exact across the LE build hosts.
 *
 * Standalone build + gate (WSL/MinGW, links nothing but libc):
 *   gcc -O2 -std=c11 -Iinclude tools/curator/xbar_manifest_gate.c \
 *       core/xbar/xbar_episode.c -o /tmp/xbar_gate && /tmp/xbar_gate
 */
#include "sp/xbar_episode.h"
#include <stdlib.h>
#include <string.h>

/* ── fixed-width little-endian put/get (no struct fwrite; padding-free) ──────── */

static void put_u8 (uint8_t **p, uint8_t v)  { *(*p)++ = v; }
static void put_u32(uint8_t **p, uint32_t v) {
    uint8_t *q = *p;
    q[0]=(uint8_t)v; q[1]=(uint8_t)(v>>8); q[2]=(uint8_t)(v>>16); q[3]=(uint8_t)(v>>24);
    *p += 4;
}
static void put_i32(uint8_t **p, int32_t v) { put_u32(p, (uint32_t)v); }
static void put_u64(uint8_t **p, uint64_t v) {
    uint8_t *q = *p;
    for (int i = 0; i < 8; i++) q[i] = (uint8_t)(v >> (8*i));
    *p += 8;
}
static void put_f32(uint8_t **p, float f) {
    uint32_t u; memcpy(&u, &f, 4); put_u32(p, u);
}

static uint8_t  get_u8 (const uint8_t **p) { return *(*p)++; }
static uint32_t get_u32(const uint8_t **p) {
    const uint8_t *q = *p; *p += 4;
    return (uint32_t)q[0] | ((uint32_t)q[1]<<8) | ((uint32_t)q[2]<<16) | ((uint32_t)q[3]<<24);
}
static int32_t  get_i32(const uint8_t **p) { return (int32_t)get_u32(p); }
static uint64_t get_u64(const uint8_t **p) {
    const uint8_t *q = *p; *p += 8; uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)q[i] << (8*i);
    return v;
}
static float get_f32(const uint8_t **p) {
    uint32_t u = get_u32(p); float f; memcpy(&f, &u, 4); return f;
}

/* on-disk sizes: header (excl. NL records), then one record */
#define XB_HDR_BYTES (8 /*magic*/ + 4 /*version*/ + SP_XBAR_SHA_LEN + 8 /*seed*/ \
                      + 5*4 /*NL,P,period,kvfs,r*/ + 8 /*store_bytes*/)
#define XB_REC_BYTES (4 /*cls,owns,vless,hff*/ + 5*4 /*nh,nkv,hd,kvd,window*/ \
                      + 4 /*own*/ + 4 /*rope_base*/ + 8 /*off*/)

/* ── build: the owner-indirect prefix-sum law ──────────────────────────────── */

int sp_xbar_manifest_build(sp_xbar_manifest *mf,
                           int NL, int P, int period, int kvfs, int r,
                           uint64_t proj_seed,
                           const uint8_t artifact_sha[SP_XBAR_SHA_LEN],
                           const sp_xbar_layer_geom *geom)
{
    if (!mf || !geom || NL <= 0 || P <= 0 || kvfs < 1 || kvfs > NL) return 1;

    sp_xbar_layer *Ls = (sp_xbar_layer *)calloc((size_t)NL, sizeof(sp_xbar_layer));
    if (!Ls) return 2;

    /* Pass 1: owners. off[L] = running prefix-sum of P*kvd*4 over owners only. */
    uint64_t cursor = 0;
    for (int L = 0; L < NL; L++) {
        const sp_xbar_layer_geom *g = &geom[L];
        sp_xbar_layer *o = &Ls[L];
        o->cls = g->cls;
        o->nh = g->nh; o->nkv = g->nkv; o->hd = g->hd;
        o->kvd = g->nkv * g->hd;
        o->window = g->window;
        o->rope_base = g->rope_base;
        o->has_freq_factors = g->has_freq_factors;
        o->vless = g->vless;
        o->owns_kv = (uint8_t)(L < kvfs);
        if (o->owns_kv) {
            o->own = L;
            o->off = cursor;
            cursor += (uint64_t)P * (uint64_t)o->kvd * 4u;
        }
    }

    /* Pass 2: sharers. own[L] = kvfs-1 (global) / kvfs-2 (SWA); off = owner's off.
     * The owner must be a real in-range owner of the SAME class (gemma4.c:198,202
     * bounds guard) — otherwise the arch struct is malformed; fail loudly. */
    for (int L = 0; L < NL; L++) {
        sp_xbar_layer *o = &Ls[L];
        if (o->owns_kv) continue;
        int src = kvfs - (o->cls == SP_XBAR_CLASS_GLOBAL ? 1 : 2);
        if (src < 0 || src >= kvfs || !Ls[src].owns_kv || Ls[src].cls != o->cls) {
            free(Ls); return 3;
        }
        o->own = src;
        o->off = Ls[src].off;
        /* kvd of a sharer equals its owner's kvd by construction (same class);
         * carry the owner's so block_off arithmetic is self-consistent. */
        o->kvd = Ls[src].kvd;
    }

    mf->version = SP_XBAR_EP_VERSION;
    if (artifact_sha) memcpy(mf->artifact_sha, artifact_sha, SP_XBAR_SHA_LEN);
    else memset(mf->artifact_sha, 0, SP_XBAR_SHA_LEN);
    mf->proj_seed = proj_seed;
    mf->NL = NL; mf->P = P; mf->period = period; mf->kvfs = kvfs; mf->r = r;
    mf->layers = Ls;
    mf->store_bytes = cursor;
    return 0;
}

uint64_t sp_xbar_block_off(const sp_xbar_manifest *mf, int L, int pos)
{
    const sp_xbar_layer *o = &mf->layers[L];
    return o->off + (uint64_t)pos * (uint64_t)o->kvd * 4u;
}

/* ── serializer ────────────────────────────────────────────────────────────── */

size_t sp_xbar_manifest_serial_size(const sp_xbar_manifest *mf)
{
    return (size_t)XB_HDR_BYTES + (size_t)mf->NL * XB_REC_BYTES;
}

size_t sp_xbar_manifest_serialize(const sp_xbar_manifest *mf, uint8_t *buf, size_t cap)
{
    size_t need = sp_xbar_manifest_serial_size(mf);
    if (!buf || cap < need) return 0;
    uint8_t *p = buf;
    memcpy(p, SP_XBAR_EP_MAGIC, 8); p += 8;
    put_u32(&p, mf->version);
    memcpy(p, mf->artifact_sha, SP_XBAR_SHA_LEN); p += SP_XBAR_SHA_LEN;
    put_u64(&p, mf->proj_seed);
    put_i32(&p, mf->NL); put_i32(&p, mf->P); put_i32(&p, mf->period);
    put_i32(&p, mf->kvfs); put_i32(&p, mf->r);
    put_u64(&p, mf->store_bytes);
    for (int L = 0; L < mf->NL; L++) {
        const sp_xbar_layer *o = &mf->layers[L];
        put_u8(&p, o->cls); put_u8(&p, o->owns_kv);
        put_u8(&p, o->vless); put_u8(&p, o->has_freq_factors);
        put_i32(&p, o->nh); put_i32(&p, o->nkv); put_i32(&p, o->hd); put_i32(&p, o->kvd);
        put_i32(&p, o->window); put_i32(&p, o->own);
        put_f32(&p, o->rope_base);
        put_u64(&p, o->off);
    }
    return (size_t)(p - buf);
}

int sp_xbar_manifest_deserialize(sp_xbar_manifest *mf, const uint8_t *buf, size_t len)
{
    if (!mf || !buf || len < (size_t)XB_HDR_BYTES) return 1;
    const uint8_t *p = buf;
    if (memcmp(p, SP_XBAR_EP_MAGIC, 8) != 0) return 2;
    p += 8;
    uint32_t ver = get_u32(&p);
    if (ver != SP_XBAR_EP_VERSION) return 3;
    mf->version = ver;
    memcpy(mf->artifact_sha, p, SP_XBAR_SHA_LEN); p += SP_XBAR_SHA_LEN;
    mf->proj_seed = get_u64(&p);
    mf->NL = get_i32(&p); mf->P = get_i32(&p); mf->period = get_i32(&p);
    mf->kvfs = get_i32(&p); mf->r = get_i32(&p);
    mf->store_bytes = get_u64(&p);
    if (mf->NL <= 0) return 4;
    if (len < (size_t)XB_HDR_BYTES + (size_t)mf->NL * XB_REC_BYTES) return 5;
    sp_xbar_layer *Ls = (sp_xbar_layer *)calloc((size_t)mf->NL, sizeof(sp_xbar_layer));
    if (!Ls) return 6;
    for (int L = 0; L < mf->NL; L++) {
        sp_xbar_layer *o = &Ls[L];
        o->cls = get_u8(&p); o->owns_kv = get_u8(&p);
        o->vless = get_u8(&p); o->has_freq_factors = get_u8(&p);
        o->nh = get_i32(&p); o->nkv = get_i32(&p); o->hd = get_i32(&p); o->kvd = get_i32(&p);
        o->window = get_i32(&p); o->own = get_i32(&p);
        o->rope_base = get_f32(&p);
        o->off = get_u64(&p);
    }
    mf->layers = Ls;
    return 0;
}

void sp_xbar_manifest_free(sp_xbar_manifest *mf)
{
    if (mf && mf->layers) { free(mf->layers); mf->layers = NULL; }
}
