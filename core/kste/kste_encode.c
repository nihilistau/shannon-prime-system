/* kste_encode.c — Knight-Spinor Tree Encoder (Phase 1F).
 *
 * Maps a K-vector of int32 components into the frozen 64-byte packed tree of
 * include/sp/kste.h. See that header for the full layout and the rationale for
 * the fixed T_{60,3} shape, int32 input, and componentwise dominance order.
 *
 * Determinism notes (these are the load-bearing choices for byte-identity):
 *   - all arithmetic is integer; no float anywhere.
 *   - every multi-byte field is emitted little-endian via explicit shifts; we
 *     never memcpy a wider type into the byte image.
 *   - the per-node label is a 6-tuple of quantized order statistics sampled at
 *     fixed fractional positions of the node's (sorted) slice. Sampling
 *     positions are computed with integer division only.
 *   - the three first-level children are sorted into a canonical order
 *     (lexicographic on their int16 label tuple) before serialization so that
 *     "same first-level child multiset" is a byte-detectable property.
 */
#include "sp/kste.h"

#include <stdint.h>

/* ---- quantization -------------------------------------------------------- */

/* Clamp a 32-bit value into int16 range. Order-preserving on the clamped
 * domain, which is what keeps the componentwise dominance order meaningful.
 * Written with explicit casts so -Wconversion/-Wsign-conversion stay quiet. */
static int16_t quantize(int32_t v) {
    if (v > 32767)  return (int16_t)32767;
    if (v < -32768) return (int16_t)(-32768);
    return (int16_t)v;
}

/* ---- a node label: 6 quantized order statistics -------------------------- */

typedef struct {
    int16_t s[SP_KSTE_LABEL_DIM];   /* min, p20, p40, p60, p80, max */
} kste_label_t;

/* Insertion sort of an int32 slice copy (n is tiny: <= 24 here, and per-node
 * slices are smaller). Deterministic, in place. */
static void sort_i32(int32_t *a, int n) {
    for (int i = 1; i < n; i++) {
        int32_t key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}

/* Compute the 6-statistic label for vec[lo..hi) (hi exclusive).
 *
 * If the slice is empty, the label is all zeros (a deterministic neutral
 * element). Otherwise we copy the slice, sort it, and sample at fixed
 * fractional positions p in {0, 1/5, 2/5, 3/5, 4/5, 1} via integer index
 * idx = (p_num * (m-1)) / 5, where m is the slice length. The endpoints land
 * exactly on min (p=0) and max (p=1); the interior four are independent sample
 * points of the sorted slice, so the components move independently under the
 * componentwise order. */
static kste_label_t label_of(const int32_t *vec, int lo, int hi) {
    kste_label_t lab;
    int m = hi - lo;

    if (m <= 0) {
        for (int i = 0; i < SP_KSTE_LABEL_DIM; i++) lab.s[i] = 0;
        return lab;
    }

    int32_t buf[24];
    /* slices never exceed the whole vector (k <= 24 in tests, and the encoder
     * is robust: if a slice somehow exceeds buf we truncate deterministically,
     * which never happens for the documented input sizes). */
    int n = m;
    if (n > 24) n = 24;
    for (int i = 0; i < n; i++) buf[i] = vec[lo + i];
    sort_i32(buf, n);

    /* fractional sample numerators over denominator 5: 0,1,2,3,4,5 */
    static const int num[SP_KSTE_LABEL_DIM] = { 0, 1, 2, 3, 4, 5 };
    for (int i = 0; i < SP_KSTE_LABEL_DIM; i++) {
        int idx = (num[i] * (n - 1)) / 5;   /* in [0, n-1] */
        lab.s[i] = quantize(buf[idx]);
    }
    return lab;
}

/* ---- little-endian serialization ----------------------------------------- */

static void put_u32_le(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
    p[2] = (uint8_t)((v >> 16) & 0xFFu);
    p[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static void put_i16_le(uint8_t *p, int16_t v) {
    uint16_t u = (uint16_t)v;            /* well-defined 2's-complement bits */
    p[0] = (uint8_t)(u & 0xFFu);
    p[1] = (uint8_t)((u >> 8) & 0xFFu);
}

static void put_label(uint8_t *p, const kste_label_t *lab) {
    for (int i = 0; i < SP_KSTE_LABEL_DIM; i++)
        put_i16_le(p + 2 * i, lab->s[i]);
}

/* lexicographic compare of two labels (canonical child ordering key) */
static int label_cmp(const kste_label_t *a, const kste_label_t *b) {
    for (int i = 0; i < SP_KSTE_LABEL_DIM; i++) {
        if (a->s[i] < b->s[i]) return -1;
        if (a->s[i] > b->s[i]) return 1;
    }
    return 0;
}

/* ---- slice boundaries ---------------------------------------------------- */

/* Split [lo,hi) into the c-th of B contiguous near-equal sub-slices.
 * Deterministic: the first (m % B) sub-slices get one extra element. */
static void subslice(int lo, int hi, int c, int B, int *o_lo, int *o_hi) {
    int m = hi - lo;
    int base = m / B;
    int extra = m % B;
    int start = lo;
    for (int i = 0; i < c; i++)
        start += base + (i < extra ? 1 : 0);
    int len = base + (c < extra ? 1 : 0);
    *o_lo = start;
    *o_hi = start + len;
}

/* ---- public API ---------------------------------------------------------- */

void sp_kste_encode(const int32_t *vec, int k, sp_kste_tree_t *out) {
    /* zero the whole image first so every reserved byte is deterministic */
    for (int i = 0; i < 64; i++) out->bytes[i] = 0;

    if (k < 0) k = 0;

    /* header */
    out->bytes[SP_KSTE_OFF_VERSION] = (uint8_t)SP_KSTE_LAYOUT_VERSION;
    out->bytes[SP_KSTE_OFF_BRANCH]  = (uint8_t)SP_KSTE_BRANCHING;
    out->bytes[SP_KSTE_OFF_DEPTH]   = (uint8_t)SP_KSTE_DEPTH;
    out->bytes[SP_KSTE_OFF_RESERVED] = 0;
    put_u32_le(out->bytes + SP_KSTE_OFF_K, (uint32_t)k);

    /* Tier-0 root label: statistics over the whole vector */
    kste_label_t root = label_of(vec, 0, k);
    put_label(out->bytes + SP_KSTE_OFF_ROOT, &root);

    /* Tier-1 first-level children: B contiguous slices, each labelled, then
     * sorted into canonical (lexicographic) order. */
    kste_label_t child[SP_KSTE_BRANCHING];
    int clo[SP_KSTE_BRANCHING], chi[SP_KSTE_BRANCHING];
    for (int c = 0; c < SP_KSTE_BRANCHING; c++) {
        subslice(0, k, c, SP_KSTE_BRANCHING, &clo[c], &chi[c]);
        child[c] = label_of(vec, clo[c], chi[c]);
    }

    /* canonical sort of children by label (insertion sort; stable, tiny).
     * We carry the slice bounds alongside so the Tier-2 digest below reflects
     * the same canonical order. */
    int order[SP_KSTE_BRANCHING];
    for (int c = 0; c < SP_KSTE_BRANCHING; c++) order[c] = c;
    for (int i = 1; i < SP_KSTE_BRANCHING; i++) {
        int key = order[i];
        int j = i - 1;
        while (j >= 0 && label_cmp(&child[order[j]], &child[key]) > 0) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = key;
    }

    for (int i = 0; i < SP_KSTE_BRANCHING; i++) {
        const kste_label_t *lab = &child[order[i]];
        put_label(out->bytes + SP_KSTE_OFF_CHILDREN + i * (2 * SP_KSTE_LABEL_DIM),
                  lab);
    }

    /* Tier-2 grandchild digest: 8 bytes. For each canonical child (3) we emit
     * the quantized-to-int8 min of its first two grandchild slices (2 each =
     * 6 bytes), then 2 bytes summarizing the root (min,max as int8). This is a
     * coarse reserved summary; it is deterministic and order-stable but is not
     * part of either dominance tier. */
    uint8_t *g = out->bytes + SP_KSTE_OFF_GRAND;
    int gi = 0;
    for (int i = 0; i < SP_KSTE_BRANCHING && gi < 6; i++) {
        int c = order[i];
        for (int gc = 0; gc < 2 && gi < 6; gc++) {
            int glo, ghi;
            subslice(clo[c], chi[c], gc, SP_KSTE_BRANCHING, &glo, &ghi);
            kste_label_t gl = label_of(vec, glo, ghi);
            /* coarse: high byte of the min, sign-preserving */
            int16_t mn = gl.s[0];
            g[gi++] = (uint8_t)((uint16_t)mn >> 8);
        }
    }
    /* root summary into remaining bytes */
    g[6] = (uint8_t)((uint16_t)root.s[0] >> 8);                       /* min */
    g[7] = (uint8_t)((uint16_t)root.s[SP_KSTE_LABEL_DIM - 1] >> 8);   /* max */
}
