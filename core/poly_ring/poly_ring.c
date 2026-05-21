/* poly_ring.c — R_q = Z_q[x]/(x^N+1) polynomial-ring attention (Phase 1C).
 *
 * Implemented strictly on top of the Phase-1B CRT-NTT kernel (sp/ntt_crt.h):
 * the negacyclic product of two polynomials is
 *     forward(a) -> (a1,a2);  forward(b) -> (b1,b2);
 *     pointwise_mul per residue -> (c1,c2);  inverse+CRT recombine -> out.
 * No ring arithmetic is re-implemented here; this module is a thin algebraic
 * layer (involution + score formation + softmax) over the kernel.
 *
 * Anti-contamination: math derived fresh from PPT-LAT-Theory §2.1 (negacyclic
 * x^N=-1) and §6.1 (involution k*_0=k_0, k*_j=-k_{N-j}). No legacy source read.
 */
#include "sp/poly_ring.h"
#include "sp/ntt_crt.h"

#include <stdlib.h>
#include <math.h>

struct sp_pr_ctx {
    ntt_ctx *ntt;
    uint32_t N;
    /* Per-call scratch, sized N, owned by the context so sp_pr_attention over
     * many keys does not re-malloc. The context is therefore single-thread. */
    uint32_t *a1, *a2;     /* residue domain of left operand            */
    uint32_t *b1, *b2;     /* residue domain of right operand           */
    uint32_t *c1, *c2;     /* residue domain of pointwise product       */
    int64_t  *prod;        /* recombined product coefficients (N)       */
    int32_t  *invk;        /* involuted key buffer for sp_pr_inner (N)  */
};

sp_pr_ctx *sp_pr_init(uint32_t N) {
    ntt_ctx *ntt = ntt_init(N);     /* rejects N outside {128,256,512} */
    if (!ntt) return NULL;

    sp_pr_ctx *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) { ntt_free(ntt); return NULL; }
    ctx->ntt = ntt;
    ctx->N = N;

    ctx->a1   = malloc(sizeof(uint32_t) * N);
    ctx->a2   = malloc(sizeof(uint32_t) * N);
    ctx->b1   = malloc(sizeof(uint32_t) * N);
    ctx->b2   = malloc(sizeof(uint32_t) * N);
    ctx->c1   = malloc(sizeof(uint32_t) * N);
    ctx->c2   = malloc(sizeof(uint32_t) * N);
    ctx->prod = malloc(sizeof(int64_t)  * N);
    ctx->invk = malloc(sizeof(int32_t)  * N);

    if (!ctx->a1 || !ctx->a2 || !ctx->b1 || !ctx->b2 ||
        !ctx->c1 || !ctx->c2 || !ctx->prod || !ctx->invk) {
        sp_pr_free(ctx);
        return NULL;
    }
    return ctx;
}

void sp_pr_free(sp_pr_ctx *ctx) {
    if (!ctx) return;
    ntt_free(ctx->ntt);
    free(ctx->a1); free(ctx->a2);
    free(ctx->b1); free(ctx->b2);
    free(ctx->c1); free(ctx->c2);
    free(ctx->prod); free(ctx->invk);
    free(ctx);
}

uint32_t sp_pr_degree(const sp_pr_ctx *ctx) {
    return ctx ? ctx->N : 0u;
}

/* Negacyclic product into `out` (N entries). Uses the context's residue
 * scratch; `out` is caller-owned and must not alias a or b. */
void sp_pr_mul(sp_pr_ctx *ctx, const int32_t *a, const int32_t *b,
               int64_t *out) {
    ntt_forward(ctx->ntt, a, ctx->a1, ctx->a2);
    ntt_forward(ctx->ntt, b, ctx->b1, ctx->b2);
    ntt_pointwise_mul(ctx->ntt, ctx->a1, ctx->a2, ctx->b1, ctx->b2,
                      ctx->c1, ctx->c2);
    ntt_inverse(ctx->ntt, ctx->c1, ctx->c2, out);
}

/* <q,k> = (q (x) k*)_0 with k*_0=k_0, k*_j=-k_{N-j} (j>0).
 *
 * Why coefficient 0. In the negacyclic product c = q (x) k*,
 *     c_0 = sum_{i+j=0} q_i k*_j - sum_{i+j=N} q_i k*_j
 *         = q_0 k*_0 - sum_{i=1..N-1} q_i k*_{N-i}.
 * The involution gives k*_{N-i} = -k_{N-(N-i)} = -k_i for i in [1,N-1] and
 * k*_0 = k_0, so
 *     c_0 = q_0 k_0 - sum_{i=1..N-1} q_i (-k_i) = sum_{i=0..N-1} q_i k_i = <q,k>.
 * The minus sign of the negacyclic wraparound (x^N=-1) exactly cancels the
 * minus sign in the involution; coefficient 0 is the only one that collects
 * every i=j-aligned term with no cross-contamination. Exact in Z while
 * |<q,k>| < M/2 (centered recovery), which holds with large margin for the
 * supported degrees and modest coefficient magnitudes. */
int64_t sp_pr_inner(sp_pr_ctx *ctx, const int32_t *q, const int32_t *k) {
    const uint32_t N = ctx->N;
    int32_t *ks = ctx->invk;
    ks[0] = k[0];
    for (uint32_t j = 1; j < N; j++) {
        ks[j] = -k[N - j];          /* k*_j = -k_{N-j} */
    }
    sp_pr_mul(ctx, q, ks, ctx->prod);
    return ctx->prod[0];
}

void sp_pr_attention(sp_pr_ctx *ctx, const int32_t *q,
                     const int32_t *const *keys, int n_keys,
                     double *probs_out) {
    if (n_keys <= 0) return;

    /* Recover the integer scores via the ring inner product. The softmax below
     * is the textbook numerically-stable form: find the max integer score,
     * subtract it in INTEGER arithmetic, then exp/normalise in double. Keeping
     * the subtract integral (score - mx before the cast) avoids any rounding
     * ambiguity. n_keys is small (the key set per query); a transient buffer is
     * cheap and keeps the score vector off the hot scratch. */
    int64_t *score = malloc(sizeof(int64_t) * (size_t)n_keys);
    if (!score) { for (int i = 0; i < n_keys; i++) probs_out[i] = 0.0; return; }

    int64_t mx = 0;
    for (int i = 0; i < n_keys; i++) {
        int64_t s = sp_pr_inner(ctx, q, keys[i]);
        score[i] = s;
        if (i == 0 || s > mx) mx = s;
    }

    double sum = 0.0;
    for (int i = 0; i < n_keys; i++) {
        double e = exp((double)(score[i] - mx));
        probs_out[i] = e;
        sum += e;
    }
    for (int i = 0; i < n_keys; i++) probs_out[i] /= sum;

    free(score);
}
