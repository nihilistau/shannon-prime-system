/* ntt_crt.c — dual-prime CRT negacyclic NTT kernel (production path).
 *
 * NO 128-bit integer type anywhere in this file (enforced at configure time,
 * T_NTT_5). All modular arithmetic is done in uint64 with Barrett reduction;
 * the two operands are < 2^30 so their product is < 2^60 and every Barrett
 * intermediate stays < 2^64 (see modmul()'s overflow note).
 *
 * Ring R_q = Z_q[x]/(x^N + 1), negacyclic. Per prime q we precompute a
 * primitive 2N-th root psi (psi^N = -1, ord(psi) = 2N) and omega = psi^2, a
 * primitive N-th root. Forward = pre-weight coeff j by psi^j, then a standard
 * length-N NTT in omega; inverse = NTT in omega^{-1}, scale by N^{-1}, then
 * post-weight by psi^{-j}. The two residue channels (mod q1, mod q2) are
 * recombined to mod M = q1*q2 by symmetric Garner CRT.
 */
#include "sp/ntt_crt.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

/* ---- frozen primes ------------------------------------------------------- */

#define Q1 SP_NTT_Q1            /* 1073738753, 30-bit prime */
#define Q2 SP_NTT_Q2            /* 1073732609, 30-bit prime */

/* Both primes are 30-bit; Barrett uses BITS = 30. */
#define Q_BITS 30u

_Static_assert(Q1 < (1u << 30), "q1 must be 30-bit");
_Static_assert(Q2 < (1u << 30), "q2 must be 30-bit");
_Static_assert(SP_NTT_M == (int64_t)Q1 * (int64_t)Q2, "M = q1*q2");

/* Per-prime parameters and twiddle tables, computed once in ntt_init. */
typedef struct {
    uint32_t  q;          /* the modulus */
    uint64_t  mu;         /* Barrett constant floor(2^60 / q), ~31-bit */
    uint32_t  psi;        /* primitive 2N-th root: psi^N = -1 (mod q) */
    uint32_t  ninv;       /* N^{-1} mod q */
    uint32_t *psi_pow;    /* psi^j      for j in [0,N)  (pre-weight) */
    uint32_t *ipsi_pow;   /* psi^{-j}   for j in [0,N)  (post-weight) */
    uint32_t *w_fwd;      /* omega^j    for j in [0,N/2) (forward butterflies) */
    uint32_t *w_inv;      /* omega^{-j} for j in [0,N/2) (inverse butterflies) */
} prime_ctx;

struct ntt_ctx {
    uint32_t   N;
    uint32_t   logN;
    uint32_t  *bitrev;    /* bit-reversal permutation of [0,N) */
    prime_ctx  p1;
    prime_ctx  p2;
    /* CRT (Garner): q1^{-1} mod q2, plus M and M/2 for centering. */
    uint64_t   q1_inv_mod_q2;
    uint64_t   M;
};

/* ---- modular arithmetic (Barrett, no 128-bit type) ----------------------- */

/* Barrett reduce of x = a*b (both < q < 2^30, so x < 2^60) modulo q.
 *
 * Constant: mu = floor(2^60 / q), which is ~31-bit (< 2^32).
 * Step:  qhat = ((x >> (Q_BITS-1)) * mu) >> (Q_BITS+1)        = ((x>>29)*mu)>>31
 *        r    = x - qhat*q,  then up to 2 conditional subtractions.
 *
 * Overflow argument (every value below stays < 2^64):
 *   x        = a*b               < q^2          < 2^60
 *   x >> 29                                     < 2^31
 *   (x>>29)*mu                   < 2^31 * 2^32  = 2^63   (measured: < 2^61)
 *   qhat = ((x>>29)*mu) >> 31                   < 2^32
 *   qhat*q                       < 2^32 * 2^30  = 2^62
 *   r = x - qhat*q  is in [0, ~2q); 2 conditional subs land it in [0,q).
 * Nothing exceeds 2^64, so plain uint64 suffices.                          */
static inline uint64_t barrett_reduce(uint64_t x, uint64_t q, uint64_t mu) {
    uint64_t qhat = ((x >> (Q_BITS - 1u)) * mu) >> (Q_BITS + 1u);
    uint64_t r = x - qhat * q;
    if (r >= q) r -= q;
    if (r >= q) r -= q;
    return r;
}

static inline uint32_t modmul(uint32_t a, uint32_t b, uint32_t q, uint64_t mu) {
    uint64_t x = (uint64_t)a * (uint64_t)b;           /* < 2^60 */
    return (uint32_t)barrett_reduce(x, (uint64_t)q, mu);
}

static inline uint32_t modadd(uint32_t a, uint32_t b, uint32_t q) {
    uint32_t s = a + b;
    if (s >= q) s -= q;
    return s;
}

static inline uint32_t modsub(uint32_t a, uint32_t b, uint32_t q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

/* x^e mod q via square-and-multiply, using Barrett modmul. */
static uint32_t modpow(uint32_t base, uint64_t e, uint32_t q, uint64_t mu) {
    uint32_t result = 1u % q;
    uint32_t b = base % q;
    while (e) {
        if (e & 1u) result = modmul(result, b, q, mu);
        b = modmul(b, b, q, mu);
        e >>= 1;
    }
    return result;
}

static uint32_t modinv(uint32_t a, uint32_t q, uint64_t mu) {
    /* q prime => a^{q-2} is the inverse (Fermat). */
    return modpow(a, (uint64_t)q - 2u, q, mu);
}

/* ---- twiddle / context setup --------------------------------------------- */

/* Find primitive 2N-th root psi: smallest base a>=2 with
 * (a^((q-1)/(2N)))^N == -1 (mod q). Returns psi. */
static uint32_t find_psi(uint32_t N, uint32_t q, uint64_t mu) {
    uint64_t exp = ((uint64_t)q - 1u) / (2u * (uint64_t)N);
    for (uint32_t a = 2u; a < q; a++) {
        uint32_t psi = modpow(a, exp, q, mu);
        if (psi == 0u) continue;
        uint32_t pN = modpow(psi, (uint64_t)N, q, mu);
        if (pN == q - 1u) {                    /* psi^N == -1 */
            return psi;
        }
    }
    return 0u; /* unreachable for the frozen primes & N<=512 */
}

static int prime_setup(prime_ctx *pc, uint32_t N, uint32_t q) {
    pc->q  = q;
    pc->mu = ((uint64_t)1 << 60) / (uint64_t)q;       /* floor(2^60/q), ~31-bit */

    uint32_t psi = find_psi(N, q, pc->mu);
    if (psi == 0u) return 0;
    pc->psi = psi;

    /* init-time invariants the math requires */
    assert(modpow(psi, (uint64_t)N, q, pc->mu) == q - 1u);        /* psi^N = -1 */
    assert(modpow(psi, 2u * (uint64_t)N, q, pc->mu) == 1u % q);   /* psi^2N = 1 */

    uint32_t ipsi  = modinv(psi, q, pc->mu);
    uint32_t omega = modmul(psi, psi, q, pc->mu);
    uint32_t iomega = modmul(ipsi, ipsi, q, pc->mu);
    pc->ninv = modinv(N % q, q, pc->mu);

    pc->psi_pow  = malloc(sizeof(uint32_t) * N);
    pc->ipsi_pow = malloc(sizeof(uint32_t) * N);
    pc->w_fwd    = malloc(sizeof(uint32_t) * (N / 2u));
    pc->w_inv    = malloc(sizeof(uint32_t) * (N / 2u));
    if (!pc->psi_pow || !pc->ipsi_pow || !pc->w_fwd || !pc->w_inv) return 0;

    uint32_t acc = 1u % q;
    for (uint32_t j = 0; j < N; j++) {       /* psi^j */
        pc->psi_pow[j] = acc;
        acc = modmul(acc, psi, q, pc->mu);
    }
    acc = 1u % q;
    for (uint32_t j = 0; j < N; j++) {       /* psi^{-j} */
        pc->ipsi_pow[j] = acc;
        acc = modmul(acc, ipsi, q, pc->mu);
    }
    acc = 1u % q;
    for (uint32_t j = 0; j < N / 2u; j++) {  /* omega^j */
        pc->w_fwd[j] = acc;
        acc = modmul(acc, omega, q, pc->mu);
    }
    acc = 1u % q;
    for (uint32_t j = 0; j < N / 2u; j++) {  /* omega^{-j} */
        pc->w_inv[j] = acc;
        acc = modmul(acc, iomega, q, pc->mu);
    }
    return 1;
}

static void prime_free(prime_ctx *pc) {
    free(pc->psi_pow);
    free(pc->ipsi_pow);
    free(pc->w_fwd);
    free(pc->w_inv);
}

static uint32_t ilog2(uint32_t n) {
    uint32_t l = 0;
    while ((1u << l) < n) l++;
    return l;
}

ntt_ctx *ntt_init(uint32_t N) {
    if (N != 128u && N != 256u && N != 512u) return NULL;

    ntt_ctx *ctx = calloc(1, sizeof(ntt_ctx));
    if (!ctx) return NULL;
    ctx->N    = N;
    ctx->logN = ilog2(N);
    ctx->M    = (uint64_t)SP_NTT_M;

    ctx->bitrev = malloc(sizeof(uint32_t) * N);
    if (!ctx->bitrev) { free(ctx); return NULL; }
    uint32_t logN = ctx->logN;
    for (uint32_t i = 0; i < N; i++) {
        uint32_t r = 0, x = i;
        for (uint32_t b = 0; b < logN; b++) { r = (r << 1) | (x & 1u); x >>= 1; }
        ctx->bitrev[i] = r;
    }

    if (!prime_setup(&ctx->p1, N, Q1) || !prime_setup(&ctx->p2, N, Q2)) {
        ntt_free(ctx);
        return NULL;
    }

    /* CRT: q1^{-1} mod q2 (used by symmetric Garner). */
    ctx->q1_inv_mod_q2 = modinv(Q1 % Q2, Q2, ctx->p2.mu);

    return ctx;
}

void ntt_free(ntt_ctx *ctx) {
    if (!ctx) return;
    prime_free(&ctx->p1);
    prime_free(&ctx->p2);
    free(ctx->bitrev);
    free(ctx);
}

/* ---- core length-N NTT (in-place, decimation-in-time, omega twiddles) ---- */

/* Cooley-Tukey: input is bit-reversed up front, then logN stages of radix-2
 * butterflies. `wtab` holds omega^j for j in [0,N/2). */
static void ntt_core(const ntt_ctx *ctx, const prime_ctx *pc,
                     uint32_t *x, const uint32_t *wtab) {
    uint32_t N = ctx->N;
    uint32_t q = pc->q;
    uint64_t mu = pc->mu;

    /* bit-reversal permutation */
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = ctx->bitrev[i];
        if (i < j) { uint32_t t = x[i]; x[i] = x[j]; x[j] = t; }
    }

    for (uint32_t len = 2; len <= N; len <<= 1) {
        uint32_t half = len >> 1;
        uint32_t step = N / len;                 /* twiddle index stride */
        for (uint32_t i = 0; i < N; i += len) {
            uint32_t widx = 0;
            for (uint32_t k = 0; k < half; k++) {
                uint32_t u = x[i + k];
                uint32_t v = modmul(x[i + k + half], wtab[widx], q, mu);
                x[i + k]        = modadd(u, v, q);
                x[i + k + half] = modsub(u, v, q);
                widx += step;
            }
        }
    }
}

/* ---- forward / inverse transforms ---------------------------------------- */

static void forward_one(const ntt_ctx *ctx, const prime_ctx *pc,
                        const int32_t *in, uint32_t *out) {
    uint32_t N = ctx->N;
    uint32_t q = pc->q;
    uint64_t mu = pc->mu;

    /* reduce signed input into [0,q), then pre-weight coeff j by psi^j */
    for (uint32_t j = 0; j < N; j++) {
        int64_t v = (int64_t)in[j] % (int64_t)q;
        if (v < 0) v += (int64_t)q;
        out[j] = modmul((uint32_t)v, pc->psi_pow[j], q, mu);
    }
    ntt_core(ctx, pc, out, pc->w_fwd);
}

void ntt_forward(const ntt_ctx *ctx, const int32_t *in,
                 uint32_t *out1, uint32_t *out2) {
    forward_one(ctx, &ctx->p1, in, out1);
    forward_one(ctx, &ctx->p2, in, out2);
}

/* Read-only forward-plan view for out-of-tree batch implementations (the
 * engine's lane-parallel sp_ntt_fwd_batch override). Exposes only what an
 * equivalent FORWARD transform needs; inverse tables stay private. */
int ntt_fwd_plan_get(const ntt_ctx *ctx, ntt_fwd_plan *plan_out) {
    if (!ctx || !plan_out) return 0;
    plan_out->N      = ctx->N;
    plan_out->logN   = ctx->logN;
    plan_out->bitrev = ctx->bitrev;
    const prime_ctx *pcs[2] = { &ctx->p1, &ctx->p2 };
    for (int i = 0; i < 2; i++) {
        plan_out->p[i].q       = pcs[i]->q;
        plan_out->p[i].mu      = pcs[i]->mu;
        plan_out->p[i].psi_pow = pcs[i]->psi_pow;
        plan_out->p[i].w_fwd   = pcs[i]->w_fwd;
    }
    return 1;
}

/* inverse transform for one prime, producing residues in [0,q). */
static void inverse_one(const ntt_ctx *ctx, const prime_ctx *pc,
                        const uint32_t *in, uint32_t *out) {
    uint32_t N = ctx->N;
    uint32_t q = pc->q;
    uint64_t mu = pc->mu;

    for (uint32_t j = 0; j < N; j++) out[j] = in[j] % q;
    ntt_core(ctx, pc, out, pc->w_inv);
    /* scale by N^{-1}, then post-weight coeff j by psi^{-j} */
    for (uint32_t j = 0; j < N; j++) {
        uint32_t s = modmul(out[j], pc->ninv, q, mu);
        out[j] = modmul(s, pc->ipsi_pow[j], q, mu);
    }
}

/* ---- CRT (symmetric Garner, no 128-bit type) ----------------------------- */

/* Recombine one residue pair into the signed centered residue mod M.
 *   t = (x2 - x1) * (q1^{-1} mod q2)  mod q2        in [0, q2)
 *   r = x1 + q1 * t                                 in [0, M)
 * q1*t <= q1*(q2-1) < M, and x1 < q1, so r < M fits uint64 (M ~ 2^60).
 * Then center to (-M/2, M/2].                                              */
static inline int64_t garner_one(const ntt_ctx *ctx, uint32_t x1, uint32_t x2) {
    uint32_t q1 = ctx->p1.q;
    uint32_t q2 = ctx->p2.q;
    uint64_t mu2 = ctx->p2.mu;

    uint32_t d = modsub(x2 % q2, x1 % q2, q2);            /* (x2 - x1) mod q2 */
    uint32_t t = (uint32_t)barrett_reduce((uint64_t)d * ctx->q1_inv_mod_q2,
                                          (uint64_t)q2, mu2);
    uint64_t r = (uint64_t)x1 + (uint64_t)q1 * (uint64_t)t; /* < M */
    int64_t v = (int64_t)r;
    int64_t M = (int64_t)ctx->M;
    if (v > M / 2) v -= M;
    return v;
}

void ntt_crt_recombine(const ntt_ctx *ctx, const uint32_t *x1,
                       const uint32_t *x2, int64_t *out) {
    for (uint32_t i = 0; i < ctx->N; i++) out[i] = garner_one(ctx, x1[i], x2[i]);
}

void ntt_inverse(const ntt_ctx *ctx, const uint32_t *in1, const uint32_t *in2,
                 int64_t *out) {
    uint32_t N = ctx->N;
    uint32_t *t1 = malloc(sizeof(uint32_t) * N);
    uint32_t *t2 = malloc(sizeof(uint32_t) * N);
    if (!t1 || !t2) { free(t1); free(t2); return; }

    inverse_one(ctx, &ctx->p1, in1, t1);
    inverse_one(ctx, &ctx->p2, in2, t2);
    ntt_crt_recombine(ctx, t1, t2, out);

    free(t1);
    free(t2);
}

/* ---- pointwise multiply (per-residue, no cross-prime mixing) -------------- */

void ntt_pointwise_mul(const ntt_ctx *ctx,
                       const uint32_t *a1, const uint32_t *a2,
                       const uint32_t *b1, const uint32_t *b2,
                       uint32_t *out1, uint32_t *out2) {
    uint32_t N = ctx->N;
    uint32_t q1 = ctx->p1.q; uint64_t mu1 = ctx->p1.mu;
    uint32_t q2 = ctx->p2.q; uint64_t mu2 = ctx->p2.mu;
    for (uint32_t i = 0; i < N; i++) {
        out1[i] = modmul(a1[i], b1[i], q1, mu1);
        out2[i] = modmul(a2[i], b2[i], q2, mu2);
    }
}
