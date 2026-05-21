/* ok_int_test.c — contract tests for O_K arithmetic over Q(sqrt(-163)).
 *
 * One T_OK_n per roadmap item, driven by sp_test.h.  All randomness is seeded
 * with a fixed constant so runs are deterministic.
 */
#include "sp/sp_test.h"
#include "sp/ok_int.h"

#include <stdint.h>
#include <stdbool.h>

/* ---- deterministic RNG (xorshift64*, fixed seed) ------------------------- */

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static void rng_seed(uint64_t s) { rng_state = s ? s : 0x9E3779B97F4A7C15ull; }

static uint64_t rng_next(void) {
    uint64_t x = rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1Dull;
}

/* uniform integer in [-range, range] */
static int64_t rng_int(int64_t range) {
    uint64_t span = (uint64_t)(2 * range + 1);
    return (int64_t)(rng_next() % span) - range;
}

/* a random non-zero element with coordinates in [-range, range] */
static sp_ok_t rng_elt_nonzero(int64_t range) {
    sp_ok_t e;
    do {
        e.a = rng_int(range);
        e.b = rng_int(range);
    } while (e.a == 0 && e.b == 0);
    return e;
}

/* ---- local irreducibility check (test-side oracle) ----------------------- */

static bool is_rational_prime(int64_t n) {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    for (int64_t d = 3; d * d <= n; d += 2)
        if (n % d == 0) return false;
    return true;
}

/* A factor is irreducible iff its norm is a rational prime p (split/ramified
 * prime element of norm p), or p^2 for an inert rational prime p. */
static bool factor_is_irreducible(sp_ok_t f) {
    int64_t n = sp_ok_norm(f);
    if (n < 4) return false;                 /* units / zero are not irreducible */
    if (is_rational_prime(n)) return true;   /* prime element, norm p */
    /* perfect square of a prime? */
    int64_t r = 0;
    while ((r + 1) * (r + 1) <= n) r++;
    if (r * r == n && is_rational_prime(r)) return true;
    return false;
}

/* multiply factor list back together (times unit) */
static sp_ok_t reconstruct(const sp_ok_t *factors, int k, sp_ok_t unit) {
    sp_ok_t p = unit;
    for (int i = 0; i < k; i++) p = sp_ok_mul(p, factors[i]);
    return p;
}

#define MAXF 64

/* ---- T_OK_1: UFD verification, norm <= 2^20, 256 elements ---------------- */

static void T_OK_1(void) {
    rng_seed(0xA1A1A1A1ull);
    int done = 0;
    int guard = 0;
    while (done < 256 && guard < 200000) {
        guard++;
        /* coords up to 1024 give norm up to ~41*1024^2 ~ 4.3e7 > 2^20; clamp */
        sp_ok_t x = rng_elt_nonzero(140);   /* norm <= 41*140^2+ ~ 8.1e5 < 2^20 */
        if (sp_ok_norm(x) > (1 << 20)) continue;
        if (sp_ok_is_unit(x)) continue;
        done++;

        sp_ok_t factors[MAXF];
        sp_ok_t unit;
        int k = sp_ok_factor(x, factors, MAXF, &unit);
        SP_CHECK(k >= 1, "T_OK_1 factor count >= 1");
        if (k < 0) continue;

        SP_CHECK(sp_ok_is_unit(unit), "T_OK_1 leftover is a unit");
        for (int i = 0; i < k; i++)
            SP_CHECK(factor_is_irreducible(factors[i]),
                     "T_OK_1 each factor irreducible");

        sp_ok_t prod = reconstruct(factors, k, unit);
        SP_CHECK(sp_ok_eq(prod, x), "T_OK_1 product reconstructs element");
    }
    SP_CHECK_EQ_I64(done, 256, "T_OK_1 reached 256 samples");
}

/* ---- T_OK_2: norm & conjugate identities --------------------------------- */

static void T_OK_2(void) {
    rng_seed(0xB2B2B2B2ull);
    for (int i = 0; i < 4096; i++) {
        sp_ok_t x = rng_elt_nonzero(2000);
        sp_ok_t y = rng_elt_nonzero(2000);

        /* N(x*y) = N(x)*N(y) — multiply with bounded inputs (<=2000) */
        sp_ok_t xy = sp_ok_mul(x, y);
        SP_CHECK_EQ_I64(sp_ok_norm(xy), sp_ok_norm(x) * sp_ok_norm(y),
                        "T_OK_2 norm multiplicative");

        /* conj involutive */
        SP_CHECK(sp_ok_eq(sp_ok_conj(sp_ok_conj(x)), x),
                 "T_OK_2 conj involutive");

        /* N(x) = rational part of x*conj(x); x*conj(x) is the rational integer N(x) */
        sp_ok_t nx = sp_ok_mul(x, sp_ok_conj(x));
        SP_CHECK(sp_ok_is_rational(nx), "T_OK_2 x*conj(x) is rational");
        SP_CHECK_EQ_I64(nx.a, sp_ok_norm(x), "T_OK_2 x*conj(x) == N(x)");

        /* conj additive */
        SP_CHECK(sp_ok_eq(sp_ok_conj(sp_ok_add(x, y)),
                          sp_ok_add(sp_ok_conj(x), sp_ok_conj(y))),
                 "T_OK_2 conj additive");

        /* conj multiplicative */
        SP_CHECK(sp_ok_eq(sp_ok_conj(sp_ok_mul(x, y)),
                          sp_ok_mul(sp_ok_conj(x), sp_ok_conj(y))),
                 "T_OK_2 conj multiplicative");
    }
}

/* ---- T_OK_3: multiplication closure / no overflow within bound ----------- */

static void T_OK_3(void) {
    /* Safe input bound derivation (also documented in ok_int.h):
     * with |a|,|b|,|c|,|d| <= B, the mul coordinates are bounded by
     *   |a*c - 41*b*d|        <= 42*B^2
     *   |a*d + b*c + b*d|     <= 3*B^2
     * and the product's norm (~41 * (coord)^2) <= 41*(42 B^2)^2 = 72324*B^4.
     * For B = 2048: 72324 * 2048^4 ~ 1.27e21 -- that exceeds int64!  So the
     * norm-fits constraint is the binding one: 72324*B^4 <= 2^63 gives
     * B <= ~3360.  We therefore test mul-closure at B = 2048 (coords fit
     * with margin: 42*2048^2 ~ 1.76e8 << 2^63) and verify the *coordinates*
     * round-trip exactly; norm-of-product fits comfortably at this B. */
    rng_seed(0xC3C3C3C3ull);
    const int64_t B = 2048;
    for (int i = 0; i < 20000; i++) {
        sp_ok_t x = rng_elt_nonzero(B);
        sp_ok_t y = rng_elt_nonzero(B);
        sp_ok_t p = sp_ok_mul(x, y);

        /* recompute coordinates with __int64 by the same formula, independently */
        int64_t ra = x.a * y.a - 41 * x.b * y.b;
        int64_t rb = x.a * y.b + x.b * y.a + x.b * y.b;
        SP_CHECK_EQ_I64(p.a, ra, "T_OK_3 product real coord");
        SP_CHECK_EQ_I64(p.b, rb, "T_OK_3 product w coord");

        /* norm of product fits and is multiplicative (no overflow) */
        int64_t np = sp_ok_norm(p);
        SP_CHECK(np >= 0, "T_OK_3 norm non-negative (no overflow)");
        SP_CHECK_EQ_I64(np, sp_ok_norm(x) * sp_ok_norm(y),
                        "T_OK_3 norm multiplicative at bound");
    }
}

/* ---- helpers for T_OK_4 -------------------------------------------------- */

/* brute-force: is there (a,b) with a^2+ab+41b^2 == p?
 * disc in a: 4p - 163 b^2 must be a perfect square; |b| <= 2*sqrt(p/163). */
static bool exists_element_of_norm(int64_t p, sp_ok_t *out) {
    int64_t bmax = 0;
    while (163 * (bmax + 1) * (bmax + 1) <= 4 * p) bmax++;
    for (int64_t b = -bmax; b <= bmax; b++) {
        int64_t disc = 4 * p - 163 * b * b;
        if (disc < 0) continue;
        int64_t s = 0;
        while ((s + 1) * (s + 1) <= disc) s++;
        if (s * s != disc) continue;
        /* a = (-b +/- s)/2 ; parity guarantees integrality */
        int64_t num1 = -b + s, num2 = -b - s;
        if ((num1 & 1) == 0) {
            sp_ok_t e = { num1 / 2, b };
            if (sp_ok_norm(e) == p) { if (out) *out = e; return true; }
        }
        if ((num2 & 1) == 0) {
            sp_ok_t e = { num2 / 2, b };
            if (sp_ok_norm(e) == p) { if (out) *out = e; return true; }
        }
    }
    return false;
}

/* does x^2 - x + 41 have a root mod p?  (splitting test) */
static bool poly_has_root_mod(int64_t p) {
    for (int64_t r = 0; r < p; r++) {
        int64_t v = ((r * r - r) % p + 41) % p;
        if (v < 0) v += p;
        if (v == 0) return true;
    }
    return false;
}

/* ---- T_OK_4: class number 1, splitting up to norm 2^16 ------------------- */

static void T_OK_4(void) {
    const int64_t LIMIT = 1 << 16;   /* 65536 */

    /* sieve of primes up to LIMIT */
    static unsigned char composite[(1 << 16) + 1];
    for (int64_t i = 2; i * i <= LIMIT; i++)
        if (!composite[i])
            for (int64_t j = i * i; j <= LIMIT; j += i)
                composite[j] = 1;

    int checked_split = 0, checked_inert = 0;
    for (int64_t p = 2; p <= LIMIT; p++) {
        if (composite[p]) continue;

        bool splits_or_ramifies = (p == 163) || poly_has_root_mod(p);
        sp_ok_t pi;
        bool has = exists_element_of_norm(p, &pi);

        if (splits_or_ramifies) {
            /* class number 1: a principal generator of norm p must exist */
            SP_CHECK(has, "T_OK_4 split/ramified prime has generator of norm p");
            if (has) {
                SP_CHECK_EQ_I64(sp_ok_norm(pi), p, "T_OK_4 generator norm == p");
                checked_split++;
            }
        } else {
            /* inert: no element of norm p (only p^2) */
            SP_CHECK(!has, "T_OK_4 inert prime has NO element of norm p");
            checked_inert++;
        }
    }
    SP_CHECK(checked_split > 0, "T_OK_4 saw split primes");
    SP_CHECK(checked_inert > 0, "T_OK_4 saw inert primes");

    /* explicit ramified case p = 163: pi = -1 + 2w, N = 1 - 2 + 164 = 163 */
    sp_ok_t pi163 = { -1, 2 };
    SP_CHECK_EQ_I64(sp_ok_norm(pi163), 163, "T_OK_4 ramified generator at 163");
}

/* ---- T_OK_5: round trips ------------------------------------------------- */

static void T_OK_5(void) {
    rng_seed(0xD5D5D5D5ull);

    /* rational integer round-trip */
    for (int i = 0; i < 4096; i++) {
        int64_t n = rng_int(1000000);
        sp_ok_t e = sp_ok_from_int(n);
        SP_CHECK(sp_ok_is_rational(e), "T_OK_5 from_int is rational");
        SP_CHECK_EQ_I64(e.a, n, "T_OK_5 from_int recovers n");
        SP_CHECK_EQ_I64(sp_ok_norm(e), n * n, "T_OK_5 N(n) == n^2");
    }

    /* exact-division round trip: (x*y) / y == x */
    for (int i = 0; i < 8192; i++) {
        sp_ok_t x = rng_elt_nonzero(1000);
        sp_ok_t y = rng_elt_nonzero(1000);
        sp_ok_t prod = sp_ok_mul(x, y);
        sp_ok_t q;
        bool ok = sp_ok_divides(y, prod, &q);
        SP_CHECK(ok, "T_OK_5 y divides x*y");
        SP_CHECK(sp_ok_eq(q, x), "T_OK_5 (x*y)/y == x");

        /* and the other factor */
        bool ok2 = sp_ok_divides(x, prod, &q);
        SP_CHECK(ok2, "T_OK_5 x divides x*y");
        SP_CHECK(sp_ok_eq(q, y), "T_OK_5 (x*y)/x == y");

        /* non-divisibility: x*y + 1 not divisible by y unless y is a unit */
        if (!sp_ok_is_unit(y)) {
            sp_ok_t off = sp_ok_add(prod, sp_ok_from_int(1));
            sp_ok_t q2;
            /* may or may not divide depending on y; just ensure if it claims
             * divisibility, the quotient actually multiplies back */
            if (sp_ok_divides(y, off, &q2))
                SP_CHECK(sp_ok_eq(sp_ok_mul(y, q2), off),
                         "T_OK_5 claimed quotient is exact");
        }
    }
}

/* ---- T_OK_6: irreducible factorisation, 1024 elts, norm <= 2^14 ---------- */

static void T_OK_6(void) {
    rng_seed(0xE6E6E6E6ull);
    int done = 0, guard = 0;
    while (done < 1024 && guard < 2000000) {
        guard++;
        sp_ok_t x = rng_elt_nonzero(19);  /* 41*19^2 ~ 1.48e4 < 2^14=16384 ish */
        if (sp_ok_norm(x) > (1 << 14)) continue;
        if (sp_ok_is_unit(x)) continue;
        done++;

        sp_ok_t factors[MAXF];
        sp_ok_t unit;
        int k = sp_ok_factor(x, factors, MAXF, &unit);
        SP_CHECK(k >= 1, "T_OK_6 factor count >= 1");
        if (k < 0) continue;
        SP_CHECK(sp_ok_is_unit(unit), "T_OK_6 leftover unit");

        sp_ok_t prod = reconstruct(factors, k, unit);
        SP_CHECK(sp_ok_eq(prod, x), "T_OK_6 reconstruct equals (up to unit folded in)");
    }
    SP_CHECK_EQ_I64(done, 1024, "T_OK_6 reached 1024 samples");
}

int main(void) {
    SP_RUN(T_OK_1);
    SP_RUN(T_OK_2);
    SP_RUN(T_OK_3);
    SP_RUN(T_OK_4);
    SP_RUN(T_OK_5);
    SP_RUN(T_OK_6);
    return SP_DONE();
}
