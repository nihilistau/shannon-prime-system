/* ok_int.c — exact integer arithmetic in O_K = Z[w], w = (1+sqrt(-163))/2.
 *
 * See include/sp/ok_int.h for the full mathematical statement.  All operations
 * are exact integer arithmetic on the (a,b) coordinates; the only "search"
 * routines are bounded prime-element discovery and trial-division
 * factorisation, both used by sp_ok_factor.
 */
#include "sp/ok_int.h"

/* ---- construction / inspection ------------------------------------------ */

sp_ok_t sp_ok_from_int(int64_t n) {
    sp_ok_t e = { n, 0 };
    return e;
}

bool sp_ok_is_rational(sp_ok_t alpha) {
    return alpha.b == 0;
}

bool sp_ok_eq(sp_ok_t x, sp_ok_t y) {
    return x.a == y.a && x.b == y.b;
}

bool sp_ok_is_zero(sp_ok_t alpha) {
    return alpha.a == 0 && alpha.b == 0;
}

bool sp_ok_is_unit(sp_ok_t alpha) {
    /* units are exactly the elements of norm 1, i.e. +1 and -1 */
    return alpha.b == 0 && (alpha.a == 1 || alpha.a == -1);
}

/* ---- ring operations ----------------------------------------------------- */

sp_ok_t sp_ok_add(sp_ok_t x, sp_ok_t y) {
    sp_ok_t r = { x.a + y.a, x.b + y.b };
    return r;
}

sp_ok_t sp_ok_sub(sp_ok_t x, sp_ok_t y) {
    sp_ok_t r = { x.a - y.a, x.b - y.b };
    return r;
}

sp_ok_t sp_ok_neg(sp_ok_t x) {
    sp_ok_t r = { -x.a, -x.b };
    return r;
}

sp_ok_t sp_ok_mul(sp_ok_t x, sp_ok_t y) {
    /* (a+bw)(c+dw) = (ac - 41 bd) + (ad + bc + bd) w, using w^2 = w - 41 */
    sp_ok_t r;
    r.a = x.a * y.a - (int64_t)SP_OK_C * x.b * y.b;
    r.b = x.a * y.b + x.b * y.a + x.b * y.b;
    return r;
}

sp_ok_t sp_ok_conj(sp_ok_t x) {
    /* conj(a+bw) = (a+b) - b w */
    sp_ok_t r = { x.a + x.b, -x.b };
    return r;
}

int64_t sp_ok_norm(sp_ok_t x) {
    /* a^2 + a b + 41 b^2 = (a + b w)(a + b conj(w)) */
    return x.a * x.a + x.a * x.b + (int64_t)SP_OK_C * x.b * x.b;
}

int64_t sp_ok_trace(sp_ok_t x) {
    return 2 * x.a + x.b;
}

/* ---- exact division ------------------------------------------------------ */

bool sp_ok_divides(sp_ok_t beta, sp_ok_t alpha, sp_ok_t *quotient) {
    if (sp_ok_is_zero(beta)) return false;

    int64_t n = sp_ok_norm(beta);            /* > 0 since beta != 0 */
    sp_ok_t gamma = sp_ok_mul(alpha, sp_ok_conj(beta));

    if (gamma.a % n != 0 || gamma.b % n != 0) return false;

    if (quotient) {
        quotient->a = gamma.a / n;
        quotient->b = gamma.b / n;
    }
    return true;
}

/* ---- factorisation helpers ----------------------------------------------- */

/* x^2 - x + 41 mod p has a root?  Brute force over residues (p small here). */
static bool sp__poly_has_root_mod(int64_t p) {
    for (int64_t r = 0; r < p; r++) {
        int64_t v = ((r * r - r) % p + (int64_t)SP_OK_C) % p;
        if (v < 0) v += p;
        if (v == 0) return true;
    }
    return false;
}

/* Find a prime element pi of O_K with N(pi) = p, for a rational prime p that
 * splits or ramifies.  Solve a^2 + ab + 41 b^2 = p as a quadratic in a:
 * disc = 4p - 163 b^2 must be a perfect square s, then a = (-b +/- s)/2.
 * Returns true and writes *pi on success. */
static bool sp__prime_element_of_norm(int64_t p, sp_ok_t *pi) {
    int64_t bmax = 0;
    while ((int64_t)163 * (bmax + 1) * (bmax + 1) <= 4 * p) bmax++;
    for (int64_t b = -bmax; b <= bmax; b++) {
        int64_t disc = 4 * p - (int64_t)163 * b * b;
        if (disc < 0) continue;
        int64_t s = 0;
        while ((s + 1) * (s + 1) <= disc) s++;
        if (s * s != disc) continue;
        int64_t num[2] = { -b + s, -b - s };
        for (int k = 0; k < 2; k++) {
            if ((num[k] & 1) != 0) continue;     /* must be even for integer a */
            sp_ok_t e = { num[k] / 2, b };
            if (sp_ok_norm(e) == p) { *pi = e; return true; }
        }
    }
    return false;
}

/* ---- factorisation ------------------------------------------------------- */

int sp_ok_factor(sp_ok_t alpha, sp_ok_t *factors, int max, sp_ok_t *unit_out) {
    if (sp_ok_is_zero(alpha)) return -1;

    /* Unit input: zero factors, the unit is alpha itself. */
    if (sp_ok_is_unit(alpha)) {
        if (unit_out) *unit_out = alpha;
        return 0;
    }

    int count = 0;
    sp_ok_t cur = alpha;          /* shrinks as we divide factors out */
    int64_t N = sp_ok_norm(alpha);

    /* Factor the rational integer N by trial division; for each rational prime
     * p | N decide split/inert/ramified and peel the corresponding
     * irreducible(s) out of `cur` by exact division. */
    for (int64_t p = 2; p * p <= N; p++) {
        if (N % p != 0) continue;
        while (N % p == 0) N /= p;            /* exhaust this prime in N */

        bool ramified = (p == 163);
        bool splits = ramified || sp__poly_has_root_mod(p);

        if (!splits) {
            /* inert: the rational prime p itself is irreducible (norm p^2);
             * it must divide cur (since p^2 | N(cur)). Peel it out. */
            sp_ok_t pe = sp_ok_from_int(p);
            sp_ok_t q;
            while (sp_ok_divides(pe, cur, &q)) {
                if (count >= max) return -1;
                factors[count++] = pe;
                cur = q;
            }
        } else {
            /* split or ramified: find pi of norm p, divide out pi then its
             * conjugate as far as each goes. */
            sp_ok_t pi;
            if (!sp__prime_element_of_norm(p, &pi)) return -1;
            sp_ok_t cj = sp_ok_conj(pi);
            sp_ok_t q;
            while (sp_ok_divides(pi, cur, &q)) {
                if (count >= max) return -1;
                factors[count++] = pi;
                cur = q;
            }
            while (sp_ok_divides(cj, cur, &q)) {
                if (count >= max) return -1;
                factors[count++] = cj;
                cur = q;
            }
        }
    }
    /* If a prime > sqrt(original N) remains, it's a single residual rational
     * prime factor of N. Handle it the same way. */
    if (N > 1) {
        int64_t p = N;
        bool ramified = (p == 163);
        bool splits = ramified || sp__poly_has_root_mod(p);
        if (!splits) {
            sp_ok_t pe = sp_ok_from_int(p);
            sp_ok_t q;
            while (sp_ok_divides(pe, cur, &q)) {
                if (count >= max) return -1;
                factors[count++] = pe;
                cur = q;
            }
        } else {
            sp_ok_t pi;
            if (!sp__prime_element_of_norm(p, &pi)) return -1;
            sp_ok_t cj = sp_ok_conj(pi);
            sp_ok_t q;
            while (sp_ok_divides(pi, cur, &q)) {
                if (count >= max) return -1;
                factors[count++] = pi;
                cur = q;
            }
            while (sp_ok_divides(cj, cur, &q)) {
                if (count >= max) return -1;
                factors[count++] = cj;
                cur = q;
            }
        }
    }

    /* Whatever remains is a unit (+1 or -1); class number 1 guarantees we have
     * fully decomposed. */
    if (!sp_ok_is_unit(cur)) return -1;       /* should not happen; defensive */
    if (unit_out) *unit_out = cur;
    return count;
}
