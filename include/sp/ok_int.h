/* sp/ok_int.h — exact integer arithmetic in the ring of integers O_K of the
 * imaginary quadratic field Q(sqrt(-163)).  (Phase 1A of the math core.)
 *
 * Field:        K = Q(sqrt(-163)), discriminant d = -163.
 * Since -163 = 1 (mod 4), O_K = Z[w] with  w = (1 + sqrt(-163)) / 2.
 * Every element is  alpha = a + b*w,  a, b in Z  (stored as int64_t).
 *
 * Defining relations (all derivable from the minimal polynomial of w):
 *   w^2          = w - 41                       (min poly x^2 - x + 41)
 *   conj(a+b*w)  = (a + b) - b*w
 *   N(a+b*w)     = a^2 + a*b + 41*b^2   (>= 0, = 0 only for the zero element)
 *   Tr(a+b*w)    = 2*a + b
 *   (a+b*w)(c+d*w) = (a*c - 41*b*d) + (a*d + b*c + b*d)*w
 *
 * Q(sqrt(-163)) is a Heegner field: O_K is a unique factorisation domain with
 * class number 1, and its only units are +1 and -1.
 *
 * This header is part of the shared public API root (include/).  Include it as
 *   #include "sp/ok_int.h"
 */
#ifndef SP_OK_INT_H
#define SP_OK_INT_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* An element a + b*w of O_K, w = (1 + sqrt(-163))/2. */
typedef struct {
    int64_t a;
    int64_t b;
} sp_ok_t;

/* The half-trace constant: w^2 = w - SP_OK_C, N(a+bw) = a^2 + ab + C b^2,
 * where C = (1 - d)/4 = (1 + 163)/4 = 41. */
#define SP_OK_C 41

/* ---- construction / inspection ------------------------------------------ */

/* Embed a rational integer n as n + 0*w. */
sp_ok_t sp_ok_from_int(int64_t n);

/* True iff alpha is a rational integer (b == 0). */
bool sp_ok_is_rational(sp_ok_t alpha);

/* True iff the two elements are equal. */
bool sp_ok_eq(sp_ok_t x, sp_ok_t y);

/* True iff alpha == 0. */
bool sp_ok_is_zero(sp_ok_t alpha);

/* True iff alpha is a unit (+1 or -1). */
bool sp_ok_is_unit(sp_ok_t alpha);

/* ---- ring operations ----------------------------------------------------- */

sp_ok_t sp_ok_add(sp_ok_t x, sp_ok_t y);
sp_ok_t sp_ok_sub(sp_ok_t x, sp_ok_t y);
sp_ok_t sp_ok_neg(sp_ok_t x);

/* (a*c - 41*b*d) + (a*d + b*c + b*d)*w.
 *
 * Safe-input bound (see ok_int.c / test T_OK_3): if |a|,|b|,|c|,|d| <= 2048
 * every intermediate and the result coordinate stay well within int64_t, and
 * the norm of the product also fits.  Out of that range overflow is the
 * caller's responsibility. */
sp_ok_t sp_ok_mul(sp_ok_t x, sp_ok_t y);

/* Galois conjugate: (a+b) - b*w.  Involutive; a ring homomorphism. */
sp_ok_t sp_ok_conj(sp_ok_t x);

/* Field norm N(a+b*w) = a^2 + a*b + 41*b^2.  Always >= 0; multiplicative.
 * Returned signed because it never overflows within the documented bound and
 * a signed type keeps -Wconversion quiet at call sites. */
int64_t sp_ok_norm(sp_ok_t x);

/* Trace Tr(a+b*w) = 2*a + b. */
int64_t sp_ok_trace(sp_ok_t x);

/* ---- exact division ------------------------------------------------------ */

/* Exact division in O_K.  Returns true iff beta divides alpha exactly; when it
 * does and quotient != NULL, *quotient is set to alpha/beta.
 *
 * Method: alpha/beta = alpha*conj(beta) / N(beta).  Let gamma = alpha*conj(beta)
 * (an O_K element) and n = N(beta) (a rational integer).  beta | alpha iff n
 * divides both integer coordinates of gamma; the quotient is
 * (gamma.a/n) + (gamma.b/n)*w.  beta == 0 returns false. */
bool sp_ok_divides(sp_ok_t beta, sp_ok_t alpha, sp_ok_t *quotient);

/* ---- factorisation ------------------------------------------------------- */

/* Factor alpha into irreducibles of O_K.
 *
 * Writes up to `max` irreducible factors into `factors[]` and returns the
 * count, or -1 if `max` was too small / alpha was invalid for factoring.
 * The decomposition is  alpha = unit * prod(factors[i]),  with the leftover
 * unit (+1 or -1) written to *unit_out when unit_out != NULL.
 *
 * Each returned factor is a genuine irreducible:
 *   - a prime element pi with N(pi) = p  (p a rational prime that splits or
 *     ramifies in O_K), or
 *   - a rational prime p that stays inert, with N(p) = p^2.
 *
 * Norm bound: this routine factors the rational integer N(alpha) by trial
 * division and so is intended for N(alpha) up to a few times 2^20 (the test
 * suite exercises up to 2^20).  alpha == 0 is rejected (returns -1); a unit
 * alpha yields 0 factors with *unit_out = alpha. */
int sp_ok_factor(sp_ok_t alpha, sp_ok_t *factors, int max, sp_ok_t *unit_out);

#ifdef __cplusplus
}
#endif

#endif /* SP_OK_INT_H */
