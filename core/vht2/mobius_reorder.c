/* mobius_reorder.c — the fixed Mobius coefficient permutation (v1).
 *
 * sp_mobius(i)     = (SP_MOBIUS_A * i) mod n
 * sp_mobius_inv(j) = (a_inv * j)       mod n,   a_inv * SP_MOBIUS_A == 1 (mod n)
 *
 * SP_MOBIUS_A = 17 is coprime to 55 (=5*11) and to every body size used, so the
 * map is a bijection on Z/nZ for any n >= 1. a_inv is recomputed per n with the
 * extended Euclidean algorithm; for n == 1 the (degenerate) map is the identity.
 *
 * Pure index permutation — no dependence on the data being permuted. Frozen v1:
 * changing SP_MOBIUS_A or the affine form requires bumping the layout version.
 */
#include "sp/spinor_block.h"

/* Non-negative reduction of a possibly-negative value mod n (n >= 1). */
static int mod_pos(long long x, int n) {
    long long r = x % n;
    if (r < 0) r += n;
    return (int)r;
}

/* Modular inverse of a mod n via extended Euclid. Requires gcd(a,n)==1.
 * For n == 1 every element is 0 and the inverse is 0 (identity map). */
static int mod_inverse(int a, int n) {
    if (n == 1) return 0;
    int t = 0, newt = 1;
    int r = n, newr = mod_pos(a, n);
    while (newr != 0) {
        int q = r / newr;
        int tmp;
        tmp = t - q * newt; t = newt; newt = tmp;
        tmp = r - q * newr; r = newr; newr = tmp;
    }
    /* r == gcd(a,n); assumed 1 by choice of SP_MOBIUS_A. */
    if (t < 0) t += n;
    return t;
}

void sp_mobius_reorder(const int32_t *in, int32_t *out, int n) {
    for (int i = 0; i < n; i++) {
        int j = mod_pos((long long)SP_MOBIUS_A * i, n);
        out[j] = in[i];
    }
}

void sp_mobius_reorder_inv(const int32_t *in, int32_t *out, int n) {
    int a_inv = mod_inverse(SP_MOBIUS_A, n);
    /* inverse of "out[A*i] = in[i]" is "back[i] = out[A*i]", i.e. for each
     * source index i recover the slot it was written to. Equivalently, using
     * a_inv: for each permuted slot j, its original index is a_inv*j mod n. */
    for (int j = 0; j < n; j++) {
        int i = mod_pos((long long)a_inv * j, n);
        out[i] = in[j];
    }
}
