/* sieve_test.c — M_POUW_1: Friedman Sieve correctness gate (Phase 5 PoUW).
 *
 * Fixture: deterministic KSTE sequences with known-dominating candidates.
 * The sieve must find them, emit events with the correct sig and seq_hash,
 * and maintain Pareto-frontier invariants throughout.
 */
#include "sp/sp_test.h"
#include "sp/sp_sieve.h"
#include "sp/kste.h"
#include "sp/sp_hash.h"
#include "sp/sp_status.h"

#include <stdint.h>
#include <string.h>

/* ---- helpers --------------------------------------------------------------- */

static int sig_eq(const sp_kste_tree_t *a, const sp_kste_tree_t *b) {
    return memcmp(a->bytes, b->bytes, 64) == 0;
}

/* ---- T_POUW_1: known-dominating sequence ----------------------------------- */

static void T_POUW_1(void) {
    /* vec_lo  = {1..8}   : root label (1,2,3,5,6,8) — the T_KSTE_4 anchor */
    /* vec_hi  = {11..18} : root label (11,12,13,15,16,18) — strictly > vec_lo */
    int32_t vec_lo[8] = { 1,  2,  3,  4,  5,  6,  7,  8 };
    int32_t vec_hi[8] = { 11, 12, 13, 14, 15, 16, 17, 18 };

    sp_kste_tree_t sig_lo, sig_hi;
    sp_kste_encode(vec_lo, 8, &sig_lo);
    sp_kste_encode(vec_hi, 8, &sig_hi);

    /* Tier-0 and Tier-1 of sig_hi vs sig_lo must both be SP_DOMINATES. */
    SP_CHECK(sp_kste_tier0(&sig_hi, &sig_lo) == SP_DOMINATES,
             "T_POUW_1 fixture: sig_hi tier0-dominates sig_lo");
    SP_CHECK(sp_kste_tier1(&sig_hi, &sig_lo) == SP_DOMINATES,
             "T_POUW_1 fixture: sig_hi tier1-dominates sig_lo");

    sp_kste_tree_t frontier[8];
    size_t         fn = 0;
    sp_sieve_event_t events[8];
    size_t           ne = 0;

    sp_kste_tree_t seq[2] = { sig_lo, sig_hi };

    sp_status rc = sp_sieve_evaluate(seq, 2, frontier, &fn, 8, events, &ne);
    SP_CHECK(rc == SP_OK,           "T_POUW_1 evaluate returns SP_OK");
    SP_CHECK_EQ_I64((int64_t)ne, 1, "T_POUW_1 exactly one sieve-fold event");
    SP_CHECK(sig_eq(&events[0].sig, &sig_hi),
             "T_POUW_1 event sig is sig_hi");
    SP_CHECK_EQ_I64((int64_t)events[0].round, 0,
                    "T_POUW_1 event round is 0");

    /* seq_hash must equal SHA-256(sig.bytes). */
    uint8_t expected_hash[32];
    sp_sha256(sig_hi.bytes, 64, expected_hash);
    SP_CHECK(memcmp(events[0].seq_hash, expected_hash, 32) == 0,
             "T_POUW_1 event seq_hash == SHA-256(sig.bytes)");

    /* Frontier must now contain only sig_hi (sig_lo was replaced). */
    SP_CHECK_EQ_I64((int64_t)fn, 1, "T_POUW_1 frontier has one member");
    SP_CHECK(sig_eq(&frontier[0], &sig_hi),
             "T_POUW_1 frontier[0] is sig_hi");
}

/* ---- T_POUW_2: dominated candidate discarded ------------------------------ */

static void T_POUW_2(void) {
    int32_t vec_lo[8] = { 1,  2,  3,  4,  5,  6,  7,  8 };
    int32_t vec_hi[8] = { 11, 12, 13, 14, 15, 16, 17, 18 };

    sp_kste_tree_t sig_lo, sig_hi;
    sp_kste_encode(vec_lo, 8, &sig_lo);
    sp_kste_encode(vec_hi, 8, &sig_hi);

    /* Frontier starts with sig_hi.  Process sig_lo (which is dominated). */
    sp_kste_tree_t frontier[8] = { sig_hi };
    size_t         fn = 1;
    sp_sieve_event_t events[8];
    size_t           ne = 0;

    sp_status rc = sp_sieve_evaluate(&sig_lo, 1, frontier, &fn, 8, events, &ne);
    SP_CHECK(rc == SP_OK,           "T_POUW_2 evaluate returns SP_OK");
    SP_CHECK_EQ_I64((int64_t)ne, 0, "T_POUW_2 no events for dominated candidate");
    SP_CHECK_EQ_I64((int64_t)fn, 1, "T_POUW_2 frontier unchanged (size 1)");
    SP_CHECK(sig_eq(&frontier[0], &sig_hi),
             "T_POUW_2 frontier still holds sig_hi");
}

/* ---- T_POUW_3: incomparable candidates extend frontier, no events --------- */

static void T_POUW_3(void) {
    /* vec_A: mostly 1s with one spike at 100.  Root = (1,1,1,1,1,100).
     * vec_B: mostly 50s with a low floor at 1.  Root = (1,50,50,50,50,50).
     * These are tier0-INCOMPARABLE (A has lower floor, B has lower peak),
     * so neither the absorbed_by nor strictly_dominates path fires. */
    int32_t vec_A[8] = { 1,  1,  1,  1,   1,   1,   1, 100 };
    int32_t vec_B[8] = { 1, 50, 50, 50,  50,  50,  50,  50 };

    sp_kste_tree_t sig_A, sig_B;
    sp_kste_encode(vec_A, 8, &sig_A);
    sp_kste_encode(vec_B, 8, &sig_B);

    SP_CHECK(sp_kste_tier0(&sig_A, &sig_B) == SP_INCOMPARABLE,
             "T_POUW_3 fixture: sig_A and sig_B are tier0-INCOMPARABLE");

    sp_kste_tree_t frontier[8];
    size_t         fn = 0;
    sp_sieve_event_t events[8];
    size_t           ne = 0;

    sp_kste_tree_t seq[2] = { sig_A, sig_B };

    sp_status rc = sp_sieve_evaluate(seq, 2, frontier, &fn, 8, events, &ne);
    SP_CHECK(rc == SP_OK,           "T_POUW_3 evaluate returns SP_OK");
    SP_CHECK_EQ_I64((int64_t)ne, 0, "T_POUW_3 no events for incomparable pair");
    SP_CHECK_EQ_I64((int64_t)fn, 2, "T_POUW_3 both candidates in frontier");
}

/* ---- T_POUW_4: SP_ESIEVE_FULL on frontier overflow ------------------------ */

static void T_POUW_4(void) {
    int32_t vec_A[8] = { 1,  1,  1,  1,  1,  1,  1, 100 };
    int32_t vec_B[8] = { 1, 50, 50, 50, 50, 50, 50,  50 };

    sp_kste_tree_t sig_A, sig_B;
    sp_kste_encode(vec_A, 8, &sig_A);
    sp_kste_encode(vec_B, 8, &sig_B);

    /* frontier_cap = 1.  sig_A fills it; sig_B (incomparable) cannot fit. */
    sp_kste_tree_t frontier[1];
    size_t         fn = 0;
    sp_sieve_event_t events[4];
    size_t           ne = 0;

    sp_kste_tree_t seq[2] = { sig_A, sig_B };

    sp_status rc = sp_sieve_evaluate(seq, 2, frontier, &fn, 1, events, &ne);
    SP_CHECK(rc == SP_ESIEVE_FULL,
             "T_POUW_4 returns SP_ESIEVE_FULL when frontier capacity exceeded");
    SP_CHECK_EQ_I64((int64_t)fn, 1,
                    "T_POUW_4 frontier holds first candidate before overflow");
}

/* ---- T_POUW_5: receipt layout constants ----------------------------------- */

static void T_POUW_5(void) {
    SP_CHECK_EQ_I64((int64_t)SP_RECEIPT_BYTES, 152,
                    "T_POUW_5 SP_RECEIPT_BYTES == 152");
    SP_CHECK_EQ_I64((int64_t)SP_RECEIPT_OFF_MAGIC,    0,
                    "T_POUW_5 offset MAGIC  ==   0");
    SP_CHECK_EQ_I64((int64_t)SP_RECEIPT_OFF_SIG,      8,
                    "T_POUW_5 offset SIG    ==   8");
    SP_CHECK_EQ_I64((int64_t)SP_RECEIPT_OFF_SEQ_HASH, 72,
                    "T_POUW_5 offset SEQ_HASH == 72");
    SP_CHECK_EQ_I64((int64_t)SP_RECEIPT_OFF_PUBKEY,   104,
                    "T_POUW_5 offset PUBKEY == 104");
    SP_CHECK_EQ_I64((int64_t)SP_RECEIPT_OFF_ROUND,    136,
                    "T_POUW_5 offset ROUND  == 136");
    SP_CHECK_EQ_I64((int64_t)SP_RECEIPT_OFF_MINTED,   144,
                    "T_POUW_5 offset MINTED == 144");
    /* layout sanity: MAGIC(8) + SIG(64) + SEQ_HASH(32) + PUBKEY(32) +
     * ROUND(8) + MINTED(8) = 152 */
    SP_CHECK_EQ_I64(
        (int64_t)(SP_RECEIPT_OFF_MINTED + 8),
        (int64_t)SP_RECEIPT_BYTES,
        "T_POUW_5 offsets sum to SP_RECEIPT_BYTES");
}

/* ---- T_POUW_6: multiple folds in one batch -------------------------------- */

static void T_POUW_6(void) {
    /* Three vectors in strict ascending order.  Processing all three in one
     * batch triggers two folds: {lo}→replaced by mid, {mid}→replaced by hi. */
    int32_t vec_lo[8]  = {  1,  2,  3,  4,  5,  6,  7,  8 };
    int32_t vec_mid[8] = {  5,  6,  7,  8,  9, 10, 11, 12 };
    int32_t vec_hi[8]  = { 11, 12, 13, 14, 15, 16, 17, 18 };

    sp_kste_tree_t sig_lo, sig_mid, sig_hi;
    sp_kste_encode(vec_lo,  8, &sig_lo);
    sp_kste_encode(vec_mid, 8, &sig_mid);
    sp_kste_encode(vec_hi,  8, &sig_hi);

    /* Verify chain: mid > lo, hi > mid. */
    SP_CHECK(sp_kste_tier0(&sig_mid, &sig_lo) == SP_DOMINATES,
             "T_POUW_6 fixture: sig_mid > sig_lo at tier0");
    SP_CHECK(sp_kste_tier0(&sig_hi, &sig_mid) == SP_DOMINATES,
             "T_POUW_6 fixture: sig_hi > sig_mid at tier0");

    sp_kste_tree_t frontier[8];
    size_t         fn = 0;
    sp_sieve_event_t events[8];
    size_t           ne = 0;

    sp_kste_tree_t seq[3] = { sig_lo, sig_mid, sig_hi };

    sp_status rc = sp_sieve_evaluate(seq, 3, frontier, &fn, 8, events, &ne);
    SP_CHECK(rc == SP_OK,           "T_POUW_6 evaluate returns SP_OK");
    SP_CHECK_EQ_I64((int64_t)ne, 2, "T_POUW_6 two sieve-fold events");
    SP_CHECK(sig_eq(&events[0].sig, &sig_mid),
             "T_POUW_6 first event sig is sig_mid");
    SP_CHECK(sig_eq(&events[1].sig, &sig_hi),
             "T_POUW_6 second event sig is sig_hi");
    SP_CHECK_EQ_I64((int64_t)events[0].round, 0,
                    "T_POUW_6 first event round == 0");
    SP_CHECK_EQ_I64((int64_t)events[1].round, 1,
                    "T_POUW_6 second event round == 1");
    SP_CHECK_EQ_I64((int64_t)fn, 1,
                    "T_POUW_6 frontier has one member at end");
    SP_CHECK(sig_eq(&frontier[0], &sig_hi),
             "T_POUW_6 final frontier member is sig_hi");
}

/* ---- main ----------------------------------------------------------------- */

int main(void) {
    SP_RUN(T_POUW_1);
    SP_RUN(T_POUW_2);
    SP_RUN(T_POUW_3);
    SP_RUN(T_POUW_4);
    SP_RUN(T_POUW_5);
    SP_RUN(T_POUW_6);
    return SP_DONE();
}
