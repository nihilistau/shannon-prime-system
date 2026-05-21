/* sp_test.h — minimal, dependency-free unit-test harness for the
 * shannon-prime-system math core.
 *
 * Header-only. No external framework (keeps every backend portable and the
 * build network-free). Shared scaffold: include it, do not edit it inside a
 * module agent task.
 *
 * Usage:
 *
 *     #include "sp/sp_test.h"
 *
 *     static void T_OK_1(void) {
 *         SP_CHECK(norm_is_multiplicative(), "N(ab)=N(a)N(b)");
 *         SP_CHECK_EQ_I64(conj(conj(x)), x, "conj involutive");
 *     }
 *
 *     int main(void) {
 *         SP_RUN(T_OK_1);
 *         SP_RUN(T_OK_2);
 *         return SP_DONE();   // nonzero exit if any check failed
 *     }
 *
 * Each SP_RUN(fn) prints "[fn] PASS" or "[fn] FAIL" so ctest output names the
 * exact T_* identifier the roadmap contract lists. The single executable
 * returns 0 only if every check in every case passed.
 */
#ifndef SP_TEST_H
#define SP_TEST_H

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

static int sp__fails = 0;   /* total failed checks across all cases */
static int sp__checks = 0;  /* total checks attempted              */

#define SP_CHECK(cond, label)                                                 \
    do {                                                                      \
        sp__checks++;                                                         \
        if (!(cond)) {                                                        \
            sp__fails++;                                                      \
            fprintf(stderr, "    [FAIL] %s:%d  %s  (%s)\n",                    \
                    __FILE__, __LINE__, (label), #cond);                      \
        }                                                                     \
    } while (0)

#define SP_CHECK_EQ_I64(got, want, label)                                     \
    do {                                                                      \
        sp__checks++;                                                         \
        int64_t sp__g = (int64_t)(got);                                       \
        int64_t sp__w = (int64_t)(want);                                      \
        if (sp__g != sp__w) {                                                 \
            sp__fails++;                                                      \
            fprintf(stderr, "    [FAIL] %s:%d  %s  got=%" PRId64              \
                    " want=%" PRId64 "\n",                                    \
                    __FILE__, __LINE__, (label), sp__g, sp__w);               \
        }                                                                     \
    } while (0)

#define SP_RUN(fn)                                                            \
    do {                                                                      \
        int sp__before = sp__fails;                                           \
        fn();                                                                 \
        fprintf(stderr, "[%s] %s\n", #fn,                                     \
                (sp__fails == sp__before) ? "PASS" : "FAIL");                 \
    } while (0)

#define SP_DONE()                                                             \
    (fprintf(stderr, "---- checks=%d fails=%d ----\n", sp__checks, sp__fails),\
     (sp__fails == 0) ? 0 : 1)

#endif /* SP_TEST_H */
