/* sp_channel_test.c — Phase TS.MAP unit tests.
 *
 * T_CHANNEL_BUILD_VIRT:    SP_FORCE_VIRT_DETECTION=1 → SP_OK + DISABLED
 * T_CHANNEL_OF_DISABLED:   sp_channel_of on DISABLED → SP_CHANNEL_UNSPECIFIED
 * T_CHANNEL_CACHE_RT:      save → load round-trip; miss → SP_OK + NULL
 * T_CHANNEL_BUILD_BARE:    sp_channel_map_build in current env → always SP_OK
 * T_CHANNEL_HEDGE_BENCH:   no crash on probe path (DISABLED graceful in CI)
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp/sp_channel.h"
#include "sp/sp_status.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Portable setenv / unsetenv ─────────────────────────────────────────── */
#ifdef _WIN32
static void tenv_set(const char *name, const char *val) { _putenv_s(name, val); }
static void tenv_unset(const char *name)                { _putenv_s(name, ""); }
#else
static void tenv_set(const char *name, const char *val) { setenv(name, val, 1); }
static void tenv_unset(const char *name)                { unsetenv(name); }
#endif

/* ── T_CHANNEL_BUILD_VIRT ────────────────────────────────────────────────── */

static int T_CHANNEL_BUILD_VIRT_1(void) {
    tenv_set("SP_FORCE_VIRT_DETECTION", "1");
    sp_channel_map *m = NULL;
    sp_status rc = sp_channel_map_build(&m);
    tenv_unset("SP_FORCE_VIRT_DETECTION");

    SP_CHECK(rc == SP_OK,                          "forced-virt: build returns SP_OK");
    SP_CHECK(m  != NULL,                           "forced-virt: map non-NULL");
    SP_CHECK(sp_channel_map_mode(m) == SP_CHANNEL_DISABLED,
             "forced-virt: mode == DISABLED");
    sp_channel_map_free(m);
    return 0;
}

/* ── T_CHANNEL_OF_DISABLED ───────────────────────────────────────────────── */

static int T_CHANNEL_OF_DISABLED_1(void) {
    /* Build a DISABLED map via forced detection */
    tenv_set("SP_FORCE_VIRT_DETECTION", "1");
    sp_channel_map *m = NULL;
    sp_status rc = sp_channel_map_build(&m);
    tenv_unset("SP_FORCE_VIRT_DETECTION");
    SP_CHECK(rc == SP_OK && m != NULL, "setup: DISABLED map");

    uint32_t ch = sp_channel_of(m, (uintptr_t)0xDEADBEEFul);
    SP_CHECK(ch == SP_CHANNEL_UNSPECIFIED, "DISABLED map → UNSPECIFIED");
    sp_channel_map_free(m);

    /* NULL handle must also return UNSPECIFIED */
    ch = sp_channel_of(NULL, 0);
    SP_CHECK(ch == SP_CHANNEL_UNSPECIFIED, "NULL map → UNSPECIFIED");
    return 0;
}

/* ── T_CHANNEL_CACHE_RT ──────────────────────────────────────────────────── */

static int T_CHANNEL_CACHE_RT_1(void) {
    /* Save a DISABLED map under a test-only fingerprint */
    tenv_set("SP_FORCE_VIRT_DETECTION", "1");
    sp_channel_map *m = NULL;
    sp_status rc = sp_channel_map_build(&m);
    tenv_unset("SP_FORCE_VIRT_DETECTION");
    SP_CHECK(rc == SP_OK && m != NULL, "setup: DISABLED map for cache test");

    const char *fp = "test_rt_00000000";
    rc = sp_channel_map_save_cached(m, fp);
    SP_CHECK(rc == SP_OK, "save_cached returns SP_OK");
    sp_channel_map_free(m);

    /* Load it back */
    sp_channel_map *m2 = NULL;
    rc = sp_channel_map_load_cached(fp, &m2);
    SP_CHECK(rc == SP_OK,  "load_cached returns SP_OK on hit");
    SP_CHECK(m2 != NULL,   "load_cached returns non-NULL on hit");
    SP_CHECK(sp_channel_map_mode(m2) == SP_CHANNEL_DISABLED,
             "round-trip preserves DISABLED mode");
    sp_channel_map_free(m2);

    /* Cache miss: absent fingerprint → SP_OK + NULL */
    sp_channel_map *m3 = NULL;
    rc = sp_channel_map_load_cached("no_such_fp_zzz_xxyy", &m3);
    SP_CHECK(rc == SP_OK, "cache miss returns SP_OK");
    SP_CHECK(m3 == NULL,  "cache miss returns NULL map");
    return 0;
}

/* ── T_CHANNEL_BUILD_BARE ────────────────────────────────────────────────── */

static int T_CHANNEL_BUILD_BARE_1(void) {
    /* In any environment — VM or bare metal — build must return SP_OK */
    sp_channel_map *m = NULL;
    sp_status rc = sp_channel_map_build(&m);
    SP_CHECK(rc == SP_OK, "build always returns SP_OK");
    SP_CHECK(m  != NULL,  "build always returns non-NULL map");

    sp_channel_mode mode = sp_channel_map_mode(m);
    SP_CHECK(mode == SP_CHANNEL_DISABLED || mode == SP_CHANNEL_LIVE,
             "mode is DISABLED or LIVE");

    if (mode == SP_CHANNEL_LIVE) {
        /* On bare metal: sp_channel_of must return a non-UNSPECIFIED value */
        uint32_t ch = sp_channel_of(m, (uintptr_t)m);
        SP_CHECK(ch != SP_CHANNEL_UNSPECIFIED, "LIVE map returns valid channel index");

        uint32_t k = 0, n = 0;
        rc = sp_channel_map_dims(m, &k, &n);
        SP_CHECK(rc == SP_OK,     "dims returns SP_OK on LIVE map");
        SP_CHECK(k >= 1 && k <= 4, "k in [1,4]");
        SP_CHECK(n >= 1,           "n >= 1");
        fprintf(stderr, "  [bare-metal: k=%u n=%u ch(self)=%u]\n", k, n, ch);
    } else {
        fprintf(stderr, "  [CI/VM environment: DISABLED — graceful fallback OK]\n");
    }
    sp_channel_map_free(m);
    return 0;
}

/* ── T_CHANNEL_HEDGE_BENCH ───────────────────────────────────────────────── */

static int T_CHANNEL_HEDGE_BENCH_1(void) {
    /* Smoke test: the build path must not crash in any environment.
     * In CI/VM it returns DISABLED immediately; on bare metal it runs probes.
     * Either way, the outcome must be SP_OK. */
    sp_channel_map *m = NULL;
    sp_status rc = sp_channel_map_build(&m);
    SP_CHECK(rc == SP_OK, "hedge bench: build completes without crash");
    SP_CHECK(m  != NULL,  "hedge bench: non-NULL map");

    sp_channel_mode mode = sp_channel_map_mode(m);
    if (mode == SP_CHANNEL_LIVE) {
        /* Spot-check: consecutive calls to sp_channel_of are deterministic */
        uintptr_t addr = (uintptr_t)0x200000ul;
        uint32_t ch_a  = sp_channel_of(m, addr);
        uint32_t ch_b  = sp_channel_of(m, addr);
        SP_CHECK(ch_a == ch_b, "sp_channel_of is deterministic");
        fprintf(stderr, "  [hedge bench: channel(%p)=%u]\n", (void *)addr, ch_a);
    }
    sp_channel_map_free(m);
    return 0;
}

/* ── Test runner ─────────────────────────────────────────────────────────── */

int main(void) {
    SP_RUN(T_CHANNEL_BUILD_VIRT_1);
    SP_RUN(T_CHANNEL_OF_DISABLED_1);
    SP_RUN(T_CHANNEL_CACHE_RT_1);
    SP_RUN(T_CHANNEL_BUILD_BARE_1);
    SP_RUN(T_CHANNEL_HEDGE_BENCH_1);
    SP_DONE();
}
