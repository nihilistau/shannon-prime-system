/* sp_channel_internal.h — shared internals for sp_channel_map.c and
 * sp_channel_probe.c.  Not part of the public include surface. */
#ifndef SP_CHANNEL_INTERNAL_H
#define SP_CHANNEL_INTERNAL_H

#include "sp/sp_channel.h"
#include <stdint.h>
#include <stddef.h>

/* Address-bit probe range.
 *   [CHAN_BIT_LO, CHAN_BIT_MID) — virtual-address arithmetic only; no pagemap.
 *   [CHAN_BIT_MID, CHAN_BIT_HI) — require /proc/self/pagemap + CAP_SYS_ADMIN.
 */
#define CHAN_BIT_LO   12
#define CHAN_BIT_MID  21
#define CHAN_BIT_HI   24
#define CHAN_N_VIRT   (CHAN_BIT_MID - CHAN_BIT_LO)   /*  9 virtual-only bits */
#define CHAN_N_PHYS   (CHAN_BIT_HI  - CHAN_BIT_LO)   /* 12 bits total        */
#define CHAN_K_MAX     4   /* max channel-select bits (supports up to 16 channels) */

struct sp_channel_map {
    sp_channel_mode mode;
    uint32_t k;       /* rows of M: number of channel-select bits */
    uint32_t n;       /* cols of M: address bits probed            */
    uint32_t n_start; /* index of first probed bit (CHAN_BIT_LO)   */
    /* M[row][col] = 1 iff address bit (n_start + col) contributes to channel row */
    uint8_t M[CHAN_K_MAX][CHAN_N_PHYS];
};

/* Timing result for one address-bit position. */
typedef struct {
    int      bit;             /* absolute bit index (e.g. 12, 13, …) */
    int      is_same_channel; /* 1 = same channel (high-P99 tail), 0 = different */
    uint64_t p99_ns;          /* P99 hedge-read latency in nanoseconds */
} sp_probe_result;

/* sp_channel_probe.c: time a single bit-flip pair.
 * base_addr must be a huge-page-aligned allocation base of ≥ 4 * huge_page_size.
 * n_probes: number of timing samples to collect.
 * Returns 0 on success, non-zero on threading / allocation failure (→ DISABLED). */
int sp_probe_bit(uintptr_t base_addr, int bit, size_t huge_page_size,
                 int n_probes, sp_probe_result *result_out);

/* Allocate n_pages of huge pages (MAP_HUGETLB / MEM_LARGE_PAGES).
 * Returns NULL on failure (VM / privilege denied). */
void *sp_alloc_huge(size_t n_pages, size_t page_size);
void  sp_free_huge(void *ptr, size_t n_pages, size_t page_size);

/* Returns 1 if running under a hypervisor / in a container that prevents
 * accurate DRAM timing.  Also respects SP_FORCE_VIRT_DETECTION=1 for tests. */
int sp_detect_virtualisation(void);

/* Returns 1 if /proc/self/pagemap resolves physical frame numbers (needs
 * CAP_SYS_ADMIN on Linux ≥ 4.0).  Always 0 on Windows. */
int sp_pagemap_privileged(void);

/* Opaque handle for sp_alloc_channel_pair / sp_free_channel_pair.
 * Defined here so bench_ts_hedge.c (internal) can inspect it if needed. */
struct sp_channel_pair_arena {
    void   *base;       /* huge-page arena (is_huge=1) or calloc block (is_huge=0) */
    size_t  n_pages;    /* non-zero only on huge-page path */
    size_t  page_size;  /* non-zero only on huge-page path */
    int     is_huge;
};

#endif /* SP_CHANNEL_INTERNAL_H */
