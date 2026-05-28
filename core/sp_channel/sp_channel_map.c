/* sp_channel_map.c — GF(2) channel-select oracle: virt detection, matrix
 * recovery from probe results, cache I/O, and the public API.
 *
 * SAFETY CONTRACT: every non-SP_ENOMEM path through sp_channel_map_build must
 * return SP_OK with mode = SP_CHANNEL_DISABLED rather than a non-OK status.
 * Crashing or returning SP_EINVAL / SP_EIO in the detection path would break
 * CI.  "Successful detection of impossibility" is the correct framing. */
#define _CRT_SECURE_NO_WARNINGS   /* getenv/snprintf on MSVC */

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

/* CPUID intrinsics: GCC/Clang (including MinGW) use <cpuid.h>; MSVC uses <intrin.h> */
#if defined(__x86_64__) || defined(__i386__)
#  if defined(__GNUC__) || defined(__clang__)
#    include <cpuid.h>   /* __get_cpuid, __get_cpuid_max */
#  elif defined(_MSC_VER)
#    include <intrin.h>  /* __cpuid */
#  endif
#endif

#ifndef _WIN32
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

#include "sp/sp_channel.h"
#include "sp/sp_error.h"
#include "sp_channel_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ── CPUID helpers (x86 only) ────────────────────────────────────────────── */

#if defined(__x86_64__) || defined(__i386__)
static unsigned int x86_cpuid_ecx(unsigned int leaf) {
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
#if defined(__GNUC__) || defined(__clang__)
    __get_cpuid(leaf, &eax, &ebx, &ecx, &edx);
#elif defined(_MSC_VER)
    int info[4] = {0};
    __cpuid(info, (int)leaf);
    ecx = (unsigned int)info[2];
#endif
    (void)eax; (void)ebx; (void)edx;
    return ecx;
}

/* __get_cpuid refuses hypervisor leaves (0x4000xxxx) because they exceed the
 * normal max leaf.  Use __cpuid_count / __cpuid directly. */
static void x86_cpuid_raw(unsigned int leaf,
                           unsigned int *a, unsigned int *b,
                           unsigned int *c, unsigned int *d) {
    *a = *b = *c = *d = 0;
#if defined(__GNUC__) || defined(__clang__)
    __cpuid_count(leaf, 0, *a, *b, *c, *d);
#elif defined(_MSC_VER)
    int info[4] = {0};
    __cpuid(info, (int)leaf);
    *a = (unsigned int)info[0]; *b = (unsigned int)info[1];
    *c = (unsigned int)info[2]; *d = (unsigned int)info[3];
#endif
}
#endif /* x86 */

/* On Windows 11, Hyper-V/VBS sets CPUID bit 31 even on the root partition
 * (bare metal). The KVP registry key is written by the Hyper-V integration
 * service only inside guest VMs — absent on the host/root partition. */
#ifdef _WIN32
static int is_hyperv_guest(void) {
    HKEY hk;
    LONG rc = RegOpenKeyExA(HKEY_LOCAL_MACHINE,
        "SOFTWARE\\Microsoft\\Virtual Machine\\Guest\\Parameters",
        0, KEY_READ, &hk);
    if (rc == ERROR_SUCCESS) { RegCloseKey(hk); return 1; }
    return 0;
}
#endif

#ifndef _WIN32
static int cpuinfo_has_hypervisor(void) {
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return 0;
    char line[256];
    int found = 0;
    while (!found && fgets(line, sizeof line, f)) {
        /* Check the "flags" line for the "hypervisor" token */
        if (strncmp(line, "flags", 5) == 0 && strstr(line, "hypervisor"))
            found = 1;
        /* Vendor / product strings from various hypervisors */
        if (strstr(line, "KVMKVM") || strstr(line, "VMware") ||
            strstr(line, "VirtualBox") || strstr(line, "Microsoft Hv") ||
            strstr(line, "Xen HVM"))
            found = 1;
    }
    fclose(f);
    return found;
}
#endif

int sp_detect_virtualisation(void) {
    /* Test hook: SP_FORCE_VIRT_DETECTION=1 forces DISABLED */
    const char *force = getenv("SP_FORCE_VIRT_DETECTION");
    if (force && force[0] == '1') return 1;

    /* CPUID leaf 1 ECX bit 31: hypervisor present (x86 only) */
#if defined(__x86_64__) || defined(__i386__)
    if (x86_cpuid_ecx(1u) & (1u << 31)) {
#ifdef _WIN32
        /* Windows 11 VBS/Hyper-V sets this bit on the root partition (bare
         * metal) as well as in guests. Check the hypervisor vendor first. */
        unsigned int hv_a, hv_b, hv_c, hv_d;
        x86_cpuid_raw(0x40000000u, &hv_a, &hv_b, &hv_c, &hv_d);
        char vendor[13] = {0};
        memcpy(vendor,     &hv_b, 4);
        memcpy(vendor + 4, &hv_c, 4);
        memcpy(vendor + 8, &hv_d, 4);
        if (memcmp(vendor, "Microsoft Hv", 12) == 0) {
            /* Hyper-V: KVP key present → guest; absent → root partition */
            if (is_hyperv_guest()) return 1;
            /* Root partition with VBS — not a VM; huge-page alloc decides */
        } else {
            return 1;  /* VMware / KVM / VirtualBox / etc. */
        }
#else
        return 1;
#endif
    }
#endif

#ifdef _WIN32
    return 0;
#else
    if (cpuinfo_has_hypervisor()) return 1;
    /* /sys/hypervisor/type present on Xen PV */
    struct stat st;
    if (stat("/sys/hypervisor/type", &st) == 0) return 1;
    return 0;
#endif
}

int sp_pagemap_privileged(void) {
#ifdef _WIN32
    return 0;  /* no pagemap on Windows */
#else
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) return 0;
    /* Read the PFN entry for a known-mapped stack address.
     * Since Linux 4.0, unprivileged reads return a zeroed PFN field (bits 0–54).
     * Bit 63 = present; bits 0–54 = PFN. */
    uint64_t entry = 0;
    uintptr_t va = (uintptr_t)&entry;
    off_t off = (off_t)((va / 4096u) * 8u);
    int privileged = 0;
    if (lseek(fd, off, SEEK_SET) != (off_t)-1) {
        uint64_t pfn_entry = 0;
        if (read(fd, &pfn_entry, 8) == 8)
            privileged = ((pfn_entry & 0x7FFFFFFFFFFFFFFFull) != 0) ? 1 : 0;
    }
    close(fd);
    return privileged;
#endif
}

/* ── Huge-page allocation ─────────────────────────────────────────────────── */

static size_t detect_huge_page_size(void) {
#ifdef _WIN32
    SIZE_T sz = GetLargePageMinimum();
    return (sz == 0) ? (2u * 1024u * 1024u) : (size_t)sz;
#else
    FILE *f = fopen("/proc/meminfo", "r");
    size_t sz = 2u * 1024u * 1024u;  /* default: 2 MB */
    if (f) {
        char line[128];
        while (fgets(line, sizeof line, f)) {
            if (strncmp(line, "Hugepagesize:", 13) == 0) {
                unsigned long kb = 0;
                if (sscanf(line + 13, "%lu kB", &kb) == 1 && kb > 0)
                    sz = (size_t)kb * 1024u;
                break;
            }
        }
        fclose(f);
    }
    return sz;
#endif
}

#ifdef _WIN32
#include <windows.h>
/* Returns 1 if SeLockMemoryPrivilege was successfully enabled, 0 otherwise.
 * AdjustTokenPrivileges succeeds (returns TRUE) even when the privilege is
 * absent from the token — GetLastError() == ERROR_NOT_ALL_ASSIGNED in that
 * case.  We check for that to surface actionable diagnostics. */
static int force_enable_large_pages(void) {
    HANDLE hToken;
    int ok = 0;
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken)) {
        TOKEN_PRIVILEGES tp;
        if (LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &tp.Privileges[0].Luid)) {
            tp.PrivilegeCount = 1;
            tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
            AdjustTokenPrivileges(hToken, FALSE, &tp, 0, NULL, 0);
            ok = (GetLastError() != ERROR_NOT_ALL_ASSIGNED);
        }
        CloseHandle(hToken);
    }
    return ok;
}
#endif

void *sp_alloc_huge(size_t n_pages, size_t page_size) {
    size_t sz = n_pages * page_size;
#ifdef _WIN32
    if (!force_enable_large_pages()) {
        fprintf(stderr,
            "SP_WARN: sp_channel: SeLockMemoryPrivilege not in token — "
            "re-run as Administrator (right-click → Run as Administrator)\n");
        return NULL;
    }
    void *ptr = VirtualAlloc(NULL, (SIZE_T)sz,
                             MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
                             PAGE_READWRITE);
    if (!ptr)
        fprintf(stderr,
            "SP_WARN: sp_channel: VirtualAlloc(MEM_LARGE_PAGES, %zu) failed — "
            "error %lu\n", sz, (unsigned long)GetLastError());
    return ptr;
#else
    void *ptr = mmap(NULL, sz, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    return (ptr == MAP_FAILED) ? NULL : ptr;
#endif
}

void sp_free_huge(void *ptr, size_t n_pages, size_t page_size) {
    if (!ptr) return;
#ifdef _WIN32
    (void)n_pages; (void)page_size;
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    munmap(ptr, n_pages * page_size);
#endif
}

/* ── GF(2) column recovery ────────────────────────────────────────────────── */
/* For each bit position i, probe_result[i].is_same_channel = 0 means flipping
 * bit (n_start + i) changes the channel → that bit is a channel-select pivot.
 * We collect the pivot columns and form the k×n matrix M. */
static uint32_t gf2_recover_M(const sp_probe_result *results, int n,
                               uint32_t n_start, struct sp_channel_map *map) {
    int pivots[CHAN_K_MAX];
    int k = 0;
    for (int i = 0; i < n && k < CHAN_K_MAX; i++) {
        if (!results[i].is_same_channel)
            pivots[k++] = i;
    }
    if (k == 0) return 0;

    map->k      = (uint32_t)k;
    map->n      = (uint32_t)n;
    map->n_start = n_start;
    memset(map->M, 0, sizeof map->M);
    for (int j = 0; j < k; j++)
        map->M[j][pivots[j]] = 1;
    return (uint32_t)k;
}

/* ── Cache file ───────────────────────────────────────────────────────────── */

#define CACHE_MAGIC   "SPCH"
#define CACHE_VERSION 1u

static uint32_t cache_checksum(const struct sp_channel_map *m,
                                const char *fp) {
    /* DJB2 over struct fields + fingerprint string */
    uint32_t h = 5381u;
#define HASH_B(PTR, LEN) \
    do { const uint8_t *_p = (const uint8_t *)(PTR); \
         for (size_t _i = 0; _i < (size_t)(LEN); _i++) \
             h = ((h << 5) + h) ^ _p[_i]; } while (0)
    HASH_B(&m->mode,    sizeof m->mode);
    HASH_B(&m->k,       sizeof m->k);
    HASH_B(&m->n,       sizeof m->n);
    HASH_B(&m->n_start, sizeof m->n_start);
    HASH_B(m->M,        sizeof m->M);
    HASH_B(fp,          strlen(fp));
#undef HASH_B
    return h;
}

static void cache_path(char *buf, size_t bufsz, const char *fp) {
#ifdef _WIN32
    const char *base = getenv("LOCALAPPDATA");
    if (!base || !base[0]) base = ".";
    snprintf(buf, bufsz, "%s\\shannon-prime\\channel_map_%s.bin", base, fp);
#else
    const char *home = getenv("HOME");
    if (!home || !home[0]) home = ".";
    snprintf(buf, bufsz, "%s/.cache/shannon-prime/channel_map_%s.bin", home, fp);
#endif
}

static void ensure_dir(const char *file_path) {
    char dir[512];
    snprintf(dir, sizeof dir, "%s", file_path);
    /* Trim to the parent directory */
    for (int i = (int)strlen(dir) - 1; i >= 0; i--) {
        if (dir[i] == '/' || dir[i] == '\\') { dir[i] = '\0'; break; }
    }
#ifdef _WIN32
    CreateDirectoryA(dir, NULL);
#else
    mkdir(dir, 0755);
#endif
}

sp_status sp_channel_map_load_cached(const char *host_fingerprint,
                                     sp_channel_map **out) {
    if (!host_fingerprint || !out) {
        sp_set_error("sp_channel_map_load_cached: NULL arg");
        return SP_EBADARG;
    }
    *out = NULL;

    char path[512];
    cache_path(path, sizeof path, host_fingerprint);

    FILE *f = fopen(path, "rb");
    if (!f) return SP_OK;  /* cache miss: absent file is not an error */

    char magic[4];
    uint32_t version = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, CACHE_MAGIC, 4) != 0 ||
        fread(&version, 4, 1, f) != 1 || version != CACHE_VERSION) {
        fclose(f);
        sp_set_error("channel map cache: bad magic or version");
        return SP_EIO;
    }

    struct sp_channel_map *m = (struct sp_channel_map *)calloc(1, sizeof *m);
    if (!m) { fclose(f); return SP_ENOMEM; }

    uint32_t mode_wire = 0;
    int ok = 1;
    ok &= (int)(fread(&mode_wire,  4, 1, f) == 1);
    ok &= (int)(fread(&m->k,       4, 1, f) == 1);
    ok &= (int)(fread(&m->n,       4, 1, f) == 1);
    ok &= (int)(fread(&m->n_start, 4, 1, f) == 1);
    ok &= (int)(fread(m->M, 1, sizeof m->M, f) == sizeof m->M);
    uint32_t stored_csum = 0;
    ok &= (int)(fread(&stored_csum, 4, 1, f) == 1);
    fclose(f);

    if (!ok) {
        free(m); sp_set_error("channel map cache: truncated"); return SP_EIO;
    }
    if (m->k > CHAN_K_MAX || m->n > CHAN_N_PHYS) {
        free(m); sp_set_error("channel map cache: out-of-range dims"); return SP_EIO;
    }
    m->mode = (sp_channel_mode)mode_wire;
    if (cache_checksum(m, host_fingerprint) != stored_csum) {
        free(m); sp_set_error("channel map cache: checksum mismatch"); return SP_EIO;
    }

    *out = m;
    return SP_OK;
}

sp_status sp_channel_map_save_cached(const sp_channel_map *m,
                                     const char *host_fingerprint) {
    if (!m || !host_fingerprint) {
        sp_set_error("sp_channel_map_save_cached: NULL arg");
        return SP_EBADARG;
    }
    char path[512];
    cache_path(path, sizeof path, host_fingerprint);
    ensure_dir(path);

    FILE *f = fopen(path, "wb");
    if (!f) {
        sp_set_error("channel map cache: cannot open for write");
        return SP_EIO;
    }
    uint32_t version   = CACHE_VERSION;
    uint32_t mode_wire = (uint32_t)m->mode;
    uint32_t csum      = cache_checksum(m, host_fingerprint);

    int ok = 1;
    ok &= (int)(fwrite(CACHE_MAGIC,  1, 4, f) == 4);
    ok &= (int)(fwrite(&version,     4, 1, f) == 1);
    ok &= (int)(fwrite(&mode_wire,   4, 1, f) == 1);
    ok &= (int)(fwrite(&m->k,        4, 1, f) == 1);
    ok &= (int)(fwrite(&m->n,        4, 1, f) == 1);
    ok &= (int)(fwrite(&m->n_start,  4, 1, f) == 1);
    ok &= (int)(fwrite(m->M, 1, sizeof m->M, f) == sizeof m->M);
    ok &= (int)(fwrite(&csum,        4, 1, f) == 1);
    fclose(f);

    if (!ok) {
        sp_set_error("channel map cache: write error");
        return SP_EIO;
    }
    return SP_OK;
}

/* ── Host fingerprint ─────────────────────────────────────────────────────── */

sp_status sp_channel_host_fingerprint(char *buf, size_t buf_len) {
    if (!buf || buf_len < 16) {
        sp_set_error("sp_channel_host_fingerprint: buffer too small (need >= 16)");
        return SP_EBADARG;
    }
    uint32_t h = 5381u;
#define HASH_B(PTR, LEN) \
    do { const uint8_t *_p = (const uint8_t *)(PTR); \
         for (size_t _i = 0; _i < (size_t)(LEN); _i++) \
             h = ((h << 5) + h) ^ _p[_i]; } while (0)

#if defined(__x86_64__) || defined(__i386__)
    /* CPU brand string: CPUID leaves 0x80000002–0x80000004 */
    {
        char brand[49] = {0};
        for (unsigned int leaf = 0; leaf < 3u; leaf++) {
            unsigned int a = 0, b = 0, c = 0, d = 0;
#if defined(__GNUC__) || defined(__clang__)
            __get_cpuid(0x80000002u + leaf, &a, &b, &c, &d);
#elif defined(_MSC_VER)
            int info[4] = {0};
            __cpuid(info, (int)(0x80000002u + leaf));
            a = (unsigned int)info[0]; b = (unsigned int)info[1];
            c = (unsigned int)info[2]; d = (unsigned int)info[3];
#endif
            memcpy(brand + leaf * 16u,      &a, 4);
            memcpy(brand + leaf * 16u + 4u, &b, 4);
            memcpy(brand + leaf * 16u + 8u, &c, 4);
            memcpy(brand + leaf * 16u + 12u,&d, 4);
        }
        HASH_B(brand, strlen(brand));
    }
#endif /* x86 */
#ifdef _WIN32
    {
        MEMORYSTATUSEX ms;
        ms.dwLength = sizeof ms;
        if (GlobalMemoryStatusEx(&ms)) {
            uint64_t total = (uint64_t)ms.ullTotalPhys;
            HASH_B(&total, sizeof total);
        }
    }
#else
    {
        FILE *f = fopen("/proc/cpuinfo", "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof line, f)) {
                if (strncmp(line, "model name", 10) == 0) {
                    HASH_B(line, strlen(line)); break;
                }
            }
            fclose(f);
        }
    }
    {
        FILE *f = fopen("/proc/meminfo", "r");
        if (f) {
            char line[128];
            while (fgets(line, sizeof line, f)) {
                if (strncmp(line, "MemTotal:", 9) == 0) {
                    HASH_B(line, strlen(line)); break;
                }
            }
            fclose(f);
        }
    }
#endif
#undef HASH_B
    snprintf(buf, buf_len, "%08x", h);
    return SP_OK;
}

/* ── Public API ───────────────────────────────────────────────────────────── */

sp_status sp_channel_map_build(sp_channel_map **out) {
    if (!out) { sp_set_error("sp_channel_map_build: NULL out"); return SP_EBADARG; }
    *out = NULL;

    struct sp_channel_map *m = (struct sp_channel_map *)calloc(1, sizeof *m);
    if (!m) return SP_ENOMEM;

    /* Gate 1: hypervisor / container detection */
    if (sp_detect_virtualisation()) {
        fprintf(stderr, "SP_INFO: sp_channel: VM/hypervisor detected — DISABLED\n");
        m->mode = SP_CHANNEL_DISABLED;
        *out = m;
        return SP_OK;
    }

    /* Gate 2: huge-page allocation */
    size_t hp_size = detect_huge_page_size();
    void *hp_base  = sp_alloc_huge(4, hp_size);
    if (!hp_base) {
        fprintf(stderr, "SP_INFO: sp_channel: huge-page allocation failed — DISABLED\n");
        m->mode = SP_CHANNEL_DISABLED;
        *out = m;
        return SP_OK;
    }

    /* Probe range: virtual-only [12,21) by default; extend to [21,24) if
     * pagemap resolves PFNs (needs CAP_SYS_ADMIN on Linux >= 4.0). */
    int n_bits = CHAN_N_VIRT;
    if (sp_pagemap_privileged()) {
        n_bits = CHAN_N_PHYS;
        fprintf(stderr, "SP_INFO: sp_channel: pagemap privileged, probing bits [%d,%d)\n",
                CHAN_BIT_LO, CHAN_BIT_LO + n_bits);
    } else {
        fprintf(stderr, "SP_INFO: sp_channel: limited recovery, probing bits [%d,%d)\n",
                CHAN_BIT_LO, CHAN_BIT_LO + n_bits);
    }

    /* Probe each bit position */
    sp_probe_result results[CHAN_N_PHYS];
    memset(results, 0, sizeof results);
    int n_probes = 512;
    int probe_ok = 1;

    probe_pool *pool = sp_probe_pool_create();
    if (!pool) {
        fprintf(stderr, "SP_WARN: sp_channel: probe pool init failed — DISABLED\n");
        sp_free_huge(hp_base, 4, hp_size);
        m->mode = SP_CHANNEL_DISABLED;
        *out = m;
        return SP_OK;
    }

    for (int i = 0; i < n_bits && probe_ok; i++) {
        if (sp_probe_bit((uintptr_t)hp_base, CHAN_BIT_LO + i,
                         hp_size, n_probes, pool, &results[i]) != 0) {
            fprintf(stderr, "SP_WARN: sp_channel: probe bit %d failed — DISABLED\n",
                    CHAN_BIT_LO + i);
            probe_ok = 0;
        }
    }
    sp_probe_pool_destroy(pool);
    sp_free_huge(hp_base, 4, hp_size);

    if (!probe_ok) {
        m->mode = SP_CHANNEL_DISABLED;
        *out = m;
        return SP_OK;
    }

    uint32_t k = gf2_recover_M(results, n_bits, CHAN_BIT_LO, m);
    if (k == 0) {
        fprintf(stderr, "SP_WARN: sp_channel: no channel differentiation found — DISABLED\n");
        m->mode = SP_CHANNEL_DISABLED;
        *out = m;
        return SP_OK;
    }

    m->mode = SP_CHANNEL_LIVE;
    fprintf(stderr, "SP_INFO: sp_channel: recovered M (%u × %u) — LIVE\n",
            k, (uint32_t)n_bits);
    *out = m;
    return SP_OK;
}

uint32_t sp_channel_of(const sp_channel_map *m, uintptr_t addr) {
    if (!m || m->mode == SP_CHANNEL_DISABLED) return SP_CHANNEL_UNSPECIFIED;
    uint32_t channel = 0;
    for (uint32_t j = 0; j < m->k; j++) {
        uint32_t bit = 0;
        for (uint32_t i = 0; i < m->n; i++) {
            if (m->M[j][i])
                bit ^= (uint32_t)((addr >> (m->n_start + i)) & 1u);
        }
        channel |= (bit << j);
    }
    return channel;
}

sp_channel_mode sp_channel_map_mode(const sp_channel_map *m) {
    return m ? m->mode : SP_CHANNEL_DISABLED;
}

sp_status sp_channel_map_dims(const sp_channel_map *m,
                               uint32_t *k_out, uint32_t *n_out) {
    if (!m || !k_out || !n_out) {
        sp_set_error("sp_channel_map_dims: NULL arg");
        return SP_EBADARG;
    }
    *k_out = m->k;
    *n_out = m->n;
    return SP_OK;
}

void sp_channel_map_free(sp_channel_map *m) { free(m); }

/* ── Channel-pair allocator ───────────────────────────────────────────────── */

#define SP_PAIR_CACHE_LINE 64u

static sp_status alloc_pair_fallback(sp_channel_pair_arena *arena,
                                     void **ptr_a_out, void **ptr_b_out) {
    void *blk = calloc(1, 2u * SP_PAIR_CACHE_LINE);
    if (!blk) return SP_ENOMEM;
    arena->base     = blk;
    arena->n_pages  = 0;
    arena->page_size = 0;
    arena->is_huge  = 0;
    *ptr_a_out = blk;
    *ptr_b_out = (char *)blk + SP_PAIR_CACHE_LINE;
    return SP_OK;
}

sp_status sp_alloc_channel_pair(const sp_channel_map *m,
                                void **ptr_a_out,
                                void **ptr_b_out,
                                sp_channel_pair_arena **arena_out) {
    if (!ptr_a_out || !ptr_b_out || !arena_out) {
        sp_set_error("sp_alloc_channel_pair: NULL arg");
        return SP_EBADARG;
    }
    *ptr_a_out = *ptr_b_out = NULL;
    *arena_out = NULL;

    struct sp_channel_pair_arena *arena =
        (struct sp_channel_pair_arena *)calloc(1, sizeof *arena);
    if (!arena) return SP_ENOMEM;

    /* DISABLED path: log warning and return malloc fallback */
    if (!m || m->mode == SP_CHANNEL_DISABLED) {
        fprintf(stderr,
            "SP_WARN: Virtualized memory controller detected, disabling TailSlayer\n");
        sp_status rc = alloc_pair_fallback(arena, ptr_a_out, ptr_b_out);
        if (rc != SP_OK) { free(arena); return rc; }
        *arena_out = arena;
        return SP_OK;
    }

    /* LIVE path: allocate huge-page arena, scan for channel-diverse pair */
    size_t hp_size = detect_huge_page_size();
    void *base = sp_alloc_huge(4, hp_size);
    if (!base) {
        fprintf(stderr,
            "SP_WARN: sp_alloc_channel_pair: huge-page alloc failed — DISABLED fallback\n");
        sp_status rc = alloc_pair_fallback(arena, ptr_a_out, ptr_b_out);
        if (rc != SP_OK) { free(arena); return rc; }
        *arena_out = arena;
        return SP_OK;
    }

    arena->base      = base;
    arena->n_pages   = 4;
    arena->page_size = hp_size;
    arena->is_huge   = 1;

    /* Scan cache-line-aligned addresses within the arena for a diverse pair */
    uintptr_t scan_start = (uintptr_t)base;
    uintptr_t scan_end   = (uintptr_t)base + 4u * hp_size - SP_PAIR_CACHE_LINE;
    void     *found_a    = NULL;
    void     *found_b    = NULL;
    uint32_t  ch_a       = SP_CHANNEL_UNSPECIFIED;

    for (uintptr_t p = scan_start; p < scan_end; p += SP_PAIR_CACHE_LINE) {
        uint32_t ch = sp_channel_of(m, p);
        if (!found_a) {
            found_a = (void *)p;
            ch_a    = ch;
        } else if (ch != ch_a) {
            found_b = (void *)p;
            break;
        }
    }

    if (!found_b) {
        /* No diverse pair found — fall back rather than failing hard */
        sp_free_huge(base, 4, hp_size);
        fprintf(stderr,
            "SP_WARN: sp_alloc_channel_pair: no channel-diverse pair in arena — DISABLED fallback\n");
        arena->base = NULL; arena->n_pages = 0; arena->page_size = 0;
        arena->is_huge = 0;
        sp_status rc = alloc_pair_fallback(arena, ptr_a_out, ptr_b_out);
        if (rc != SP_OK) { free(arena); return rc; }
        *arena_out = arena;
        return SP_OK;
    }

    *ptr_a_out = found_a;
    *ptr_b_out = found_b;
    *arena_out = arena;
    return SP_OK;
}

void sp_free_channel_pair(sp_channel_pair_arena *arena) {
    if (!arena) return;
    if (arena->is_huge)
        sp_free_huge(arena->base, arena->n_pages, arena->page_size);
    else
        free(arena->base);
    free(arena);
}
