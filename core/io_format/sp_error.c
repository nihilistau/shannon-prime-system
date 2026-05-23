/* sp_error.c — the L1 ABI thread-local error-detail string. Lives in a backend-
 * agnostic TU so both halves of the surface are always defined whether or not the
 * CUDA/Vulkan/Hexagon backends are linked: the read side sp_last_error() (frozen in
 * sp/sp_status.h, L2-facing) and the backend-facing write side sp_set_error() (public
 * in sp/sp_error.h — the producer half the backends call). */
#include "sp/sp_error.h"   /* declares sp_set_error; transitively sp/sp_status.h's sp_last_error */
#include <string.h>

#if defined(_MSC_VER)
#  define SP_TLS __declspec(thread)
#else
#  define SP_TLS _Thread_local
#endif

static SP_TLS char g_err[512];

const char *sp_last_error(void) { return g_err; }

/* The backend-facing producer half (public; declared in sp/sp_error.h): set the
 * thread-local error detail (truncated to fit). Called by the L1 impl and the backends;
 * L2/Rust only ever read it back via sp_last_error(). */
void sp_set_error(const char *msg) {
    if (!msg) { g_err[0] = '\0'; return; }
    size_t n = strlen(msg);
    if (n >= sizeof(g_err)) n = sizeof(g_err) - 1;
    memcpy(g_err, msg, n);
    g_err[n] = '\0';
}
