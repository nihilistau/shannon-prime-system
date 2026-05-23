/* sp_error.h — the backend-facing error-detail contract of libshannonprime.
 *
 * The L1 ABI's read side, sp_last_error() (the L2/Rust-facing getter), is frozen in
 * sp/sp_status.h (PPT-LAT-L1-ABI-v0 Appendix A §7). Its write-side companion,
 * sp_set_error(), is the symbol the L1 implementation and the four backends (CUDA /
 * Vulkan / Hexagon) call to stash the human-readable detail for the next sp_last_error().
 * It is the same thread-local error surface, but on the producer side of the boundary.
 *
 * This header exists so backends consume sp_set_error through a declaration they can SEE,
 * rather than extern-declaring it ad hoc (a contract a math-core refactor could silently
 * break). L2/Rust bindings never call sp_set_error — they only read sp_last_error. The
 * definition lives in core/io_format/sp_error.c (a backend-agnostic TU, always linked).
 */
#ifndef SP_ERROR_H
#define SP_ERROR_H

#include "sp/sp_status.h"   /* the read side: sp_last_error() */

#ifdef __cplusplus
extern "C" {
#endif

/* Set the thread-local error detail string read back by sp_last_error() on the same
 * thread (truncated to the internal buffer; a NULL msg clears it). The backend-facing
 * producer half of the error surface — called by the L1 implementation and the backends;
 * never by an L2 binding. */
void sp_set_error(const char *msg);

#ifdef __cplusplus
}
#endif
#endif /* SP_ERROR_H */
