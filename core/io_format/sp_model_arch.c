/* sp_model_arch.c — sp_model_arch (PPT-LAT-L1-ABI-v0 §2; declared in sp/sp_l1.h).
 *
 * The L1 arch-discovery query: projects the loaded model's header arch_struct —
 * the memcpy-direct sp_arch_info payload written by the transcoder
 * (PPT-LAT-SP-MODEL-v0 §3) — into the caller-stack-allocated sp_arch_info. L2
 * reads this once at load to size the logits buffer (vocab_size * sizeof(float))
 * and never re-queries. L1 fills *out in place; no allocation crosses the FFI.
 *
 * Implemented over the public header accessor (sp_model_get_header), so it does
 * not reach into the opaque sp_model layout. */
#include "sp/sp_l1.h"
#include "sp/sp_model.h"
#include "sp/sp_error.h"   /* sp_set_error — producer half of the error surface */
#include <string.h>

/* The arch payload memcpy's into the 256-byte arch_struct reservation
 * (PPT-LAT-SP-MODEL-v0 §3, arch_struct_capacity = 256). Appended fields (2-L1.FP16)
 * must keep sizeof within the cap. Asserted in this C TU (sp_l1.h is also included by
 * C++ backends, where _Static_assert is not valid). */
_Static_assert(sizeof(sp_arch_info) <= 256, "sp_arch_info must fit the 256-byte arch_struct reservation");

sp_status sp_model_arch(const sp_model *m, sp_arch_info *out) {
    if (!m || !out) { sp_set_error("sp_model_arch: null arg"); return SP_EBADARG; }
    const sp_model_header *h = sp_model_get_header(m);
    if (!h) { sp_set_error("sp_model_arch: model has no header"); return SP_EBADARG; }

    /* arch_struct is the on-disk sp_arch_info image; copy the recorded bytes,
     * zero-filling any tail the producer didn't write. The loader has already
     * validated arch_struct_capacity == 256 and arch_struct_size <= 256. */
    memset(out, 0, sizeof *out);
    size_t n = h->arch_struct_size;
    if (n > sizeof *out) n = sizeof *out;
    memcpy(out, h->arch_struct, n);
    return SP_OK;
}
