/* c2_ring2_measure.c — CONTRACT C2 Ring-2 measurement (gate C2_RING2_RECALL).
 *
 * The C2 KV measurement (c2_kv_measure.c) showed neither per-vector overlay is
 * "120x reconstructable KV": the faithful Spinor codec is ~3.5x/f32. So the
 * unlimited-context headline must be (faithful ~3.5x) x (Ring-2 effective-context
 * multiplier). This harness measures Ring-2 directly, self-contained (math-core
 * Spinor codec only, no model):
 *
 *   1. BIT-EXACT RECALL (the gate). KV head-vectors -> Spinor blocks -> a RAM
 *      window of the W most-recent tokens, older tokens SPILLED to a disk file.
 *      Recall a spilled token: read its blocks back, decode, and compare to the
 *      decode of the same blocks that were never spilled. Must be byte-identical
 *      (Spinor decode is deterministic; disk is a transparent byte store).
 *
 *   2. RECALL LATENCY vs RECOMPUTE. Time read+decode per token from the spill
 *      file (tmpfs models the byte-addressable near-RAM Optane tier, E:/F:).
 *      Compare to the recompute alternative (re-run the layer QKV projection for
 *      that token = hidden*KVD MACs + needs weights resident).
 *
 *   3. EFFECTIVE-CONTEXT MULTIPLIER. For a bounded in-RAM KV window, how much
 *      total context fits once cold blocks spill to an Optane budget:
 *      multiplier = (RAM_window + Optane_budget) / RAM_window. THIS is the
 *      honest meaning of the "~120x / unlimited context" headline.
 */
#include "sp/spinor_block.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

static uint64_t s0=0x9E3779B97F4A7C15ULL,s1=0xBF58476D1CE4E5B9ULL;
static double urand(void){uint64_t x=s0,y=s1;s0=y;x^=x<<23;s1=x^y^(x>>17)^(y>>26);return (double)((s1+y)>>11)/9007199254740992.0;}
static double nrand(void){double u=urand(),v=urand();if(u<1e-12)u=1e-12;return sqrt(-2.0*log(u))*cos(6.283185307179586*v);}
static double now_s(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

/* one token's KV across all layers/kv-heads, as Spinor blocks (K and V) */
typedef struct { int L,NKV,HD,NBLK; } cfg_t;
static size_t blocks_per_token(const cfg_t*c){return (size_t)c->L*c->NKV*c->NBLK*2;} /*K+V*/
static size_t bytes_per_token(const cfg_t*c){return blocks_per_token(c)*63;}

int main(void){
    /* representative configs (Executive Qwen3-0.6B-ish, and a long-ctx 7B-ish) */
    cfg_t cfgs[2]={ {28,8,128, sp_spinor_blocks_for(128)}, {32,8,128, sp_spinor_blocks_for(128)} };
    const char* names[2]={"qwen3-0.6B-like (L28,NKV8,HD128)","7B-like (L32,NKV8,HD128)"};

    printf("# CONTRACT C2 — Ring-2 offload/recall measured (math-core Spinor only, no model)\n");
    printf("# spill file on tmpfs (models byte-addressable near-RAM Optane E:/F:)\n\n");

    for(int ci=0; ci<2; ci++){
        cfg_t c=cfgs[ci];
        size_t bpt=bytes_per_token(&c), blkpt=blocks_per_token(&c);
        printf("## %s : %d blocks/token, %zu B/token (Spinor)  | f32 KV/token = %zu B (%.2fx)\n",
               names[ci], (int)blkpt, bpt, (size_t)c.L*c.NKV*c.HD*2*4, (double)(c.L*c.NKV*c.HD*2*4)/bpt);

        const int NTOK=4000;              /* generate this many tokens of KV */
        const int W=512;                  /* Ring-1 in-RAM window (most-recent tokens) */
        /* RAM ref: keep a decoded copy of EVERY token (to check recall bit-exactness) */
        float *refdec=(float*)malloc((size_t)NTOK*c.NKV*c.HD*2*sizeof(float)); /* K|V per kvhead */
        /* spill file */
        const char* path="/tmp/c2_ring2_spill.bin";
        FILE* f=fopen(path,"wb+"); if(!f){printf("  fopen fail\n");return 1;}

        sp_spinor_block_t *tokblk=(sp_spinor_block_t*)malloc(blkpt*sizeof(sp_spinor_block_t));
        float *vbuf=(float*)malloc((size_t)c.HD*sizeof(float));

        double t_write=0;
        /* generate + encode + (spill all to disk so we can recall any) */
        for(int t=0;t<NTOK;t++){
            size_t bi=0;
            for(int l=0;l<c.L;l++) for(int kv=0;kv<2;kv++) for(int h=0;h<c.NKV;h++){
                for(int i=0;i<c.HD;i++) vbuf[i]=(float)(nrand()*2.0);
                if(t<NTOK){ /* store the canonical decode for token t (only for l==0,kv,h to bound RAM) */ }
                sp_spinor_encode_vec(vbuf,c.HD,&tokblk[bi]);
                /* keep a reference decode for layer 0 only (enough to prove bit-exact recall) */
                if(l==0){ float dec[256]; sp_spinor_decode_vec(&tokblk[bi],c.HD,dec);
                          memcpy(refdec+((size_t)t*c.NKV*2+(size_t)kv*c.NKV+h)*c.HD, dec, c.HD*sizeof(float)); }
                bi+=c.NBLK;
            }
            double w0=now_s(); fwrite(tokblk,sizeof(sp_spinor_block_t),blkpt,f); t_write+=now_s()-w0;
        }
        fflush(f);

        /* (1)+(2) RECALL: read back random spilled tokens, decode layer0, compare to ref */
        const int NREC=2000; int bad=0; double t_read=0,t_dec=0;
        sp_spinor_block_t *rd=(sp_spinor_block_t*)malloc(blkpt*sizeof(sp_spinor_block_t));
        float dec[256];
        for(int r=0;r<NREC;r++){
            int t=(int)(urand()*NTOK); if(t>=NTOK)t=NTOK-1;
            double r0=now_s();
            fseek(f,(long)((size_t)t*blkpt*sizeof(sp_spinor_block_t)),SEEK_SET);
            fread(rd,sizeof(sp_spinor_block_t),blkpt,f);
            t_read+=now_s()-r0;
            /* decode layer0's K/V heads and compare bit-exact to ref */
            size_t bi=0;
            for(int l=0;l<c.L;l++) for(int kv=0;kv<2;kv++) for(int h=0;h<c.NKV;h++){
                if(l==0){ double d0=now_s(); sp_spinor_decode_vec(&rd[bi],c.HD,dec); t_dec+=now_s()-d0;
                          float *ref=refdec+((size_t)t*c.NKV*2+(size_t)kv*c.NKV+h)*c.HD;
                          if(memcmp(dec,ref,c.HD*sizeof(float))!=0) bad++; }
                bi+=c.NBLK;
            }
        }
        printf("   (1) BIT-EXACT RECALL: %d/%d recalled tokens layer0 K/V byte-identical to in-RAM decode  -> %s\n",
               NREC-bad, NREC, bad==0?"PASS":"FAIL");
        printf("   (2) recall cost: read %.2f us/token (%.1f MB/s), decode %.2f us/block ; spill write %.1f MB/s\n",
               t_read/NREC*1e6, (double)bpt*NREC/(t_read+1e-12)/1e6,
               t_dec/(NREC*c.NKV*2)*1e6, (double)bpt*NTOK/(t_write+1e-12)/1e6);
        printf("       recompute alternative = re-run layer QKV proj for that token (hidden*KVD MACs + weights resident);\n");
        printf("       recall reads %zu compressed B + %d Spinor-decodes, no weights, no matmul.\n", bpt, (int)blkpt);

        /* (3) EFFECTIVE-CONTEXT MULTIPLIER for Optane budgets */
        double ram_window_B=(double)W*bpt;
        printf("   (3) effective-context multiplier (in-RAM window W=%d tokens = %.1f MB):\n", W, ram_window_B/1e6);
        long budgets[3]={16L<<30,32L<<30,(16L+32L)<<30}; const char* bn[3]={"E: 16GB","F: 32GB","E+F 48GB"};
        for(int b=0;b<3;b++){
            double tot_tokens=(ram_window_B+budgets[b])/bpt;
            printf("        Optane %-9s -> %.0f total tokens vs %d in-RAM  = %.1fx effective context\n",
                   bn[b], tot_tokens, W, tot_tokens/W);
        }
        printf("\n");
        fclose(f); remove(path); free(refdec); free(tokblk); free(vbuf); free(rd);
    }
    printf("# done.\n");
    return 0;
}
