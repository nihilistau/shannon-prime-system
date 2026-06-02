/* c2_kv_measure.c — CONTRACT C2 measurement harness (gate C2_KV_RATIO + fidelity).
 *
 * Measures, on representative K/V head-vectors, the TWO KV-compression overlays
 * the canonical theory distinguishes (PPT-LAT-Systems-v1 §4.2):
 *   (a) sp_spinor_encode_vec — the FAITHFUL, dimension-preserving int8 codec
 *       (63 B / 55-elem block; NBLK=ceil(HD/55)). Reports the byte ratio AND
 *       the reconstruction fidelity (max-abs err, RMSE, cosine) after a real
 *       encode->decode round-trip.
 *   (b) sp_kste_encode — the LOSSY ~130x SIGNATURE (64 B regardless of HD,
 *       valid up to dominance ⪯_d). Reports the byte ratio (= HD*4/64) AND the
 *       discrimination (distinct-tree fraction over distinct inputs) at the
 *       forward-path quant scale SP_KSTE_KV_SCALE=65536, which is where the
 *       i16-clamp interaction (M.5 gotcha) actually bites.
 *
 * Self-contained: links only math-core (vht2 + kste). No model, no backend.
 * Build (Linux sandbox, gcc): see the build line in the C2 closure note.
 */
#include "sp/spinor_block.h"
#include "sp/kste.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define SP_KSTE_KV_SCALE 65536.0   /* mirrors forward.c:52 */

/* deterministic PRNG so the run is reproducible (xorshift128+) */
static uint64_t s0 = 0x243F6A8885A308D3ULL, s1 = 0x13198A2E03707344ULL;
static double urand(void){ uint64_t x=s0,y=s1; s0=y; x^=x<<23; s1=x^y^(x>>17)^(y>>26); uint64_t r=s1+y; return (double)(r>>11)/9007199254740992.0; }
static double nrand(void){ double u=urand(), v=urand(); if(u<1e-12)u=1e-12; return sqrt(-2.0*log(u))*cos(6.283185307179586*v); }

/* Representative head-vector: gaussian core + a few heavy-tail spikes, scaled.
 * `sigma` ~ typical post-RoPE K magnitude. */
static void make_vec(float *v, int k, double sigma){
    for(int i=0;i<k;i++) v[i]=(float)(nrand()*sigma);
    /* a couple of outlier channels (RoPE/attention sinks are spiky) */
    if(k>3){ v[1]+=(float)(6.0*sigma); v[k/2]-=(float)(5.0*sigma); }
}

int main(void){
    const int HDs[] = {64,128,256,512,896,1024,2048,4096};
    const int NHD = (int)(sizeof(HDs)/sizeof(HDs[0]));
    const double sigma = 2.0;        /* representative K magnitude */

    printf("# CONTRACT C2 — KV-compression measured (gcc, math-core only)\n");
    printf("# sigma=%.1f, NBLK=ceil(HD/55), in-RAM block=63 B (on-disk 64 B w/ 0xA5 pad)\n\n", sigma);

    /* ---------- (a) Spinor faithful codec ---------- */
    printf("## (a) sp_spinor_encode_vec — FAITHFUL dimension-preserving codec\n");
    printf("%6s %5s %10s %10s %8s %8s %9s %9s %9s\n",
           "HD","NBLK","spinorB","f32B","/f32","/f16","maxAbsE","RMSE","cosine");
    for(int hi=0; hi<NHD; hi++){
        int k=HDs[hi];
        int nblk=sp_spinor_blocks_for(k);
        if(nblk>4096){ printf("%6d  (skip: too many blocks)\n",k); continue; }
        float *v=(float*)malloc(sizeof(float)*k);
        float *r=(float*)malloc(sizeof(float)*k);
        sp_spinor_block_t *blk=(sp_spinor_block_t*)malloc(sizeof(sp_spinor_block_t)*nblk);
        make_vec(v,k,sigma);
        sp_spinor_encode_vec(v,k,blk);
        int crc_bad=sp_spinor_decode_vec(blk,k,r);
        double maxe=0,se=0,dot=0,nv=0,nr=0;
        for(int i=0;i<k;i++){ double e=fabs((double)v[i]-r[i]); if(e>maxe)maxe=e; se+=e*e;
                              dot+=(double)v[i]*r[i]; nv+=(double)v[i]*v[i]; nr+=(double)r[i]*r[i]; }
        double rmse=sqrt(se/k);
        double cosv=(nv>0&&nr>0)?dot/(sqrt(nv)*sqrt(nr)):0;
        size_t spinorB=(size_t)nblk*63, f32B=(size_t)k*4, f16B=(size_t)k*2;
        printf("%6d %5d %10zu %10zu %8.3f %8.3f %9.4f %9.4f %9.6f%s\n",
               k,nblk,spinorB,f32B,(double)f32B/spinorB,(double)f16B/spinorB,maxe,rmse,cosv,
               crc_bad?"  [CRC FAIL]":"");
        free(v);free(r);free(blk);
    }

    /* ---------- (b) KSTE lossy signature ---------- */
    printf("\n## (b) sp_kste_encode — LOSSY ~130x SIGNATURE (64 B, valid up to ⪯_d)\n");
    printf("# ratio /f32 = HD*4/64 = HD/16.  Discrimination = distinct 64-B trees / N distinct inputs.\n");
    printf("# quant: kq[i]=lrintf(K[i]*65536) then i16-clamp in label_of (the M.5 interaction).\n");
    printf("%6s %8s %9s %12s %14s\n","HD","ksteB","/f32","distinctFr","distinctFr@s=2e-4");
    const int N=2000;
    for(int hi=0; hi<NHD; hi++){
        int k=HDs[hi];
        int32_t *kq=(int32_t*)malloc(sizeof(int32_t)*k);
        float *v=(float*)malloc(sizeof(float)*k);
        sp_kste_tree_t *trees=(sp_kste_tree_t*)malloc(sizeof(sp_kste_tree_t)*N);
        sp_kste_tree_t *trees2=(sp_kste_tree_t*)malloc(sizeof(sp_kste_tree_t)*N);
        for(int n=0;n<N;n++){
            make_vec(v,k,sigma);
            /* scale 1: the forward-path 65536 (saturates for |K|*65536>32767) */
            for(int i=0;i<k;i++) kq[i]=(int32_t)lrintf(v[i]*(float)SP_KSTE_KV_SCALE);
            sp_kste_encode(kq,k,&trees[n]);
            /* scale 2: a corrected small scale 2e-4*... i.e. *2.0 (keeps |val|<~16) so labels don't saturate */
            for(int i=0;i<k;i++) kq[i]=(int32_t)lrintf(v[i]*2.0f);
            sp_kste_encode(kq,k,&trees2[n]);
        }
        /* count distinct (O(N^2) memcmp, N=2000 ok) */
        int distinct=0, distinct2=0;
        for(int a=0;a<N;a++){ int dup=0,dup2=0;
            for(int b=0;b<a;b++){ if(!memcmp(&trees[a],&trees[b],64)) {dup=1;}
                                  if(!memcmp(&trees2[a],&trees2[b],64)){dup2=1;} }
            if(!dup)distinct++; if(!dup2)distinct2++; }
        size_t ksteB=64, f32B=(size_t)k*4;
        printf("%6d %8zu %9.3f %12.4f %14.4f\n",
               k,ksteB,(double)f32B/ksteB,(double)distinct/N,(double)distinct2/N);
        free(kq);free(v);free(trees);free(trees2);
    }
    printf("\n# done.\n");
    return 0;
}
