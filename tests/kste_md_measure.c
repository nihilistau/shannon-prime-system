/* kste_md_measure.c — measure the v2 magnitude-as-depth encoder (kste_md) and
 * put it head-to-head with the frozen v1 order-statistic encoder on the SAME
 * cluster-discrimination probe. Fresh harness (anti-contamination).
 *
 * [A] frontier plateau + eviction on i.i.d. Gaussian (v2 Dickson dominance)
 * [B] dominance-check wall-time (v2)
 * [C] discrimination: intra-cluster vs inter-cluster "would-dedup" rate, v1 vs v2
 *
 * Build (S=shannon-prime-system):
 *   gcc -O2 -std=c11 -D_POSIX_C_SOURCE=199309L -Wall -I"$S/include" \
 *     "$S/tests/kste_md_measure.c" "$S/core/kste_md/kste_md.c" \
 *     "$S/core/kste/kste_encode.c" "$S/core/kste/kste_dominance.c" \
 *     -o kste_md_measure -lm && ./kste_md_measure
 */
#include "sp/kste.h"
#include "sp/kste_md.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define K_DIM 128

static uint64_t RNG = 0x9E3779B97F4A7C15ULL;
static inline uint64_t xr(void){ uint64_t x=RNG; x^=x>>12; x^=x<<25; x^=x>>27; RNG=x; return x*0x2545F4914F6CDD1DULL; }
static inline double u01(void){ return (double)(xr()>>11)*(1.0/9007199254740992.0); }
static inline double gauss(void){ double u=u01(); if(u<1e-300)u=1e-300; return sqrt(-2.0*log(u))*cos(6.283185307179586*u01()); }
static void quantize(const double*x,int k,int32_t*o,double s){ for(int i=0;i<k;i++){ double v=x[i]*s; if(v>2.147e9)v=2.147e9; if(v<-2.147e9)v=-2.147e9; o[i]=(int32_t)v; } }
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return (x<y)?-1:(x>y)?1:0; }

/* v1 "would-dedup": both tiers non-incomparable */
static int v1_comparable(const sp_kste_tree_t*a,const sp_kste_tree_t*b){
    return sp_kste_tier0(a,b)!=SP_INCOMPARABLE && sp_kste_tier1(a,b)!=SP_INCOMPARABLE;
}
/* v2 "would-dedup": Dickson non-incomparable */
static int v2_comparable(const sp_kste_md_sig_t*a,const sp_kste_md_sig_t*b){
    return sp_kste_md_dom(a,b)!=SP_INCOMPARABLE;
}

int main(void){
    printf("=== G-KSTE-MD : magnitude-as-depth encoder (v2) vs frozen v1 (K=%d) ===\n\n",K_DIM);

    /* ---------- [A] v2 frontier plateau + eviction (i.i.d. Gaussian) ---------- */
    {
        const size_t N=200000, CAP=200000;
        sp_kste_md_sig_t *fr=malloc(CAP*sizeof(*fr)); size_t fn=0, added=0,folded=0,disc=0;
        double xv[K_DIM]; int32_t qv[K_DIM]; RNG=12345;
        for(size_t i=0;i<N;i++){
            for(int j=0;j<K_DIM;j++)xv[j]=gauss(); quantize(xv,K_DIM,qv,1e6);
            sp_kste_md_sig_t c; sp_kste_md_encode(qv,K_DIM,&c);
            int dominated=0, dominates_any=0;
            for(size_t f=0;f<fn;f++){ sp_dom_t d=sp_kste_md_dom(&c,&fr[f]);
                if(d==SP_DOMINATED||d==SP_EQUIVALENT){dominated=1;break;} if(d==SP_DOMINATES)dominates_any=1; }
            if(dominated){disc++;}
            else if(dominates_any){ /* replace all it dominates */
                size_t w=0; for(size_t f=0;f<fn;f++){ if(sp_kste_md_dom(&c,&fr[f])!=SP_DOMINATES) fr[w++]=fr[f]; } fn=w; fr[fn++]=c; folded++; }
            else { if(fn<CAP) fr[fn++]=c; added++; }
        }
        printf("[A] i.i.d. Gaussian N=%zu (v2 Dickson)\n",N);
        printf("    PLATEAU=%zu slots  added=%zu folded=%zu discarded=%zu\n",fn,added,folded,disc);
        printf("    eviction (folded+disc)/N=%.2f%%  kept-as-new=%.2f%%\n",100.0*(folded+disc)/N,100.0*added/N);
        free(fr);
    }

    /* ---------- [B] v2 dominance wall-time ---------- */
    {
        double xa[K_DIM],xb[K_DIM]; int32_t qa[K_DIM],qb[K_DIM]; RNG=777;
        for(int j=0;j<K_DIM;j++){xa[j]=gauss();xb[j]=gauss();}
        quantize(xa,K_DIM,qa,1e6); quantize(xb,K_DIM,qb,1e6);
        sp_kste_md_sig_t A,B; sp_kste_md_encode(qa,K_DIM,&A); sp_kste_md_encode(qb,K_DIM,&B);
        volatile int sink=0; const long M=20000000; double t0=now_s();
        for(long i=0;i<M;i++) sink^=(int)sp_kste_md_dom(&A,&B);
        double dt=now_s()-t0;
        printf("\n[B] v2 Dickson dominance: %.2f ns/pair (%.1fM/s) [sink=%d]\n",dt/M*1e9,M/dt/1e6,sink);
        /* per-encode cost */
        const long E=200000; double te=now_s();
        for(long i=0;i<E;i++){ for(int j=0;j<K_DIM;j++)xa[j]=gauss(); quantize(xa,K_DIM,qa,1e6); sp_kste_md_encode(qa,K_DIM,&A);}
        printf("    v2 encode: %.3f us/vector (%ld encodes)\n",(now_s()-te)/E*1e6,E);
    }

    /* ---------- [C] discrimination: v1 vs v2, intra vs inter cluster ---------- */
    {
        printf("\n[C] discrimination — intra vs inter cluster 'would-dedup' rate (v1 vs v2)\n");
        printf("    sigma  intra-cos |   v1 intra  v1 inter  v1 ratio |   v2 intra  v2 inter  v2 ratio\n");
        const int NC=60, PER=20;
        double noises[5]={0.005,0.02,0.05,0.10,0.20};
        for(int ni=0;ni<5;ni++){
            double sg=noises[ni];
            sp_kste_tree_t *t1=malloc(NC*PER*sizeof(*t1));
            sp_kste_md_sig_t *t2=malloc(NC*PER*sizeof(*t2));
            double *bases=malloc(NC*K_DIM*sizeof(double)); double coss=0; int cosn=0;
            RNG=2024u+ni*101u;
            for(int c=0;c<NC;c++) for(int j=0;j<K_DIM;j++) bases[c*K_DIM+j]=gauss();
            for(int c=0;c<NC;c++) for(int p=0;p<PER;p++){
                double xv[K_DIM]; int32_t qv[K_DIM]; double dot=0,nb=0,nx=0;
                for(int j=0;j<K_DIM;j++){ double b=bases[c*K_DIM+j],x=b+sg*gauss(); xv[j]=x; dot+=b*x; nb+=b*b; nx+=x*x; }
                if(p>0){coss+=dot/(sqrt(nb*nx)+1e-12);cosn++;}
                quantize(xv,K_DIM,qv,1e6);
                sp_kste_encode(qv,K_DIM,&t1[c*PER+p]);
                sp_kste_md_encode(qv,K_DIM,&t2[c*PER+p]);
            }
            long i1c=0,i1n=0,e1c=0,e1n=0,i2c=0,i2n=0,e2c=0,e2n=0;
            for(int c=0;c<NC;c++) for(int p=0;p<PER;p++) for(int q=p+1;q<PER;q++){
                i1n++; if(v1_comparable(&t1[c*PER+p],&t1[c*PER+q]))i1c++;
                i2n++; if(v2_comparable(&t2[c*PER+p],&t2[c*PER+q]))i2c++;
            }
            for(int c=0;c<NC;c++){ int d=(c+1)%NC; for(int p=0;p<PER;p++) for(int q=0;q<PER;q++){
                e1n++; if(v1_comparable(&t1[c*PER+p],&t1[d*PER+q]))e1c++;
                e2n++; if(v2_comparable(&t2[c*PER+p],&t2[d*PER+q]))e2c++;
            }}
            double v1i=(double)i1c/i1n,v1e=(double)e1c/e1n,v2i=(double)i2c/i2n,v2e=(double)e2c/e2n;
            printf("    %.3f  %.4f  |   %.3f     %.4f    %5.1fx |   %.3f     %.4f    %5.1fx\n",
                   sg,coss/(cosn?cosn:1), v1i,v1e, v1e>1e-9?v1i/v1e:0.0, v2i,v2e, v2e>1e-9?v2i/v2e:0.0);
            free(t1);free(t2);free(bases);
        }
        printf("    (ratio >> 1 = near-duplicates deduped far more than cross-cluster = real discrimination)\n");
    }
    printf("\n=== done ===\n");
    return 0;
}
