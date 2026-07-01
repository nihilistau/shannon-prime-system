/* sieve_measure.c — FRESH re-derivation of the Friedman-sieve application numbers
 * against the CURRENT math-core (core/kste + core/sieve), written from scratch
 * (anti-contamination: no code taken from the old shannon-prime/ repos; those
 * proved it, this re-proves it on the new substrate).
 *
 * Measures three things the archived Paper III sec 11.6 claimed but had no
 * on-disk log for in these repos:
 *   A) dedup / eviction rate + frontier PLATEAU on i.i.d. Gaussian streams
 *   B) dominance-check + per-candidate sieve wall-time (throughput, p50/p99)
 *   C) DISCRIMINATION: intra-cluster vs inter-cluster "would-dedup" rate
 *
 * Build (from repo root, S=shannon-prime-system):
 *   gcc -O2 -std=c11 -Wall -I"$S/include" "$S/tests/sieve_measure.c" \
 *     "$S/core/sieve/sp_sieve.c" "$S/core/kste/kste_encode.c" \
 *     "$S/core/kste/kste_dominance.c" "$S/core/io_hash/sp_hash.c" \
 *     -o sieve_measure -lm && ./sieve_measure
 */
#include "sp/kste.h"
#include "sp/sp_sieve.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define K_DIM 128

/* deterministic xorshift64* so the whole run is reproducible */
static uint64_t RNG = 0x9E3779B97F4A7C15ULL;
static inline uint64_t xr(void){ uint64_t x=RNG; x^=x>>12; x^=x<<25; x^=x>>27; RNG=x; return x*0x2545F4914F6CDD1DULL; }
static inline double u01(void){ return (double)(xr()>>11) * (1.0/9007199254740992.0); }
static inline double gauss(void){ double u=u01(); if(u<1e-300)u=1e-300; return sqrt(-2.0*log(u))*cos(6.283185307179586*u01()); }

static void quantize(const double *x, int k, int32_t *out, double scale){
    for(int i=0;i<k;i++){ double v=x[i]*scale; if(v>2.147e9)v=2.147e9; if(v<-2.147e9)v=-2.147e9; out[i]=(int32_t)v; }
}
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return (x<y)?-1:(x>y)?1:0; }

/* would the sieve treat a,b as comparable (i.e. dedup/fold rather than keep both)? */
static int comparable(const sp_kste_tree_t*a,const sp_kste_tree_t*b){
    sp_dom_t t0=sp_kste_tier0(a,b), t1=sp_kste_tier1(a,b);
    int c0=(t0!=SP_INCOMPARABLE), c1=(t1!=SP_INCOMPARABLE);
    return c0 && c1;               /* both tiers non-incomparable => sieve acts */
}

int main(void){
    printf("=== G-SIEVE-MEASURE : fresh re-derivation on current core (K=%d) ===\n\n", K_DIM);

    /* ---------- A) dedup / eviction rate + plateau ---------- */
    {
        const size_t N = 200000;
        const size_t CAP = N;               /* never artificially full */
        sp_kste_tree_t *frontier = malloc(CAP*sizeof(*frontier));
        sp_sieve_event_t ev;
        size_t fn=0, added=0, folded=0, discarded=0;
        double xv[K_DIM]; int32_t qv[K_DIM];
        RNG=12345;
        size_t plateau_hit=0; size_t last=0, stable=0;
        for(size_t i=0;i<N;i++){
            for(int j=0;j<K_DIM;j++) xv[j]=gauss();
            sp_kste_tree_t cand; quantize(xv,K_DIM,qv,1e6); sp_kste_encode(qv,K_DIM,&cand);
            size_t nev=0, before=fn;
            sp_sieve_evaluate(&cand,1,frontier,&fn,CAP,&ev,&nev);
            if(nev>0) folded++; else if(fn>before) added++; else discarded++;
            if(fn==last){ if(++stable==20000 && !plateau_hit) plateau_hit=i; } else { stable=0; last=fn; }
        }
        printf("[A] i.i.d. Gaussian stream, N=%zu\n", N);
        printf("    final frontier (PLATEAU) = %zu slots\n", fn);
        printf("    added(novel/incomp)=%zu  folded(strict-dom)=%zu  discarded(dominated/equiv)=%zu\n", added,folded,discarded);
        printf("    EVICTION RATE (folded+discarded)/N = %.2f%%   (kept-as-new = %.2f%%)\n",
               100.0*(folded+discarded)/N, 100.0*added/N);
        if(plateau_hit) printf("    frontier stopped growing (~20k steps no change) around i=%zu\n", plateau_hit);
        free(frontier);
    }

    /* ---------- B) wall-time: raw dominance pair + per-candidate sieve ---------- */
    {
        printf("\n[B] wall-time\n");
        /* B1: raw tier0+tier1 pair throughput */
        double xa[K_DIM],xb[K_DIM]; int32_t qa[K_DIM],qb[K_DIM]; sp_kste_tree_t A,B;
        RNG=777; for(int j=0;j<K_DIM;j++){xa[j]=gauss();xb[j]=gauss();}
        quantize(xa,K_DIM,qa,1e6); quantize(xb,K_DIM,qb,1e6); sp_kste_encode(qa,K_DIM,&A); sp_kste_encode(qb,K_DIM,&B);
        volatile int sink=0; const long M=20000000;
        double t0=now_s();
        for(long i=0;i<M;i++){ sink ^= (int)sp_kste_tier0(&A,&B); sink ^= (int)sp_kste_tier1(&A,&B); }
        double dt=now_s()-t0;
        printf("    raw tier0+tier1 dominance: %.2f ns/pair  (%.1fM pairs/s)  [sink=%d]\n",
               dt/M*1e9, M/dt/1e6, sink);

        /* B2: per-candidate sieve against a realistic frontier (~F members) */
        const size_t F=4096; sp_kste_tree_t *fr=malloc((F+1)*sizeof(*fr));
        sp_sieve_event_t ev; size_t fn=0; double xv[K_DIM]; int32_t qv[K_DIM]; RNG=999;
        for(size_t i=0;i<200000 && fn<F;i++){ for(int j=0;j<K_DIM;j++)xv[j]=gauss(); sp_kste_tree_t c; quantize(xv,K_DIM,qv,1e6); sp_kste_encode(qv,K_DIM,&c); size_t nev=0; sp_sieve_evaluate(&c,1,fr,&fn,F,&ev,&nev); }
        const int Q=20000; double *lat=malloc(Q*sizeof(double));
        for(int i=0;i<Q;i++){ for(int j=0;j<K_DIM;j++)xv[j]=gauss(); sp_kste_tree_t c; quantize(xv,K_DIM,qv,1e6); sp_kste_encode(qv,K_DIM,&c);
            size_t fn2=fn, nev=0; double s=now_s(); sp_sieve_evaluate(&c,1,fr,&fn2,F+1,&ev,&nev); lat[i]=(now_s()-s)*1e6; }
        qsort(lat,Q,sizeof(double),cmp_d);
        double mean=0; for(int i=0;i<Q;i++)mean+=lat[i]; mean/=Q;
        printf("    per-candidate sieve vs frontier=%zu: mean=%.3f us  p50=%.3f us  p99=%.3f us  (Q=%d)\n",
               fn, mean, lat[Q/2], lat[(int)(Q*0.99)], Q);
        printf("    50us-gate: %s (p99 %.3f us)\n", lat[(int)(Q*0.99)]<50.0?"CLEAR":"MISS", lat[(int)(Q*0.99)]);
        free(fr); free(lat);
    }

    /* ---------- C) discrimination: intra vs inter cluster ---------- */
    {
        printf("\n[C] discrimination (intra-cluster vs inter-cluster 'would-dedup' rate)\n");
        const int NC=60, PER=20;                 /* 60 clusters x 20 members */
        double noises[4]={0.005,0.02,0.05,0.10};
        for(int ni=0; ni<4; ni++){
            double sigma=noises[ni];
            sp_kste_tree_t *trees=malloc(NC*PER*sizeof(*trees));
            double *bases=malloc(NC*K_DIM*sizeof(double));
            double coss=0; int cosn=0;
            RNG=2024u+ni*101u;
            for(int c=0;c<NC;c++){ for(int j=0;j<K_DIM;j++) bases[c*K_DIM+j]=gauss(); }
            for(int c=0;c<NC;c++) for(int p=0;p<PER;p++){
                double xv[K_DIM]; int32_t qv[K_DIM]; double dot=0,nb=0,nx=0;
                for(int j=0;j<K_DIM;j++){ double b=bases[c*K_DIM+j]; double x=b+sigma*gauss(); xv[j]=x; dot+=b*x; nb+=b*b; nx+=x*x; }
                if(p>0){ coss+= dot/(sqrt(nb*nx)+1e-12); cosn++; }
                quantize(xv,K_DIM,qv,1e6); sp_kste_encode(qv,K_DIM,&trees[c*PER+p]);
            }
            /* intra pairs */
            long intra_c=0,intra_n=0,inter_c=0,inter_n=0;
            for(int c=0;c<NC;c++) for(int p=0;p<PER;p++) for(int q=p+1;q<PER;q++){ intra_n++; if(comparable(&trees[c*PER+p],&trees[c*PER+q])) intra_c++; }
            /* inter pairs: sample cluster c vs c+1, member 0 vs all */
            for(int c=0;c<NC;c++){ int d=(c+1)%NC; for(int p=0;p<PER;p++) for(int q=0;q<PER;q++){ inter_n++; if(comparable(&trees[c*PER+p],&trees[d*PER+q])) inter_c++; } }
            double ir=(double)intra_c/intra_n, er=(double)inter_c/inter_n;
            printf("    sigma=%.3f  mean intra-cos=%.4f | intra-dedup=%.3f (%ld/%ld)  inter-dedup=%.4f (%ld/%ld)  ratio=%.1fx\n",
                   sigma, coss/(cosn?cosn:1), ir,intra_c,intra_n, er,inter_c,inter_n, er>1e-9? ir/er : ir/ (1.0/inter_n));
            free(trees); free(bases);
        }
        printf("    (ratio >> 1 => near-duplicates within a cluster are deduped far more than across clusters = real discrimination)\n");
    }

    printf("\n=== done ===\n");
    return 0;
}
