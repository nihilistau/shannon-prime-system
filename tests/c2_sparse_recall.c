/* c2_sparse_recall.c — CONTRACT C2 sparse-recall fidelity (the usable-context number).
 *
 * Ring-2 (c2_ring2_measure.c) gives a ~hundreds-x STORAGE multiplier but solves
 * only the memory wall: full attention is still O(N). To make the stored context
 * USABLE at bounded compute we must recall only a budget B << N of the cached
 * tokens per query. This harness measures, on a needle-in-haystack synthetic
 * attention (recent-coherent cluster + a few planted distant "needles" + diffuse
 * background), how well each recall pattern reproduces FULL attention vs budget B:
 *
 *   SWA        last B tokens (Gemma-style sliding window)
 *   PHI        Fibonacci/Three-Gap sub-sample  idx = floor(k*phi*N) mod N
 *   RECENT_PHI last B/2  UNION  PHI(B/2)   (local fidelity + long-range coverage)
 *   KSTE       top-B by KSTE-signature similarity to the query (tests whether the
 *              lossy 64-B dominance signature is a viable directional recall router)
 *   ORACLE     top-B by TRUE attention score (the upper bound)
 *
 * Metric: cosine(o_pattern, o_full) and needle-capture fraction. Self-contained
 * (math-core kste only). The result decides the usable-context multiplier and
 * feeds the System-1/2 crossover oracle (derived from the measured budget).
 */
#include "sp/kste.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

static uint64_t s0=0xD1B54A32D192ED03ULL,s1=0xAEF17502108EF2D9ULL;
static double urand(void){uint64_t x=s0,y=s1;s0=y;x^=x<<23;s1=x^y^(x>>17)^(y>>26);return (double)((s1+y)>>11)/9007199254740992.0;}
static double nrnd(void){double u=urand(),v=urand();if(u<1e-12)u=1e-12;return sqrt(-2.0*log(u))*cos(6.283185307179586*v);}

#define N    4096
#define HD   128
#define NDL  8            /* number of distant needles */
static const double PHI=0.6180339887498949;

static float *K,*V,q[HD],ofull[HD];
static double score[N];
static int needle[NDL];
static sp_kste_tree_t ktree[N], qtree;

static int16_t tier0(const sp_kste_tree_t*t,int j){ /* root label j in [0,6): bytes 8..19, int16 LE */
    int o=8+2*j; return (int16_t)((uint8_t)t->bytes[o] | ((uint8_t)t->bytes[o+1]<<8)); }

static void attn_subset(const int*idx,int m,float*o){
    double mx=-1e300; for(int j=0;j<m;j++){double s=score[idx[j]]; if(s>mx)mx=s;}
    double Z=0; static double p[N]; for(int j=0;j<m;j++){p[j]=exp(score[idx[j]]-mx); Z+=p[j];}
    for(int d=0;d<HD;d++)o[d]=0;
    for(int j=0;j<m;j++){double w=p[j]/Z; const float*v=V+(size_t)idx[j]*HD; for(int d=0;d<HD;d++)o[d]+=(float)(w*v[d]);}
}
static double cosfull(const float*o){double dot=0,na=0,nb=0;for(int d=0;d<HD;d++){dot+=(double)o[d]*ofull[d];na+=(double)o[d]*o[d];nb+=(double)ofull[d]*ofull[d];}return (na>0&&nb>0)?dot/(sqrt(na)*sqrt(nb)):0;}
static int needles_in(const int*idx,int m){int c=0;for(int n=0;n<NDL;n++)for(int j=0;j<m;j++)if(idx[j]==needle[n]){c++;break;}return c;}

int main(void){
    K=malloc((size_t)N*HD*sizeof(float)); V=malloc((size_t)N*HD*sizeof(float));
    for(int d=0;d<HD;d++)q[d]=(float)nrnd();
    /* needles: distinct distant indices in [0, N-256) */
    for(int n=0;n<NDL;n++)needle[n]=(int)(urand()*(N-256));
    int recentlo=N-64;
    for(int i=0;i<N;i++){
        for(int d=0;d<HD;d++)K[(size_t)i*HD+d]=(float)(0.30*nrnd());
        double a=0;
        if(i>=recentlo) a=0.9;                         /* recent coherent cluster */
        for(int n=0;n<NDL;n++) if(i==needle[n]) a=1.6; /* planted distant needles (strong) */
        if(a>0) for(int d=0;d<HD;d++)K[(size_t)i*HD+d]+=(float)(a*q[d]);
        /* V: needles + recent share a signal direction so missing them moves the output */
        for(int d=0;d<HD;d++)V[(size_t)i*HD+d]=(float)(0.3*nrnd());
        if(a>0.5) for(int d=0;d<HD;d++)V[(size_t)i*HD+d]+=(float)(a* (d%2?1.0:-1.0));
    }
    /* true scores + full attention */
    double inv=1.0/sqrt((double)HD);
    for(int i=0;i<N;i++){double s=0;const float*k=K+(size_t)i*HD;for(int d=0;d<HD;d++)s+=(double)q[d]*k[d];score[i]=s*inv;}
    { int *all=malloc(N*sizeof(int)); for(int i=0;i<N;i++)all[i]=i; attn_subset(all,N,ofull); free(all);}
    /* KSTE signatures (scale 256, no saturation) for K and q */
    { int32_t kq[HD]; for(int i=0;i<N;i++){const float*k=K+(size_t)i*HD;for(int d=0;d<HD;d++)kq[d]=(int32_t)lrintf(k[d]*256.0f);sp_kste_encode(kq,HD,&ktree[i]);}
      for(int d=0;d<HD;d++)kq[d]=(int32_t)lrintf(q[d]*256.0f); sp_kste_encode(kq,HD,&qtree); }

    const int Bs[]={64,128,256,512,1024}; const int NB=5;
    int *idx=malloc(N*sizeof(int));
    printf("# CONTRACT C2 — sparse-recall fidelity vs FULL attention (N=%d, HD=%d, %d needles)\n",N,HD,NDL);
    printf("# metric: cosine(o_pattern,o_full) ; (needles captured / %d)\n\n",NDL);
    printf("%6s | %16s | %16s | %16s | %16s | %16s\n","B","SWA(last B)","PHI(Fib)","RECENT+PHI","KSTE-guided","ORACLE(top-B)");
    for(int bi=0;bi<NB;bi++){
        int B=Bs[bi]; char cell[5][48];
        /* SWA */
        for(int j=0;j<B;j++)idx[j]=N-B+j; { float o[HD];attn_subset(idx,B,o); snprintf(cell[0],48,"%.4f (%d/%d)",cosfull(o),needles_in(idx,B),NDL);}
        /* PHI */
        { int m=0; static char seen[N]; memset(seen,0,N); for(int k=1;m<B && k<8*N;k++){int p=(int)(fmod((double)k*PHI,1.0)*N); if(p<0)p+=N; if(p>=N)p%=N; if(!seen[p]){seen[p]=1;idx[m++]=p;}} float o[HD];attn_subset(idx,m,o); snprintf(cell[1],48,"%.4f (%d/%d)",cosfull(o),needles_in(idx,m),NDL);}
        /* RECENT + PHI */
        { int m=0; static char seen[N]; memset(seen,0,N); int half=B/2; for(int j=0;j<half;j++){int p=N-1-j; if(p>=0&&!seen[p]){seen[p]=1;idx[m++]=p;}} for(int k=1;m<B && k<8*N;k++){int p=(int)(fmod((double)k*PHI,1.0)*N); if(p<0)p+=N; if(p>=N)p%=N; if(!seen[p]){seen[p]=1;idx[m++]=p;}} float o[HD];attn_subset(idx,m,o); snprintf(cell[2],48,"%.4f (%d/%d)",cosfull(o),needles_in(idx,m),NDL);}
        /* KSTE-guided: top-B by -L1(tier0(K_i),tier0(q)) */
        { typedef struct{double s;int i;}si; si*arr=malloc(N*sizeof(si));
          for(int i=0;i<N;i++){long d=0;for(int j=0;j<6;j++){long e=(long)tier0(&ktree[i],j)-tier0(&qtree,j); d+= e<0?-e:e;} arr[i].s=-(double)d; arr[i].i=i;}
          /* partial selection: simple O(N*B) max-extract */
          for(int t=0;t<B;t++){int best=t;for(int u=t+1;u<N;u++)if(arr[u].s>arr[best].s)best=u; si tmp=arr[t];arr[t]=arr[best];arr[best]=tmp; idx[t]=arr[t].i;}
          float o[HD];attn_subset(idx,B,o); snprintf(cell[3],48,"%.4f (%d/%d)",cosfull(o),needles_in(idx,B),NDL); free(arr);}
        /* ORACLE: top-B by true score */
        { typedef struct{double s;int i;}si; si*arr=malloc(N*sizeof(si));
          for(int i=0;i<N;i++){arr[i].s=score[i];arr[i].i=i;}
          for(int t=0;t<B;t++){int best=t;for(int u=t+1;u<N;u++)if(arr[u].s>arr[best].s)best=u; si tmp=arr[t];arr[t]=arr[best];arr[best]=tmp; idx[t]=arr[t].i;}
          float o[HD];attn_subset(idx,B,o); snprintf(cell[4],48,"%.4f (%d/%d)",cosfull(o),needles_in(idx,B),NDL); free(arr);}
        printf("%6d | %16s | %16s | %16s | %16s | %16s\n",B,cell[0],cell[1],cell[2],cell[3],cell[4]);
    }
    printf("\n# read: cosine ~1.0 + needles=%d/%d = pattern reproduces full attention at that budget.\n",NDL,NDL);
    return 0;
}
