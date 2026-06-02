/* c2_kste_router_adv.c — adversarial stress-test of the KSTE recall router.
 *
 * c2_sparse_recall.c reported KSTE-signature-guided recall catching 6/8 needles.
 * SUSPECT: that may be an artifact of building needles as K = bg + 1.6*q (which is
 * BOTH high-dot-product AND histogram-shifted). The KSTE tier-0 label is order
 * statistics (a histogram of component VALUES) — permutation-invariant, blind to
 * WHICH dimension holds which value. Dot product q.k is directional. So a vector
 * with the SAME histogram as a needle but PERMUTED components has ~identical KSTE
 * tier-0 signature yet ~zero alignment with q.
 *
 * Test: plant NDEC permuted-decoys (same multiset as a needle, components shuffled
 * -> ~0 score). If the KSTE router is a real DIRECTIONAL router it ignores them;
 * if it routes on the histogram it gets FOOLED (pulls decoys, misses budget).
 */
#include "sp/kste.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

static uint64_t s0=0xC2B2AE3D27D4EB4FULL,s1=0x165667B19E3779F9ULL;
static double urand(void){uint64_t x=s0,y=s1;s0=y;x^=x<<23;s1=x^y^(x>>17)^(y>>26);return (double)((s1+y)>>11)/9007199254740992.0;}
static double nrnd(void){double u=urand(),v=urand();if(u<1e-12)u=1e-12;return sqrt(-2.0*log(u))*cos(6.283185307179586*v);}

#define N 4096
#define HD 128
#define NDL 8
#define NDEC 32          /* permuted decoys of needles */

static float *K,*V,q[HD],ofull[HD];
static double score[N];
static int needle[NDL], decoy[NDEC];
static sp_kste_tree_t ktree[N], qtree;

static int16_t t0(const sp_kste_tree_t*t,int j){int o=8+2*j;return (int16_t)((uint8_t)t->bytes[o]|((uint8_t)t->bytes[o+1]<<8));}
static void attn(const int*idx,int m,float*o){double mx=-1e300;for(int j=0;j<m;j++){if(score[idx[j]]>mx)mx=score[idx[j]];}double Z=0;static double p[N];for(int j=0;j<m;j++){p[j]=exp(score[idx[j]]-mx);Z+=p[j];}for(int d=0;d<HD;d++)o[d]=0;for(int j=0;j<m;j++){double w=p[j]/Z;const float*v=V+(size_t)idx[j]*HD;for(int d=0;d<HD;d++)o[d]+=(float)(w*v[d]);}}
static double cosf2(const float*o){double dt=0,na=0,nb=0;for(int d=0;d<HD;d++){dt+=(double)o[d]*ofull[d];na+=(double)o[d]*o[d];nb+=(double)ofull[d]*ofull[d];}return (na>0&&nb>0)?dt/(sqrt(na)*sqrt(nb)):0;}
static int count_in(const int*idx,int m,const int*set,int ns){int c=0;for(int s=0;s<ns;s++)for(int j=0;j<m;j++)if(idx[j]==set[s]){c++;break;}return c;}

int main(void){
    K=malloc((size_t)N*HD*sizeof(float));V=malloc((size_t)N*HD*sizeof(float));
    for(int d=0;d<HD;d++)q[d]=(float)nrnd();
    for(int n=0;n<NDL;n++)needle[n]=100+(int)(urand()*(N-400));
    for(int e=0;e<NDEC;e++)decoy[e]=100+(int)(urand()*(N-400));
    int rlo=N-64;
    for(int i=0;i<N;i++){for(int d=0;d<HD;d++)K[(size_t)i*HD+d]=(float)(0.30*nrnd());
        double a=0; if(i>=rlo)a=0.9; for(int n=0;n<NDL;n++)if(i==needle[n])a=1.6;
        if(a>0)for(int d=0;d<HD;d++)K[(size_t)i*HD+d]+=(float)(a*q[d]);
        for(int d=0;d<HD;d++)V[(size_t)i*HD+d]=(float)(0.3*nrnd());
        if(a>0.5)for(int d=0;d<HD;d++)V[(size_t)i*HD+d]+=(float)(a*(d%2?1.0:-1.0));
    }
    /* decoys: each = a random PERMUTATION of needle[e % NDL]'s components (same histogram, scrambled dims) */
    for(int e=0;e<NDEC;e++){int src=needle[e%NDL];float tmp[HD];for(int d=0;d<HD;d++)tmp[d]=K[(size_t)src*HD+d];
        for(int d=HD-1;d>0;d--){int r=(int)(urand()*(d+1));float t=tmp[d];tmp[d]=tmp[r];tmp[r]=t;}
        for(int d=0;d<HD;d++)K[(size_t)decoy[e]*HD+d]=tmp[d];
        /* decoy V = background-ish (so wrongly recalling it pollutes the output) */
        for(int d=0;d<HD;d++)V[(size_t)decoy[e]*HD+d]=(float)(0.3*nrnd());
    }
    double inv=1.0/sqrt((double)HD);
    for(int i=0;i<N;i++){double s=0;const float*k=K+(size_t)i*HD;for(int d=0;d<HD;d++)s+=(double)q[d]*k[d];score[i]=s*inv;}
    {int*all=malloc(N*sizeof(int));for(int i=0;i<N;i++)all[i]=i;attn(all,N,ofull);free(all);}
    {int32_t kq[HD];for(int i=0;i<N;i++){const float*k=K+(size_t)i*HD;for(int d=0;d<HD;d++)kq[d]=(int32_t)lrintf(k[d]*256.0f);sp_kste_encode(kq,HD,&ktree[i]);}
     for(int d=0;d<HD;d++)kq[d]=(int32_t)lrintf(q[d]*256.0f);sp_kste_encode(kq,HD,&qtree);}

    /* sanity: mean score + tier-0 L1 distance to q, for needles vs their decoys */
    double sn=0,sd=0; for(int n=0;n<NDL;n++)sn+=score[needle[n]]; for(int e=0;e<NDEC;e++)sd+=score[decoy[e]];
    long ldn=0; for(int n=0;n<NDL;n++){long d=0;for(int j=0;j<6;j++){long e=(long)t0(&ktree[needle[n]],j)-t0(&qtree,j);d+=e<0?-e:e;}ldn+=d;}
    long ldd=0; for(int e=0;e<NDEC;e++){long d=0;for(int j=0;j<6;j++){long ee=(long)t0(&ktree[decoy[e]],j)-t0(&qtree,j);d+=ee<0?-ee:ee;}ldd+=d;}
    printf("# KSTE recall router — ADVERSARIAL (permuted decoys; same histogram as needles, ~0 dot product)\n");
    printf("# mean TRUE score: needle=%.3f  decoy=%.3f  (decoy should be ~0 = no alignment)\n", sn/NDL, sd/NDEC);
    printf("# mean KSTE tier-0 L1 dist to q: needle=%.1f  decoy=%.1f  (CLOSE => router cannot tell them apart)\n\n", (double)ldn/NDL,(double)ldd/NDEC);

    int Bs[]={64,128,256,512}; int*idx=malloc(N*sizeof(int));
    printf("%6s | %-26s | %-26s\n","B","KSTE-router top-B","ORACLE top-B");
    printf("%6s | %-26s | %-26s\n","","cos  needles  decoys","cos  needles  decoys");
    for(int bi=0;bi<4;bi++){int B=Bs[bi];
        /* KSTE top-B by -L1(t0(K_i),t0(q)) */
        typedef struct{double s;int i;}si; si*a=malloc(N*sizeof(si));
        for(int i=0;i<N;i++){long d=0;for(int j=0;j<6;j++){long e=(long)t0(&ktree[i],j)-t0(&qtree,j);d+=e<0?-e:e;}a[i].s=-(double)d;a[i].i=i;}
        for(int t=0;t<B;t++){int b=t;for(int u=t+1;u<N;u++)if(a[u].s>a[b].s)b=u;si tm=a[t];a[t]=a[b];a[b]=tm;idx[t]=a[t].i;}
        float o[HD];attn(idx,B,o); char c1[64];snprintf(c1,64,"%.4f  %d/%d     %d",cosf2(o),count_in(idx,B,needle,NDL),NDL,count_in(idx,B,decoy,NDEC));
        /* oracle */
        for(int i=0;i<N;i++){a[i].s=score[i];a[i].i=i;}
        for(int t=0;t<B;t++){int b=t;for(int u=t+1;u<N;u++)if(a[u].s>a[b].s)b=u;si tm=a[t];a[t]=a[b];a[b]=tm;idx[t]=a[t].i;}
        attn(idx,B,o); char c2[64];snprintf(c2,64,"%.4f  %d/%d     %d",cosf2(o),count_in(idx,B,needle,NDL),NDL,count_in(idx,B,decoy,NDEC));
        printf("%6d | %-26s | %-26s\n",B,c1,c2); free(a);
    }
    printf("\n# verdict: if KSTE 'decoys' column is high (router pulls many zero-score decoys) and needle\n");
    printf("#          capture is LOW, the order-statistics signature is NOT a directional router.\n");
    return 0;
}
