/* c2_router_proj.c — recall-router shootout: KSTE (order-stats) vs ±1 RANDOM
 * PROJECTION (JL, integer/discrete) vs ORACLE, on the adversarial needle+decoy
 * set that FALSIFIED the KSTE router (c2_kste_router_adv.c).
 *
 * The recall problem: pick a budget B of the cached tokens that reproduces full
 * attention. KSTE failed because order-statistics are a magnitude signature,
 * blind to direction → it can't tell a q-aligned needle from a permuted decoy
 * with the same histogram but ~0 dot. A random projection PRESERVES the dot
 * (Johnson-Lindenstrauss), and a ±1 (Rademacher) projection does so in INTEGER
 * arithmetic — discrete/Z_q-native, no float "tax". Test: does a ±1 proj of rank
 * r get 8/8 needles AND reject the decoys, where KSTE got 6/8 + 21 decoys?
 *
 * Storage cost per token: r int16 (r=32 → 64 B, same as the KSTE signature).
 */
#include "sp/kste.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

static uint64_t s0=0x2545F4914F6CDD1DULL,s1=0x9E3779B97F4A7C15ULL;
static double urand(void){uint64_t x=s0,y=s1;s0=y;x^=x<<23;s1=x^y^(x>>17)^(y>>26);return (double)((s1+y)>>11)/9007199254740992.0;}
static double nrnd(void){double u=urand(),v=urand();if(u<1e-12)u=1e-12;return sqrt(-2.0*log(u))*cos(6.283185307179586*v);}

#define N 4096
#define HD 128
#define NDL 8
#define NDEC 32
#define RMAX 64

static float *K,*V,q[HD],ofull[HD];
static double score[N];
static int needle[NDL], decoy[NDEC];
static sp_kste_tree_t ktree[N], qtree;
static int16_t t0(const sp_kste_tree_t*t,int j){int o=8+2*j;return (int16_t)((uint8_t)t->bytes[o]|((uint8_t)t->bytes[o+1]<<8));}

static void attn(const int*idx,int m,float*o){double mx=-1e300;for(int j=0;j<m;j++)if(score[idx[j]]>mx)mx=score[idx[j]];double Z=0;static double p[N];for(int j=0;j<m;j++){p[j]=exp(score[idx[j]]-mx);Z+=p[j];}for(int d=0;d<HD;d++)o[d]=0;for(int j=0;j<m;j++){double w=p[j]/Z;const float*v=V+(size_t)idx[j]*HD;for(int d=0;d<HD;d++)o[d]+=(float)(w*v[d]);}}
static double cosf2(const float*o){double dt=0,na=0,nb=0;for(int d=0;d<HD;d++){dt+=(double)o[d]*ofull[d];na+=(double)o[d]*o[d];nb+=(double)ofull[d]*ofull[d];}return (na>0&&nb>0)?dt/(sqrt(na)*sqrt(nb)):0;}
static int cnt(const int*idx,int m,const int*set,int ns){int c=0;for(int s=0;s<ns;s++)for(int j=0;j<m;j++)if(idx[j]==set[s]){c++;break;}return c;}
typedef struct{double s;int i;}si;
static void topB(si*a,int B,int*idx){for(int t=0;t<B;t++){int b=t;for(int u=t+1;u<N;u++)if(a[u].s>a[b].s)b=u;si tm=a[t];a[t]=a[b];a[b]=tm;idx[t]=a[t].i;}}

/* ±1 Rademacher projection: R is r×HD of ±1; proj = R·vec (integer-ish in float). */
static signed char R[RMAX][HD];
static void build_R(int r){for(int p=0;p<r;p++)for(int d=0;d<HD;d++)R[p][d]=(urand()<0.5)?-1:1;}
static void project(const float*vec,int r,float*out){for(int p=0;p<r;p++){double a=0;for(int d=0;d<HD;d++)a+=R[p][d]*vec[d];out[p]=(float)a;}}

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
        if(a>0.5)for(int d=0;d<HD;d++)V[(size_t)i*HD+d]+=(float)(a*(d%2?1.0:-1.0)); }
    for(int e=0;e<NDEC;e++){int src=needle[e%NDL];float tmp[HD];for(int d=0;d<HD;d++)tmp[d]=K[(size_t)src*HD+d];
        for(int d=HD-1;d>0;d--){int rr=(int)(urand()*(d+1));float t=tmp[d];tmp[d]=tmp[rr];tmp[rr]=t;}
        for(int d=0;d<HD;d++)K[(size_t)decoy[e]*HD+d]=tmp[d];
        for(int d=0;d<HD;d++)V[(size_t)decoy[e]*HD+d]=(float)(0.3*nrnd()); }
    double inv=1.0/sqrt((double)HD);
    for(int i=0;i<N;i++){double s=0;const float*k=K+(size_t)i*HD;for(int d=0;d<HD;d++)s+=(double)q[d]*k[d];score[i]=s*inv;}
    {int*all=malloc(N*sizeof(int));for(int i=0;i<N;i++)all[i]=i;attn(all,N,ofull);free(all);}
    {int32_t kq[HD];for(int i=0;i<N;i++){const float*k=K+(size_t)i*HD;for(int d=0;d<HD;d++)kq[d]=(int32_t)lrintf(k[d]*256.0f);sp_kste_encode(kq,HD,&ktree[i]);}for(int d=0;d<HD;d++)kq[d]=(int32_t)lrintf(q[d]*256.0f);sp_kste_encode(kq,HD,&qtree);}

    int Bs[]={64,128,256,512}; int*idx=malloc(N*sizeof(int)); si*a=malloc(N*sizeof(si));
    printf("# RECALL-ROUTER SHOOTOUT — needle+decoy (KSTE falsified here). cos / needles(of %d) / decoys(of %d)\n\n",NDL,NDEC);
    int rs[]={16,32}; float pj[N][2][RMAX]; float pq[2][RMAX];
    for(int ri=0;ri<2;ri++){build_R(rs[ri]); for(int i=0;i<N;i++)project(K+(size_t)i*HD,rs[ri],pj[i][ri]); project(q,rs[ri],pq[ri]);}
    printf("%6s | %-22s | %-22s | %-22s | %-18s\n","B","KSTE(64B)","±1 PROJ r=16(32B)","±1 PROJ r=32(64B)","ORACLE");
    for(int bi=0;bi<4;bi++){int B=Bs[bi]; char c[4][48];
        for(int i=0;i<N;i++){long d=0;for(int j=0;j<6;j++){long e=(long)t0(&ktree[i],j)-t0(&qtree,j);d+=e<0?-e:e;}a[i].s=-(double)d;a[i].i=i;}
        topB(a,B,idx);{float o[HD];attn(idx,B,o);snprintf(c[0],48,"%.4f %d/%d %d",cosf2(o),cnt(idx,B,needle,NDL),NDL,cnt(idx,B,decoy,NDEC));}
        for(int ri=0;ri<2;ri++){for(int i=0;i<N;i++){double dp=0;for(int p=0;p<rs[ri];p++)dp+=pq[ri][p]*pj[i][ri][p];a[i].s=dp;a[i].i=i;}topB(a,B,idx);float o[HD];attn(idx,B,o);snprintf(c[1+ri],48,"%.4f %d/%d %d",cosf2(o),cnt(idx,B,needle,NDL),NDL,cnt(idx,B,decoy,NDEC));}
        for(int i=0;i<N;i++){a[i].s=score[i];a[i].i=i;}topB(a,B,idx);{float o[HD];attn(idx,B,o);snprintf(c[3],48,"%.4f %d/%d %d",cosf2(o),cnt(idx,B,needle,NDL),NDL,cnt(idx,B,decoy,NDEC));}
        printf("%6d | %-22s | %-22s | %-22s | %-18s\n",B,c[0],c[1],c[2],c[3]);
    }
    printf("\n# win = needles=%d/%d + few decoys + cos~1.0. JL projection should reject decoys (preserves dot); KSTE can't.\n",NDL,NDL);
    return 0;
}
