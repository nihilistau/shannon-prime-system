/* kste_md_realdata.c — the EXPENSIVE test: does the v2 encoder's synthetic 37.6x
 * discrimination survive REAL gemma-4-12B global-Q vectors?
 *
 * Uses the on-disk faithfulness captures (exact-query global-Q vs paraphrase-query
 * global-Q for the same 61 facts). Tests the OPEN problem: same-fact/different-
 * surface should cluster; different-fact should separate. Head-to-head:
 *   - raw cosine on the 65536-d float vectors (dense gold-standard similarity)
 *   - KSTE-MD signature L1 distance (v2)
 *   - KSTE v1 tree (order-stat) via #comparable tiers
 *
 * File format (per _ln1_build_npz.py): <u32 ng><u32 d><f32 payload...>; we read
 * all floats after the 8-byte header. Alignment: sort each dir's q_*.bin by numeric
 * cid; index i = fact i (same convention the LN-1 adapter used).
 *
 * argv[1]=exact qdump dir, argv[2]=paraphrase qdump dir
 * Build: gcc -O2 -std=c11 -Wall -Iinclude tests/kste_md_realdata.c \
 *        core/kste_md/kste_md.c core/kste/kste_encode.c core/kste/kste_dominance.c -o rd -lm
 */
#include "sp/kste_md.h"
#include "sp/kste.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <dirent.h>

#define MAXV 128
typedef struct { long cid; char path[1024]; } ent_t;

static int by_cid(const void*a,const void*b){ long x=((const ent_t*)a)->cid,y=((const ent_t*)b)->cid; return (x<y)?-1:(x>y)?1:0; }

static int list_q(const char*dir, ent_t*out, int cap){
    DIR*d=opendir(dir); if(!d) return -1; struct dirent*e; int n=0;
    while((e=readdir(d))&&n<cap){ const char*nm=e->d_name;
        if(strncmp(nm,"q_",2)==0 && strstr(nm,".bin")){ long cid=atol(nm+2);
            out[n].cid=cid; snprintf(out[n].path,sizeof(out[n].path),"%s/%s",dir,nm); n++; } }
    closedir(d); qsort(out,n,sizeof(ent_t),by_cid); return n;
}
static float* load_floats(const char*path, long*count){
    FILE*f=fopen(path,"rb"); if(!f){*count=0;return NULL;}
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    uint32_t hdr[2]; if(fread(hdr,4,2,f)!=2){fclose(f);*count=0;return NULL;}
    long nf=(sz-8)/4; float*buf=malloc((size_t)nf*sizeof(float));
    if(fread(buf,4,(size_t)nf,f)!=(size_t)nf){free(buf);fclose(f);*count=0;return NULL;}
    fclose(f); *count=nf; return buf;
}
static double cosd(const float*a,const float*b,long n){ double d=0,na=0,nb=0; for(long i=0;i<n;i++){d+=(double)a[i]*b[i];na+=(double)a[i]*a[i];nb+=(double)b[i]*b[i];} return d/(sqrt(na*nb)+1e-30); }
static void to_i32(const float*a,long n,int32_t*o){ long m=n<65536?n:65536; for(long i=0;i<m;i++){ double v=(double)a[i]*1e6; if(v>2.147e9)v=2.147e9; if(v<-2.147e9)v=-2.147e9; o[i]=(int32_t)v; } }
static long md_l1(const sp_kste_md_sig_t*a,const sp_kste_md_sig_t*b){ long s=0; for(int i=0;i<SP_KMD_SIGDIM;i++){ long d=(long)a->v[i]-b->v[i]; s+= d<0?-d:d; } return s; }

int main(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: %s <exact_dir> <para_dir>\n",argv[0]); return 2; }
    ent_t ex[MAXV], pa[MAXV];
    int ne=list_q(argv[1],ex,MAXV), np=list_q(argv[2],pa,MAXV);
    if(ne<=0||np<=0){ fprintf(stderr,"no q_*.bin found (ne=%d np=%d)\n",ne,np); return 1; }
    int n = ne<np?ne:np;
    printf("=== G-KSTE-MD-REALDATA : real gemma4-12B global-Q, exact vs paraphrase (n=%d) ===\n",n);

    static float *E[MAXV],*P[MAXV]; long nf=0, dim=0;
    static sp_kste_md_sig_t Es[MAXV],Ps[MAXV]; static sp_kste_tree_t Et[MAXV],Pt[MAXV];
    static int32_t qbuf[65536];
    for(int i=0;i<n;i++){
        long ce,cp; E[i]=load_floats(ex[i].path,&ce); P[i]=load_floats(pa[i].path,&cp);
        if(!E[i]||!P[i]||ce!=cp){ fprintf(stderr,"load/dim mismatch at %d (ce=%ld cp=%ld)\n",i,ce,cp); return 1; }
        if(dim==0){dim=ce;nf=ce;}
        to_i32(E[i],ce,qbuf); sp_kste_md_encode(qbuf,(int)(ce<65536?ce:65536),&Es[i]); sp_kste_encode(qbuf,(int)(ce<65536?ce:65536),&Et[i]);
        to_i32(P[i],cp,qbuf); sp_kste_md_encode(qbuf,(int)(cp<65536?cp:65536),&Ps[i]); sp_kste_encode(qbuf,(int)(cp<65536?cp:65536),&Pt[i]);
    }
    printf("loaded %d exact + %d para, dim=%ld floats each\n\n",n,n,dim);

    /* recall@1: for each exact_i, which para_j is 'closest'? correct iff j==i */
    int rc_cos=0, rc_md=0;
    double cos_same=0, cos_diff=0; long ns=0,nd=0;
    for(int i=0;i<n;i++){
        int best_cos=-1; double bc=-1e9;
        int best_md=-1; long bl=(long)1e18;
        for(int j=0;j<n;j++){
            double c=cosd(E[i],P[j],nf); if(c>bc){bc=c;best_cos=j;}
            long l=md_l1(&Es[i],&Ps[j]); if(l<bl){bl=l;best_md=j;}
            if(i==j){cos_same+=c;ns++;} else {cos_diff+=c;nd++;}
        }
        if(best_cos==i)rc_cos++; if(best_md==i)rc_md++;
    }
    /* KSTE-MD 'would-dedup' discrimination ratio (comparable = Dickson non-incomparable) */
    long i_c=0,i_n=0,e_c=0,e_n=0, v1_ic=0,v1_in=0,v1_ec=0,v1_en=0;
    for(int i=0;i<n;i++) for(int j=0;j<n;j++){
        int mdc = sp_kste_md_dom(&Es[i],&Ps[j])!=SP_INCOMPARABLE;
        int v1c = sp_kste_tier0(&Et[i],&Pt[j])!=SP_INCOMPARABLE && sp_kste_tier1(&Et[i],&Pt[j])!=SP_INCOMPARABLE;
        if(i==j){ i_n++; if(mdc)i_c++; v1_in++; if(v1c)v1_ic++; }
        else    { e_n++; if(mdc)e_c++; v1_en++; if(v1c)v1_ec++; }
    }
    printf("RECALL@1 (exact query i -> its own paraphrase i, over %d candidates):\n",n);
    printf("  raw cosine  : %d/%d = %.1f%%\n", rc_cos,n,100.0*rc_cos/n);
    printf("  KSTE-MD (v2): %d/%d = %.1f%%\n", rc_md,n,100.0*rc_md/n);
    printf("  random floor: %.1f%%\n\n",100.0/n);
    printf("SIMILARITY separation (cosine): same-fact mean=%.4f  diff-fact mean=%.4f  gap=%.4f\n",
           cos_same/ns, cos_diff/nd, cos_same/ns-cos_diff/nd);
    printf("KSTE-MD 'would-dedup' rate: intra(same)= %.3f (%ld/%ld)  inter(diff)= %.3f (%ld/%ld)  ratio=%.2fx\n",
           (double)i_c/i_n,i_c,i_n, (double)e_c/e_n,e_c,e_n, e_c? ((double)i_c/i_n)/((double)e_c/e_n):0.0);
    printf("KSTE v1 'would-dedup' rate: intra= %.3f  inter= %.3f  ratio=%.2fx\n",
           (double)v1_ic/v1_in,(double)v1_ec/v1_en, v1_ec?((double)v1_ic/v1_in)/((double)v1_ec/v1_en):0.0);
    printf("\n(interpretation: high recall@1 + ratio>>1 => the win transfers to real vectors;\n");
    printf(" near-floor recall + ratio~1 => the encoder inherits the content-poverty of last-token global-Q)\n");
    return 0;
}
