/* kste_md_confirm.c — confirm KSTE-MD amplifies the winning L5 query representation.
 * Reads two <u32 n><u32 dim><i32 data[n*dim]> files (exact, para) produced by
 * rep_layer.py (best-layer mean-heads, scaled int32). Measures exact_i->para_i
 * recall@1 via KSTE-MD signature L1 distance + Dickson would-dedup ratio, with
 * cosine on the same int32 rows as reference.
 * Build: gcc -O2 -std=c11 -Iinclude tests/kste_md_confirm.c core/kste_md/kste_md.c -o kc -lm
 */
#include "sp/kste_md.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

static int32_t* loadi(const char*p,int*n,int*d){
    FILE*f=fopen(p,"rb"); if(!f)return NULL; uint32_t h[2];
    if(fread(h,4,2,f)!=2){fclose(f);return NULL;} *n=(int)h[0];*d=(int)h[1];
    int32_t*a=malloc((size_t)(*n)*(*d)*sizeof(int32_t));
    if(fread(a,4,(size_t)(*n)*(*d),f)!=(size_t)(*n)*(*d)){free(a);fclose(f);return NULL;} fclose(f); return a;
}
static double cosd(const int32_t*a,const int32_t*b,int d){ double dp=0,na=0,nb=0; for(int i=0;i<d;i++){dp+=(double)a[i]*b[i];na+=(double)a[i]*a[i];nb+=(double)b[i]*b[i];} return dp/(sqrt(na*nb)+1e-30); }
static long l1(const sp_kste_md_sig_t*a,const sp_kste_md_sig_t*b){ long s=0; for(int i=0;i<SP_KMD_SIGDIM;i++){long q=(long)a->v[i]-b->v[i]; s+=q<0?-q:q;} return s; }

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: %s exact.i32 para.i32\n",argv[0]);return 2;}
    int ne,de,npv,dp; int32_t*E=loadi(argv[1],&ne,&de),*P=loadi(argv[2],&npv,&dp);
    if(!E||!P||ne!=npv||de!=dp){fprintf(stderr,"load/shape err\n");return 1;}
    int n=ne,d=de; printf("=== KSTE-MD CONFIRM on winning-layer rep  n=%d dim=%d ===\n",n,d);
    sp_kste_md_sig_t *Es=malloc(n*sizeof(*Es)),*Ps=malloc(n*sizeof(*Ps));
    for(int i=0;i<n;i++){ sp_kste_md_encode(E+(size_t)i*d,d,&Es[i]); sp_kste_md_encode(P+(size_t)i*d,d,&Ps[i]); }
    int rc_cos=0,rc_md=0; long ic=0,in=0,ec=0,en=0;
    for(int i=0;i<n;i++){
        int bc=-1,bm=-1; double bcv=-1e9; long bmv=(long)1e18;
        for(int j=0;j<n;j++){
            double c=cosd(E+(size_t)i*d,P+(size_t)j*d,d); if(c>bcv){bcv=c;bc=j;}
            long l=l1(&Es[i],&Ps[j]); if(l<bmv){bmv=l;bm=j;}
            int cmp=sp_kste_md_dom(&Es[i],&Ps[j])!=SP_INCOMPARABLE;
            if(i==j){in++; if(cmp)ic++;} else {en++; if(cmp)ec++;}
        }
        if(bc==i)rc_cos++;
        if(bm==i)rc_md++;
    }
    printf("recall@1 exact->its-paraphrase (n=%d, floor %.1f%%):\n",n,100.0/n);
    printf("  cosine(int32 rows) : %d/%d = %.1f%%\n",rc_cos,n,100.0*rc_cos/n);
    printf("  KSTE-MD (L1 of sig): %d/%d = %.1f%%\n",rc_md,n,100.0*rc_md/n);
    printf("KSTE-MD would-dedup ratio: intra %.3f (%ld/%ld)  inter %.3f (%ld/%ld)  ratio %.2fx\n",
           (double)ic/in,ic,in,(double)ec/en,ec,en, ec?((double)ic/in)/((double)ec/en):0.0);
    return 0;
}
