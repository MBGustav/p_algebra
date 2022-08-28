// Kalman Filter

#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "libmat.hpp"
#include "structs.hpp"
#include "defines.h"
//#include "matrices.hpp"


void ekf_p( wm *wm,  ss_d *ss)
{
        // Declaration of the variables
        Matrix *x, *xp, *Phi, *Q, *P, *Pp;
        Matrix *PhiP, *Phi_T, *PhiPPhi_T;
        int nrPhi, ncPhi, ncP;

        x  = ss->x;
        xp = ss->xp;
        Phi = ss->Phi;
        Q  = wm->Q;
        P = wm->P;
        Pp = wm->Pp;

        nrPhi = ss->Phi->n_row;
        ncPhi = ss->Phi->n_col;
        ncP = wm->P->n_col;

        *PhiP = Matrix(nrPhi, ncP);
        *Phi_T = Matrix(ncPhi, nrPhi);
        *PhiPPhi_T = Matrix(nrPhi, nrPhi);

        times_m(*Phi,*x,*xp);
        times_m(*Phi,*P,*PhiP);
        transp_m(*Phi,*Phi_T);
        times_m (*PhiP, *Phi_T, *PhiPPhi_T);
        sum_m(*PhiPPhi_T,*Q,*Pp);

        delete PhiP;
        delete Phi_T;
        delete PhiPPhi_T;
}

void ekf_u(struct wm *wm, struct ss_d *ss) {

        Matrix *H, *P, *Pp, *R, *x, *xp, *z;
        Matrix *h;
        Matrix *deltaz;
        Matrix *H_T, *PpH_T, *HPpH_T, *M, *invM, *K, *I, *KH, *IKH, *Kdeltaz;

        int nrz, ncz, nrH, ncH, nrR, ncR, nrPp, ncPp;

        H = ss->H;
        nrH = ss->H->n_row;
        ncH = ss->H->n_col;

        R = wm->R;
        nrR = wm->R->n_row;
        ncR = wm->R->n_col;

        P = wm->P;
        Pp = wm->Pp;
        nrPp = wm->Pp->n_row;
        ncPp = wm->Pp->n_col;

        x = ss->x;
        xp = ss->xp;

        z = ss->z;
        nrz = ss->z->n_row;
        ncz = ss->z->n_col;

        *h =  Matrix(nrz,ncz);
        *deltaz = Matrix(nrz,ncz);
        *H_T = Matrix(ncH, nrH);
        *PpH_T = Matrix(nrPp,nrH);
        *HPpH_T = Matrix(nrH, nrH);
        *M = Matrix(nrR,ncR);
        *invM = Matrix(nrR,ncR);
        *K = Matrix(nrPp, ncR);
        *I = Matrix(nrPp,ncPp);
        *KH = Matrix(nrPp,ncH);
        *IKH = Matrix(nrPp,ncH);
        *Kdeltaz = Matrix(nrPp,ncz);
        transp_m(*H,*H_T);
        times_m(*Pp,*H_T, *PpH_T);
        times_m(*H,*PpH_T, *HPpH_T);
        sum_m(*HPpH_T, *R, *M);
        inv_m(*M,*invM);
        times_m(*PpH_T,*invM,*K);
        I->eye();
        times_m(*K,*H,*KH);
        less_m(*I,*KH,*IKH);
        times_m(*IKH,*Pp,*P);

        update_z(x->elem,h->elem);
        less_m(*z,*h,*deltaz);

        times_m(*K,*deltaz,*Kdeltaz);
        sum_m(*xp,*Kdeltaz,*x);

        delete h;
        delete deltaz;
        delete H_T;
        delete PpH_T;
        delete HPpH_T;
        delete M;
        delete invM;
        delete K;
        delete I;
        delete KH;
        delete IKH;
        delete Kdeltaz;
}

void ekf_p_array(struct ss_d *ss) {
        // Declaration of the variables
        Matrix *x, *xp, *Phi;
        int nrPhi, ncPhi;

        x  = ss->x;
        xp = ss->xp;
        Phi = ss->Phi;

        nrPhi = ss->Phi->n_row;
        ncPhi = ss->Phi->n_col;

        times_m(*Phi,*x,*xp);
}

void ekf_u_array(struct wm *wm, struct ss_d *ss) {

        Matrix *Phi, *G, *H, *P, *P_sr, *Q_sr, *R_sr, *x, *xp, *z;
        Matrix *pre_array, *pos_array;
        Matrix *h;
        Matrix *deltaz;
        Matrix *HPhi, *HPhiP_sr, *HG, *HGQ_sr, *PhiP_sr, *GQ_sr;
        Matrix *ZEROS;

        int nrz, ncz, nrPhi, ncPhi, nrG, ncG, nrH, ncH;
        int nrQ_sr, ncQ_sr, nrR_sr, ncR_sr, nrP, ncP;

        int i,j;
        Phi = ss->Phi;
        nrPhi = ss->Phi->n_row;
        ncPhi = ss->Phi->n_col;

        G = ss->G;
        nrG = ss->G->n_row;
        ncG = ss->G->n_col;

        H = ss->H;
        nrH = ss->H->n_row;
        ncH = ss->H->n_col;

        Q_sr = wm->Qeta_sr;
        nrQ_sr = wm->Qeta_sr->n_row;
        ncQ_sr = wm->Qeta_sr->n_col;

        R_sr = wm->R_sr;
        nrR_sr = wm->R_sr->n_row;
        ncR_sr = wm->R_sr->n_col;

        P = wm->P;
        nrP = wm->P->n_row;
        ncP = wm->P->n_col;

        P_sr = wm->P_sr;


        x = ss->x;
        xp = ss->xp;

        z = ss->z;
        nrz = ss->z->n_row;
        ncz = ss->z->n_col;

        HPhi = Matrix(nrH,ncPhi);
        HPhiP_sr = Matrix(nrH,ncP);
        HG = Matrix(nrH, ncG);
        HGQ_sr = Matrix(nrH, ncQ_sr);
        ZEROS  = Matrix(nrPhi,ncR_sr);
        PhiP_sr = Matrix(nrPhi, ncP);
        GQ_sr = Matrix(nrG, ncQ_sr);
        pre_array =  Matrix((nrR_sr+nrPhi), (ncR_sr + ncP+ ncQ_sr));

        eye_m(0,ZEROS);
        equalpart_m(pre_array,0,0,R_sr);
        equalpart_m(pre_array,0,ncR_sr,HPhiP_sr);
        equalpart_m(pre_array,0,(ncR_sr+ncP),HGQ_sr);

        equalpart_m(pre_array,nrR_sr,0,ZEROS);
        equalpart_m(*pre_array,*nrR_sr,ncR_sr,*PhiP_sr);
        equalpart_m(*pre_array,*nrR_sr,(ncR_sr+ncP),*GQ_sr);

        givens_m(*pre_array,*pos_array);
        getpart_m(*pos_array,*ncR_sr,nrP,*P_sr);

        times_m(*P_sr,*P_sr,*P);

        delete HPhi;
        delete HPhiP_sr;
        delete HG;
        delete HGQ_sr;
        delete ZEROS ;
        delete PhiP_sr;
        delete GQ_sr;
        delete pre_array;
        delete pos_array;

}

