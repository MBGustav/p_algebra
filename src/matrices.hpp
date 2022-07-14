// Prototypes of matrices.c
#include "structs.hpp"

void updateOmega (Matrix &omega, Matrix &Omega);
void update_q(Matrix &omega, Matrix &q);
void update_rot(Matrix &quat, Matrix &Rot);
void update_z(DVector &q, DVector &z);
void sim_atitude (Matrix &omega, Matrix &q_sim, struct ss_d *sys_a);
void update_sys_a(Matrix &omega_hat, struct ss_d *sys_a);
void updadate_wm_a(struct ss_d *ss_a, struct wm *wm_a);



