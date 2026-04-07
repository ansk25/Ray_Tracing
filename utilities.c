#include "utilities.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include "romu.h"

inline double inline_fabs(const double a) {
    return a<0.0 ? -a : a;
}

inline double dot_product(const double x1, const double y1, const double z1, const double x2, const double y2, const double z2) {
    return x1*x2 + y1*y2 + z1*z2;
}

inline bool w_initialization(const double Wy, const double Wmax, double Vx, double Vy, double Vz, double *Wx, double *Wz) {
    const double coeff = Wy/Vy;
    *Wx = coeff*Vx;
    *Wz = coeff*Vz;

    return fabs(*Wx) < Wmax && fabs(*Wz) < Wmax;
    //return inline_fabs(*Wx) < Wmax && inline_fabs(*Wz) < Wmax;
}

inline void i_initialization(const double Vx, const double Vy, const double Vz, double *Ix, double *Iy,double *Iz, const double t) {
    *Ix = t*Vx;
    *Iy = t*Vy;
    *Iz = t*Vz;
}

inline void normalize_vector_n(const double Ix, const double Iy, const double Iz, const double Cx, const double Cy, const double Cz, const double inv_R, double *Nx, double *Ny,double *Nz) {
    const double sub_x = Ix-Cx;
    const double sub_y = Iy-Cy;
    const double sub_z = Iz-Cz;

    *Nx = sub_x*inv_R;
    *Ny = sub_y*inv_R;
    *Nz = sub_z*inv_R;
}

inline void normalize_vector_s( const double Lx, const double Ly, const double Lz, const double Ix, const double Iy, const double Iz, double *Sx, double *Sy,double *Sz) {
    const double sub_x = Lx-Ix;
    const double sub_y = Ly-Iy;
    const double sub_z = Lz-Iz;
    const double norm_val = sqrt(sub_x*sub_x + sub_y*sub_y + sub_z*sub_z);
    const double inv_norm = 1.0 / norm_val;

    *Sx = sub_x*inv_norm;
    *Sy = sub_y*inv_norm;
    *Sz = sub_z*inv_norm;
}

inline double t_real_solution(const double Cx, const double Cy, const double Cz, const double R_sq_minus_CC,const double Vx, const double Vy, const double Vz, double *VC_dot_product) {
    *VC_dot_product = dot_product(Vx,Vy,Vz,Cx,Cy,Cz);
    return (*VC_dot_product)*(*VC_dot_product)+ R_sq_minus_CC;
}

inline double inline_fmax(const double a, const double b) {
    return a>b ?a:b;
}

void save_grid_dat(const char *filename,
                   double **grid,
                   const int n,
                   int Nrays)
{
    FILE *f = fopen(filename, "wb");
    if (!f) {
        perror("fopen");
        return;
    }

    int32_t N = (int32_t) Nrays;
    int32_t R = (int32_t) n;
    int32_t C = (int32_t) n;

    fwrite(&N, sizeof(int32_t), 1, f);
    fwrite(&R, sizeof(int32_t), 1, f);
    fwrite(&C, sizeof(int32_t), 1, f);

    for (int i = 0; i < n; i++) {
        fwrite(grid[i], sizeof(double), n, f);
    }

    fclose(f);
}

