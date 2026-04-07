#include "float_utilities.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include "romu.h"

inline float inline_fabs(const float a) {
    return a<0.0 ? -a : a;
}

inline float dot_product(const float x1, const float y1, const float z1, const float x2, const float y2, const float z2) {
    return x1*x2 + y1*y2 + z1*z2;
}

inline bool w_initialization(const float Wy, const float Wmax, float Vx, float Vy, float Vz, float *Wx, float *Wz) {
    const float coeff = Wy/Vy;
    *Wx = coeff*Vx;
    *Wz = coeff*Vz;

    return fabs(*Wx) < Wmax && fabs(*Wz) < Wmax;
    //return inline_fabs(*Wx) < Wmax && inline_fabs(*Wz) < Wmax;
}

inline void i_initialization(const float Vx, const float Vy, const float Vz, float *Ix, float *Iy,float *Iz, const float t) {
    *Ix = t*Vx;
    *Iy = t*Vy;
    *Iz = t*Vz;
}

inline void normalize_vector_n(const float Ix, const float Iy, const float Iz, const float Cx, const float Cy, const float Cz, const float inv_R, float *Nx, float *Ny,float *Nz) {
    const float sub_x = Ix-Cx;
    const float sub_y = Iy-Cy;
    const float sub_z = Iz-Cz;

    *Nx = sub_x*inv_R;
    *Ny = sub_y*inv_R;
    *Nz = sub_z*inv_R;
}

inline void normalize_vector_s( const float Lx, const float Ly, const float Lz, const float Ix, const float Iy, const float Iz, float *Sx, float *Sy,float *Sz) {
    const float sub_x = Lx-Ix;
    const float sub_y = Ly-Iy;
    const float sub_z = Lz-Iz;
    const float norm_val = sqrt(sub_x*sub_x + sub_y*sub_y + sub_z*sub_z);
    const float inv_norm = 1.0 / norm_val;

    *Sx = sub_x*inv_norm;
    *Sy = sub_y*inv_norm;
    *Sz = sub_z*inv_norm;
}

inline float t_real_solution(const float Cx, const float Cy, const float Cz, const float R_sq_minus_CC,const float Vx, const float Vy, const float Vz, float *VC_dot_product) {
    *VC_dot_product = dot_product(Vx,Vy,Vz,Cx,Cy,Cz);
    return (*VC_dot_product)*(*VC_dot_product)+ R_sq_minus_CC;
}

inline float inline_fmax(const float a, const float b) {
    return a>b ?a:b;
}

void save_grid_dat(const char *filename,
                   float **grid,
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
        fwrite(grid[i], sizeof(float), n, f);
    }

    fclose(f);
}

