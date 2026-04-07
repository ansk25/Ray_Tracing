
#ifndef PROJECT_2_WINTER_2026_AGNES02_GIF_FLOAT_UTILITIES_H
#define PROJECT_2_WINTER_2026_AGNES02_GIF_FLOAT_UTILITIES_H
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

 float inline_fabs(const float a);

 float dot_product(const float x1, const float y1, const float z1, const float x2, const float y2, const float z2);

 bool w_initialization(const float Wy, const float Wmax, float Vx, float Vy, float Vz, float *Wx, float *Wz);

 void i_initialization(float Vx, float Vy, float Vz, float *Ix, float *Iy,float *Iz, const float t);

 void normalize_vector_n(const float Ix, const float Iy, const float Iz, const float Cx, const float Cy, const float Cz, const float inv_R, float *Nx, float *Ny,float *Nz);

 void normalize_vector_s( const float Lx, const float Ly, const float Lz, const float Ix, const float Iy, const float Iz, float *Sx, float *Sy,float *Sz);

 float t_real_solution(const float Cx, const float Cy, const float Cz, const float R_sq_minus_CC,const float Vx, const float Vy, const float Vz, float *VC_dot_product);

 float inline_fmax(const float a, const float b);

void save_grid_dat(const char *filename,  float **grid, const int n, int Nrays);

#ifdef __cplusplus
}
#endif

#endif