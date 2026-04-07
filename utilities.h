
#ifndef PROJECT_2_WINTER_2026_AGNES02_GIF_UTILITIES_H
#define PROJECT_2_WINTER_2026_AGNES02_GIF_UTILITIES_H
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

 double inline_fabs(const double a);

 double dot_product(const double x1, const double y1, const double z1, const double x2, const double y2, const double z2);

 bool w_initialization(const double Wy, const double Wmax, double Vx, double Vy, double Vz, double *Wx, double *Wz);

 void i_initialization(double Vx, double Vy, double Vz, double *Ix, double *Iy,double *Iz, const double t);

 void normalize_vector_n(const double Ix, const double Iy, const double Iz, const double Cx, const double Cy, const double Cz, const double inv_R, double *Nx, double *Ny,double *Nz);

 void normalize_vector_s( const double Lx, const double Ly, const double Lz, const double Ix, const double Iy, const double Iz, double *Sx, double *Sy,double *Sz);

 double t_real_solution(const double Cx, const double Cy, const double Cz, const double R_sq_minus_CC,const double Vx, const double Vy, const double Vz, double *VC_dot_product);

 double inline_fmax(const double a, const double b);

void save_grid_dat(const char *filename,  double **grid, const int n, int Nrays);

#ifdef __cplusplus
}
#endif

#endif