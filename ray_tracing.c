#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <stdint.h>

const double INV_2_53 = 1.0 / 9007199254740992.0;

#include <stdint.h>

/* This is a fixed-increment version of Java 8's SplittableRandom generator
   See http://dx.doi.org/10.1145/2714064.2660195 and
   http://docs.oracle.com/javase/8/docs/api/java/util/SplittableRandom.html

   It is a very fast generator passing BigCrush, and it can be useful if
   for some reason you absolutely want 64 bits of state. */

static uint64_t x; /* The state can be seeded with any value. */

void split64_seed(uint64_t seed)
{
    x=seed;
}

uint64_t next() {
    uint64_t z = (x += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

#define ROTL(d,lrot) ((d<<(lrot)) | (d>>(8*sizeof(d)-(lrot))))

uint64_t xState, yState, zState;

inline uint64_t romuTrio_random(void)
{
    const uint64_t xp = xState;
    const uint64_t yp = yState;
    const uint64_t zp = zState;
    xState = 15241094284759029579u * zp;
    yState = yp - xp;  yState = ROTL(yState, 12);
    zState = zp - yp;  zState = ROTL(zState, 44);
    return xp;
}

// void romuTrio_seed(uint64_t seed)
void romuTrio_seed()
{
    // uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
    // z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    // z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    // xState = z ^ (z >> 31);
    //
    // z += 0x9E3779B97F4A7C15ULL;
    // z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    // z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    // yState = z ^ (z >> 31);
    //
    // z += 0x9E3779B97F4A7C15ULL;
    // z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    // z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    // zState = z ^ (z >> 31);


    xState = next();
    yState = next();
    zState = next();


    if (!xState) xState = 1;
    if (!yState) yState = 2;
    if (!zState) zState = 3;
}

static inline double romuTrio_double(void)
{
    return (romuTrio_random() >> 11) * INV_2_53;
}


double Vx, Vy, Vz;
double Wx, Wz;
double Zx, Zy, Zz;
double Ix, Iy, Iz;
double Nx,Ny,Nz;
double Sx,Sy,Sz;
double VC_dot_product;
long rejectedRays =0;
double t;


inline double inline_fabs(const double a) {
        return a<0.0 ? -a : a;
}

inline double dot_product(const double x1, const double y1, const double z1, const double x2, const double y2, const double z2) {
    return x1*x2 + y1*y2 + z1*z2;
}

inline bool w_initialization(const double Wy, const double Wmax) {
    const double coeff = Wy/Vy;
    Wx = coeff*Vx;
    Wz = coeff*Vz;

    // return fabs(Wx) < Wmax && fabs(Wz) < Wmax;
    return inline_fabs(Wx) < Wmax && inline_fabs(Wz) < Wmax;
}

inline void i_initialization() {
    Ix = t*Vx;
    Iy = t*Vy;
    Iz = t*Vz;
}

inline double norm(const double x1, const double y1, const double z1) {
    return sqrt(x1*x1 + y1*y1 + z1*z1);
}

inline void normalize_vector_n(const double x2, const double y2, const double z2, const double x3, const double y3, const double z3, const double inv_R) {
    const double sub_x = x2-x3;
    const double sub_y = y2-y3;
    const double sub_z = z2-z3;

    Nx = sub_x*inv_R;
    Ny = sub_y*inv_R;
    Nz = sub_z*inv_R;
}

inline void normalize_vector_s( const double x2, const double y2, const double z2, const double x3, const double y3, const double z3) {
    const double sub_x = x2-x3;
    const double sub_y = y2-y3;
    const double sub_z = z2-z3;
    // const double norm_val = norm(sub_x,sub_y,sub_z);
    const double norm_val = sqrt(sub_x*sub_x + sub_y*sub_y + sub_z*sub_z);
    const double inv_norm = 1.0 / norm_val;

    Sx = sub_x*inv_norm;
    Sy = sub_y*inv_norm;
    Sz = sub_z*inv_norm;
}

inline void v_sampling() {

    // const double phi = pcg32_double(&rng)*M_PI-0;
    // const double cos_theta = pcg32_double(&rng)*2-1;
    const double phi = (romuTrio_random()>>11)*INV_2_53*M_PI;
    const double cos_theta = (romuTrio_random()>>11) * INV_2_53*2-1;
    const double sin_theta = sqrt(1-cos_theta*cos_theta);
    Vy = sin_theta*sin(phi);
    Vx = sin_theta*cos(phi);
    Vz = cos_theta;
}

inline double t_real_solution(const double Cx, const double Cy, const double Cz, const double R_sq_minus_CC) {
    VC_dot_product = dot_product(Vx,Vy,Vz,Cx,Cy,Cz);
    return VC_dot_product*VC_dot_product + R_sq_minus_CC;
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


int main(int argc, char *argv[]) {
    const double Lx = strtod(argv[1], NULL);
    const double Ly = strtod(argv[2], NULL);
    const double Lz = strtod(argv[3], NULL);
    const double Wy = strtod(argv[4], NULL);
    const double Wmax = strtod(argv[5], NULL);
    const double Cx = strtod(argv[6], NULL);
    const double Cy = strtod(argv[7], NULL);
    const double Cz = strtod(argv[8], NULL);
    const double R = strtod(argv[9], NULL);
    const int n = atoi(argv[10]);
    const long Nrays = atol(argv[11]);

    const double inv_delta = (double)n/(2.0*Wmax);;

    double **grid;
    double *rows;
    grid = (double**)malloc(n*sizeof(double*));
    rows = (double*)malloc(n*n*sizeof(double));
    for (int i=0; i<n; i++) {
        grid[i] = rows + i*n;
    }
    for (int i=0; i<n*n; i++) {
        rows[i] = 0;
    }
    // pcg32_srandom(&rng, 42u, 54u);
    split64_seed(1337ULL);
    romuTrio_seed();

    const double CC_dot_product = dot_product(Cx,Cy,Cz,Cx,Cy,Cz);
    const double R_sq = R*R;
    const double R_Sq_minus_CC = R_sq-CC_dot_product;
    const double inv_R = 1/R;

    clock_t start = clock();
    for (long l=1; l<=Nrays; l++) {
        while (true) {
            v_sampling();
            if (Vy<=0) {
                rejectedRays++;
                continue;
            }
            if (!w_initialization(Wy, Wmax)) {
                rejectedRays++;
                continue;
            }
            VC_dot_product = dot_product(Vx,Vy,Vz,Cx,Cy,Cz);
            const double t_term = VC_dot_product*VC_dot_product + R_Sq_minus_CC;
            if (t_term<0) {
                rejectedRays++;
                continue;
            }
            t = VC_dot_product - sqrt(t_term);
            break;
        }

        // i_initialization();
        Ix = t*Vx;
        Iy = t*Vy;
        Iz = t*Vz;
        normalize_vector_n( Ix, Iy, Iz, Cx, Cy, Cz,inv_R);

        normalize_vector_s(Lx,Ly,Lz,Ix, Iy, Iz);

        double const SN_dot_product = dot_product(Sx,Sy,Sz,Nx,Ny,Nz);

        const double b = fmax(SN_dot_product, 0.0);

        const int i = (int)((Wx + Wmax)*inv_delta);
        const int j = (int)((Wz + Wmax)*inv_delta);
        // if (i >= 0 && i < n && j >= 0 && j < n) {
            // grid[i][j] += b;
        *(grid[i] + j) +=b;
        // rows[i*n+j] +=b;
        // }
    }

    clock_t end = clock();

    save_grid_dat("sphere_serial.bin", grid, n, Nrays);

    double time_spent = (double)(end-start)/CLOCKS_PER_SEC;

    size_t memory_grids = (size_t) 2*n*n*sizeof(double);
    double total_memory_mb = memory_grids/(1024*1024);

    printf("Time taken = %.6lf\n", time_spent);
    printf("Total memory MB = %.6lf\n", total_memory_mb);
    printf("Rejected rays = %ld\n", rejectedRays);

    free(grid);
    free(rows);
}