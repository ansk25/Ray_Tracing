#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <stdint.h>

const float INV_2_53 = 1.0 / 9007199254740992.0;

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

static inline float romuTrio_double(void)
{
    return (romuTrio_random() >> 11) * INV_2_53;
}


float Vx, Vy, Vz;
float Wx, Wz;
float Zx, Zy, Zz;
float Ix, Iy, Iz;
float Nx,Ny,Nz;
float Sx,Sy,Sz;
float VC_dot_product;
long rejectedRays =0;
float t;


inline float inline_fabs(const float a) {
        return a<0.0 ? -a : a;
}

inline float dot_product(const float x1, const float y1, const float z1, const float x2, const float y2, const float z2) {
    return x1*x2 + y1*y2 + z1*z2;
}

inline bool w_initialization(const float Wy, const float Wmax) {
    const float coeff = Wy/Vy;
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

inline float norm(const float x1, const float y1, const float z1) {
    return sqrt(x1*x1 + y1*y1 + z1*z1);
}

inline void normalize_vector_n(const float x2, const float y2, const float z2, const float x3, const float y3, const float z3, const float inv_R) {
    const float sub_x = x2-x3;
    const float sub_y = y2-y3;
    const float sub_z = z2-z3;

    Nx = sub_x*inv_R;
    Ny = sub_y*inv_R;
    Nz = sub_z*inv_R;
}

inline void normalize_vector_s( const float x2, const float y2, const float z2, const float x3, const float y3, const float z3) {
    const float sub_x = x2-x3;
    const float sub_y = y2-y3;
    const float sub_z = z2-z3;
    // const float norm_val = norm(sub_x,sub_y,sub_z);
    const float norm_val = sqrt(sub_x*sub_x + sub_y*sub_y + sub_z*sub_z);
    const float inv_norm = 1.0 / norm_val;

    Sx = sub_x*inv_norm;
    Sy = sub_y*inv_norm;
    Sz = sub_z*inv_norm;
}

inline void v_sampling() {

    // const double phi = pcg32_double(&rng)*M_PI-0;
    // const double cos_theta = pcg32_double(&rng)*2-1;
    const float phi = (romuTrio_random()>>11)*INV_2_53*M_PI;
    const float cos_theta = (romuTrio_random()>>11) * INV_2_53*2-1;
    const float sin_theta = sqrt(1-cos_theta*cos_theta);
    Vy = sin_theta*sin(phi);
    Vx = sin_theta*cos(phi);
    Vz = cos_theta;
}

inline float t_real_solution(const float Cx, const float Cy, const float Cz, const float R_sq_minus_CC) {
    VC_dot_product = dot_product(Vx,Vy,Vz,Cx,Cy,Cz);
    return VC_dot_product*VC_dot_product + R_sq_minus_CC;
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


int main(int argc, char *argv[]) {
    const float Lx = strtod(argv[1], NULL);
    const float Ly = strtod(argv[2], NULL);
    const float Lz = strtod(argv[3], NULL);
    const float Wy = strtod(argv[4], NULL);
    const float Wmax = strtod(argv[5], NULL);
    const float Cx = strtod(argv[6], NULL);
    const float Cy = strtod(argv[7], NULL);
    const float Cz = strtod(argv[8], NULL);
    const float R = strtod(argv[9], NULL);
    const int n = atoi(argv[10]);
    const long Nrays = atol(argv[11]);

    const float inv_delta = (float)n/(2.0*Wmax);;

    float **grid;
    float *rows;
    grid = (float**)malloc(n*sizeof(float*));
    rows = (float*)malloc(n*n*sizeof(float));
    for (int i=0; i<n; i++) {
        grid[i] = rows + i*n;
    }
    for (int i=0; i<n*n; i++) {
        rows[i] = 0;
    }
    // pcg32_srandom(&rng, 42u, 54u);
    split64_seed(1337ULL);
    romuTrio_seed();

    const float CC_dot_product = dot_product(Cx,Cy,Cz,Cx,Cy,Cz);
    const float R_sq = R*R;
    const float R_Sq_minus_CC = R_sq-CC_dot_product;
    const float inv_R = 1/R;

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
            const float t_term = VC_dot_product*VC_dot_product + R_Sq_minus_CC;
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

        float const SN_dot_product = dot_product(Sx,Sy,Sz,Nx,Ny,Nz);

        const float b = fmaxf(SN_dot_product, 0.0);

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

    float time_spent = (float)(end-start)/CLOCKS_PER_SEC;

    size_t memory_grids = (size_t) 2*n*n*sizeof(float);
    float total_memory_mb = memory_grids/(1024*1024);

    printf("Time taken = %.6lf\n", time_spent);
    printf("Total memory MB = %.6lf\n", total_memory_mb);
    printf("Rejected rays = %ld\n", rejectedRays);

    free(grid);
    free(rows);
}