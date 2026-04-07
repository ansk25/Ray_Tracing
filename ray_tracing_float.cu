#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <stdint.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "romu.h"
#include "float_utilities.h"
#include <omp.h>

#define INV_2_53 (1.0 / 9007199254740992.0)

__global__ void initialize_state(curandState *states, int total_threads);
__global__ void ray_tracing_cuda(curandState *states, float *grid, int *rejectedRays, float Cx, float Cy, float Cz, float R_sq_minus_CC, float inv_R, float Lx, float Ly, float Lz, float Wy, float Wmax, long Nrays, float inv_delta, int n);
__host__ void ray_tracing_omp(float **grid, int *rejectedRays, float Cx, float Cy, float Cz, float R_sq_minus_CC, float inv_R, float Lx, float Ly, float Lz, float Wy, float Wmax, long Nrays, float inv_delta, int n);

int main(int argc, char *argv[]) {
    const float Lx = 4;
    const float Ly = 4;
    const float Lz = -1;
    const float Wy = 2;
    const float Wmax = 2;
    const float Cx =  0;
    const float Cy =  12;
    const float Cz =  0;
    const float R =  6;
    const int n =  atoi(argv[1]);
    const long Nrays = atol(argv[2]);
    const int ngrid = atoi(argv[3]);
    const int nblocks = atoi(argv[4]);
    const int nthreads_per_block = atoi(argv[5]);
    const char cpu_type = argv[6][0];

	dim3 dimgrid(nblocks);

    const float inv_delta = (float)n/(2.0*Wmax);;

    const float CC_dot_product = Cx*Cx+Cy*Cy+Cz*Cz;
    const float R_sq = R*R;
    const float R_sq_minus_CC = R_sq-CC_dot_product;
    const float inv_R = 1/R;

    cudaEvent_t start_device, stop_device;
	cudaEvent_t start_device_kernel, stop_device_kernel;
    float time_device;
	float time_device_kernel;

    float host_start, host_end, time_host;
	float host_start_kernel, host_end_kernel, time_host_kernel;


    cudaEventCreate(&start_device);
    cudaEventCreate(&stop_device);
    cudaEventCreate(&start_device_kernel);
    cudaEventCreate(&stop_device_kernel);

    host_start = omp_get_wtime();

    float **grid_h = (float **) malloc(n * sizeof(float *));
    float *rows_h = (float *) malloc(n * n * sizeof(float));
    for (int i=0; i<n; i++) {
        grid_h[i] = rows_h + i*n;
    }
    for (int i=0; i<n*n; i++) {
        rows_h[i] = 0;
    }
    int rejectedRays_h =0;
	host_start_kernel = omp_get_wtime();
    ray_tracing_omp(grid_h, &rejectedRays_h, Cx, Cy, Cz, R_sq_minus_CC,inv_R, Lx,Ly,Lz,Wy,Wmax, Nrays, inv_delta, n);
	host_end_kernel = omp_get_wtime();
    host_end = omp_get_wtime();

    time_host = host_end-host_start;
	time_host_kernel = host_end_kernel-host_start_kernel;
    long long samples_omp = 2LL * ((long long)rejectedRays_h + (long long)Nrays);

    char filename[256];
    sprintf(filename, "sphere_omp_n%d.bin", n);
    save_grid_dat(filename, grid_h, n, Nrays);

    printf("Time taken omp = %.6lf\n", time_host);
	printf("Time taken omp kernel = %.6lf\n", time_host_kernel);
    printf("Rejected rays omp = %ld\n", rejectedRays_h);
	printf("Sample rays omp = %lld\n", samples_omp);

    cudaEventRecord(start_device,0);

    curandState *states;
    cudaMalloc((void**) &states, nblocks*nthreads_per_block*sizeof(curandState));

    initialize_state<<<dimgrid, nthreads_per_block>>>(states, nblocks*nthreads_per_block);
    cudaDeviceSynchronize();

    float *grid;
    cudaMalloc((void**) &grid, n * n * sizeof(float));
    cudaMemset(grid, 0, n * n * sizeof(float));

    int *rejectedRays;
    cudaMalloc((void**) &rejectedRays, sizeof(int));
    cudaMemset(rejectedRays, 0, sizeof(int));

    cudaMemcpy(grid,rows_h, n*n* sizeof(float), cudaMemcpyHostToDevice);
	cudaEventRecord(start_device_kernel,0);
    ray_tracing_cuda<<<dimgrid,nthreads_per_block>>>(states, grid, rejectedRays, Cx, Cy, Cz, R_sq_minus_CC,inv_R, Lx,Ly,Lz,Wy,Wmax, Nrays, inv_delta, n);

	cudaEventRecord(stop_device_kernel,0);
	cudaEventSynchronize(stop_device_kernel);
    cudaDeviceSynchronize();

    cudaMemcpy(rows_h, grid, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&rejectedRays_h, rejectedRays, sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_device,0);
    cudaEventSynchronize(stop_device);

    cudaEventElapsedTime(&time_device,start_device,stop_device);
	cudaEventElapsedTime(&time_device_kernel,start_device_kernel,stop_device_kernel);

    cudaFree(grid); cudaFree(rejectedRays); cudaFree(states);
    cudaEventDestroy(start_device);
    cudaEventDestroy(stop_device);
	cudaEventDestroy(start_device_kernel);
	cudaEventDestroy(stop_device_kernel);

    sprintf(filename, "sphere_cuda_%c_%d.bin", cpu_type, n);
    save_grid_dat(filename, grid_h, n, Nrays);

    size_t memory_grids = (size_t) 2*n*n*sizeof(float);
    float total_memory_mb = memory_grids/(1024*1024);
	long long samples_cuda = 2LL * ((long long)rejectedRays_h + (long long)Nrays);

    printf("Time taken cuda = %.6lf\n", time_device);
	printf("Time taken cuda kernel = %.6lf\n", time_device_kernel);
    printf("Rejected rays cuda = %ld\n", rejectedRays_h);
	printf("Sample rays cuda = %lld\n", samples_cuda);

    free(grid_h); free(rows_h);

}

__global__ void initialize_state(curandState *states, int total_threads) {
    int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
    if (thread_id < total_threads) {
        curand_init(thread_id*4238811, thread_id, 0, &states[thread_id]);
    }
    return;
}

__global__ void ray_tracing_cuda(curandState *states, float *grid, int *rejectedRays, float Cx, float Cy, float Cz, float R_sq_minus_CC, float inv_R, float Lx, float Ly, float Lz, float Wy, float Wmax, long Nrays, float inv_delta, int n) {
    int rejectedRays_local = 0;
    __shared__ int s[256];
    int thread_id = threadIdx.x + blockIdx.x*blockDim.x;
    curandState state_local = states[thread_id];
    for (int i_thread= thread_id; i_thread<Nrays; i_thread+=blockDim.x*gridDim.x) {
        float Vx, Vy, Vz;
        float Wx, Wz;
        float Ix, Iy, Iz;
        float Sx,Sy,Sz;
        float t;

            while (true) {
                const float phi = curand_uniform(&state_local)*M_PI;
                const float cos_theta = curand_uniform(&state_local)*2-1;
                const float sin_theta = sqrt(1-cos_theta*cos_theta);
                Vy = sin_theta*sin(phi);
                Vx = sin_theta*cos(phi);
                Vz = cos_theta;
                if (Vy<=0) {
                    rejectedRays_local++;
                    continue;
                }
                const float coeff = Wy/Vy;
                Wx = coeff*Vx;
                Wz = coeff*Vz;
                if (!(fabsf(Wx) < Wmax && fabsf(Wz) < Wmax)) {
                    rejectedRays_local++;
                    continue;
                }
                const float VC_dot_product = Vx*Cx+Vy*Cy+Vz*Cz;
                const float t_term = VC_dot_product*VC_dot_product+ R_sq_minus_CC;
                if (t_term<0) {
                    rejectedRays_local++;
                    continue;
                }
                t = VC_dot_product - sqrt(t_term);
                break;
            }
            Ix = t*Vx;
            Iy = t*Vy;
            Iz = t*Vz;

            const float Nx = (Ix-Cx)*inv_R;
            const float Ny = (Iy-Cy)*inv_R;
            const float Nz = (Iz-Cz)*inv_R;

            const float sub_x2 = Lx-Ix;
            const float sub_y2 = Ly-Iy;
            const float sub_z2 = Lz-Iz;
            //const double norm_val = sqrt(sub_x2*sub_x2 + sub_y2*sub_y2 + sub_z2*sub_z2);
            //const double inv_norm = 1.0 / norm_val;
			const float inv_norm = rsqrt(sub_x2*sub_x2 + sub_y2*sub_y2 + sub_z2*sub_z2);

            Sx = sub_x2*inv_norm;
            Sy = sub_y2*inv_norm;
            Sz = sub_z2*inv_norm;

            float const SN_dot_product = Sx*Nx+Sy*Ny+Sz*Nz;

            const float b = fmaxf(SN_dot_product, 0.0);

            const int i = (int)((Wx + Wmax)*inv_delta);
            const int j = (int)((Wz + Wmax)*inv_delta);
            if (i>=0 && i<n && j>=0 && j<n) {
                atomicAdd(&grid[i*n+j], b);
            }
        }
    states[thread_id] = state_local;
    s[threadIdx.x] = rejectedRays_local;
    __syncthreads();
    if (threadIdx.x==0) {
        int sum = 0;
        for (int i=0; i<blockDim.x; i++) {
            sum += s[i];
        }
        atomicAdd(rejectedRays, sum);
    }
    return;
}

__host__ void ray_tracing_omp(float **grid, int *rejectedRays, float Cx, float Cy, float Cz, float R_sq_minus_CC, float inv_R, float Lx, float Ly, float Lz, float Wy, float Wmax, long Nrays, float inv_delta, int n) {
    long l;
    float Vx, Vy, Vz;
    float Wx, Wz;
    float Ix, Iy, Iz;
    float Nx,Ny,Nz;
    float Sx,Sy,Sz;
    float VC_dot_product;
    float t;

#pragma omp parallel default(none) shared(Nrays, Wy, Wmax, Cx, Cy, Cz, R_sq_minus_CC, inv_R, inv_delta, grid, Lx, Ly,Lz, rejectedRays,n) private(l,Vx,Vy,Vz,Wx,Wz,Ix,Iy,Iz,Nx,Ny,Nz,Sx,Sy,Sz,VC_dot_product,t)
    {
        int rejectedRays_local = 0;
        int thread_id = omp_get_thread_num();
        romu_state prng;
        uint64_t seed = time(NULL) + thread_id*0x9e3779b97f4a7c15ULL;
        romuTrio_seed(&prng,seed);
#pragma omp for schedule(static)
    for (l=1; l<=Nrays; l++) {

        while (true) {
            float const u = (romuTrio_random(&prng)>>11)*INV_2_53;
            float const v = (romuTrio_random(&prng)>>11)*INV_2_53;
            const float phi = u*M_PI;
            const float cos_theta = v*2-1;
            const float sin_theta = sqrt(1-cos_theta*cos_theta);
            Vy = sin_theta*sin(phi);
            Vx = sin_theta*cos(phi);
            Vz = cos_theta;
            if (Vy<=0) {
                rejectedRays_local += 1;
                continue;
            }
            if (!w_initialization(Wy, Wmax,Vx,Vy,Vz,&Wx,&Wz)) {
                rejectedRays_local +=1;
                continue;
            }
            const float t_term = t_real_solution(Cx, Cy, Cz, R_sq_minus_CC,Vx,Vy,Vz, &VC_dot_product);
            if (t_term<0) {
                rejectedRays_local +=1;
                continue;
            }
            t = VC_dot_product - sqrt(t_term);
            break;
        }

        i_initialization(Vx,Vy,Vz,&Ix,&Iy,&Iz,t);

        normalize_vector_n(Ix,Iy,Iz,Cx,Cy,Cz,inv_R,&Nx,&Ny,&Nz);

        normalize_vector_s(Lx,Ly,Lz,Ix, Iy, Iz,&Sx,&Sy,&Sz);

        float const SN_dot_product = dot_product(Sx,Sy,Sz,Nx,Ny,Nz);

        const float b = fmaxf(SN_dot_product, 0.0);

        const int i = (int)((Wx + Wmax)*inv_delta);
        const int j = (int)((Wz + Wmax)*inv_delta);
        if (i>=0 && i<n && j>=0 && j<n) {
#pragma omp atomic
            *(grid[i] + j) +=b;
        }
    }
#pragma omp atomic
        *rejectedRays += rejectedRays_local;
#pragma omp single
        printf("NUmber of threads: %d\n", omp_get_num_threads());
}
    return;
}
