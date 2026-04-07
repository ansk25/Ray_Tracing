#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <stdint.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <mpi.h>

#define INV_2_53 (1.0 / 9007199254740992.0)

__global__ void initialize_state(curandState *states, int total_threads, int rank);
__global__ void ray_tracing_cuda(int rays_per_rank, curandState *states, double *grid, int *rejectedRays, double Cx, double Cy, double Cz, double R_sq_minus_CC, double inv_R, double Lx, double Ly, double Lz, double Wy, double Wmax, long Nrays, double inv_delta, int n);

int main(int argc, char *argv[]) {
    const double Lx = 4;
    const double Ly = 4;
    const double Lz = -1;
    const double Wy = 2;
    const double Wmax = 2;
    const double Cx =  0;
    const double Cy =  12;
    const double Cz =  0;
    const double R =  6;
    const int n =  atoi(argv[1]);
    const long Nrays = atol(argv[2]);
    const int ngrid = atoi(argv[3]);
    const int nblocks = atoi(argv[4]);
    const int nthreads_per_block = atoi(argv[5]);

	dim3 dimgrid(nblocks);

    const double inv_delta = (double)n/(2.0*Wmax);;

    const double CC_dot_product = Cx*Cx+Cy*Cy+Cz*Cz;
    const double R_sq = R*R;
    const double R_sq_minus_CC = R_sq-CC_dot_product;
    const double inv_R = 1/R;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cudaSetDevice(rank);

	cudaEvent_t start_device_kernel, stop_device_kernel;
    float time_device;
	float time_device_kernel;

    cudaEventCreate(&start_device_kernel);
    cudaEventCreate(&stop_device_kernel);

    double time_start_mpi = MPI_Wtime();

    double **grid_h = (double **) malloc(n * sizeof(double *));
    double *rows_h = (double *) malloc(n * n * sizeof(double));
    for (int i=0; i<n; i++) {
        grid_h[i] = rows_h + i*n;
    }
    for (int i=0; i<n*n; i++) {
        rows_h[i] = 0;
    }
    int rejectedRays_h =0;


    int rays_per_rank = Nrays/size;
    int remainder = Nrays%size;
    if (rank<remainder) {
        rays_per_rank++;
    }

    curandState *states;
    cudaMalloc((void**) &states, nblocks*nthreads_per_block*sizeof(curandState));

    initialize_state<<<dimgrid, nthreads_per_block>>>(states, nblocks*nthreads_per_block, rank);
    cudaDeviceSynchronize();


    double *grid;
    cudaMalloc((void**) &grid, n * n * sizeof(double));
    cudaMemset(grid, 0, n * n * sizeof(double));

    int *rejectedRays;
    cudaMalloc((void**) &rejectedRays, sizeof(int));
    cudaMemset(rejectedRays, 0, sizeof(int));

    cudaMemcpy(grid,rows_h, n*n* sizeof(double), cudaMemcpyHostToDevice);
	cudaEventRecord(start_device_kernel,0);
    ray_tracing_cuda<<<dimgrid,nthreads_per_block>>>(rays_per_rank, states, grid, rejectedRays, Cx, Cy, Cz, R_sq_minus_CC,inv_R, Lx,Ly,Lz,Wy,Wmax, Nrays, inv_delta, n);

	cudaEventRecord(stop_device_kernel,0);
	cudaEventSynchronize(stop_device_kernel);

    cudaMemcpy(rows_h, grid, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&rejectedRays_h, rejectedRays, sizeof(int), cudaMemcpyDeviceToHost);


    int rejectedRays_global = 0;
    int rejectedRays_local = (int)rejectedRays_h;

    MPI_Reduce(&rejectedRays_local,&rejectedRays_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double **grid_global = NULL;
    double *rows_global = NULL;
    if (rank == 0) {
        rows_global = (double *) malloc(n*n* sizeof(double));
    }

    MPI_Reduce(rows_h, rows_global, n * n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank==0) {
        grid_global = (double **) malloc(n * sizeof(double *));
        for (int i=0; i<n; i++) {
            grid_global[i] = rows_global + i*n;
        }
    }

    double time_end_mpi = MPI_Wtime();
    double time_mpi_local = time_end_mpi - time_start_mpi;

    double total_time_all_ranks;
    MPI_Reduce(&time_mpi_local, &total_time_all_ranks, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Total MPI time = %.6lf s\n", total_time_all_ranks);
    }

    if (rank == 0) {
        uint64_t samples_cuda = 2ULL * ((uint64_t)rejectedRays_global + (uint64_t)Nrays);
        printf("Rejected rays cuda = %ld\n", rejectedRays_global);
        printf("Sample rays cuda = %llu\n", samples_cuda);

        char filename[256];
        sprintf(filename, "sphere_cuda_MPI_n%d.bin", n);
        FILE *f = fopen(filename, "wb");
        if (!f) {
            perror("fopen");
            return;
        }

        int32_t N_save = (int32_t) Nrays;
        int32_t R_save = (int32_t) n;
        int32_t C_save = (int32_t) n;

        fwrite(&N_save, sizeof(int32_t), 1, f);
        fwrite(&R_save, sizeof(int32_t), 1, f);
        fwrite(&C_save, sizeof(int32_t), 1, f);

        for (int k = 0; k < n; k++) {
            fwrite(grid_global[k], sizeof(double), n, f);
        }

        fclose(f);
    }

	cudaEventElapsedTime(&time_device_kernel,start_device_kernel,stop_device_kernel);

    float max_kernel_time_ms;
    MPI_Reduce(&time_device_kernel, &max_kernel_time_ms, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Max GPU kernel time across all ranks = %.6f ms\n", max_kernel_time_ms);
    }

    cudaFree(grid); cudaFree(rejectedRays); cudaFree(states);
	cudaEventDestroy(start_device_kernel);
	cudaEventDestroy(stop_device_kernel);

    free(grid_h); free(rows_h);
    MPI_Finalize();

}

__global__ void initialize_state(curandState *states, int total_threads, int rank) {
    int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
    if (thread_id < total_threads) {
        curand_init((rank+1)*1234567 + thread_id*4238811, thread_id, rank+1, &states[thread_id]);
    }
    return;
}

__global__ void ray_tracing_cuda(int rays_per_rank, curandState *states, double *grid, int *rejectedRays, double Cx, double Cy, double Cz, double R_sq_minus_CC, double inv_R, double Lx, double Ly, double Lz, double Wy, double Wmax, long Nrays, double inv_delta, int n) {
    int thread_id = threadIdx.x + blockIdx.x*blockDim.x;
    int total_threads_in_rank = blockDim.x*gridDim.x;
    int rejectedRays_local = 0;
    __shared__ int s[256];
    curandState state_local = states[thread_id];
    for (int i_thread= thread_id; i_thread<rays_per_rank; i_thread+=total_threads_in_rank) {
        double Vx, Vy, Vz;
        double Wx, Wz;
        double Ix, Iy, Iz;
        double Sx,Sy,Sz;
        double t;

            while (true) {
                const double phi = curand_uniform(&state_local)*M_PI;
                const double cos_theta = curand_uniform(&state_local)*2-1;
                const double sin_theta = sqrt(1-cos_theta*cos_theta);
                Vy = sin_theta*sin(phi);
                Vx = sin_theta*cos(phi);
                Vz = cos_theta;
                if (Vy<=0) {
                    rejectedRays_local++;
                    continue;
                }
                const double coeff = Wy/Vy;
                Wx = coeff*Vx;
                Wz = coeff*Vz;
                if (!(fabs(Wx) < Wmax && fabs(Wz) < Wmax)) {
                    rejectedRays_local++;
                    continue;
                }
                const double VC_dot_product = Vx*Cx+Vy*Cy+Vz*Cz;
                const double t_term = VC_dot_product*VC_dot_product+ R_sq_minus_CC;
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

            const double Nx = (Ix-Cx)*inv_R;
            const double Ny = (Iy-Cy)*inv_R;
            const double Nz = (Iz-Cz)*inv_R;

            const double sub_x2 = Lx-Ix;
            const double sub_y2 = Ly-Iy;
            const double sub_z2 = Lz-Iz;
            //const double norm_val = sqrt(sub_x2*sub_x2 + sub_y2*sub_y2 + sub_z2*sub_z2);
            //const double inv_norm = 1.0 / norm_val;
			const double inv_norm = rsqrt(sub_x2*sub_x2 + sub_y2*sub_y2 + sub_z2*sub_z2);

            Sx = sub_x2*inv_norm;
            Sy = sub_y2*inv_norm;
            Sz = sub_z2*inv_norm;

            double const SN_dot_product = Sx*Nx+Sy*Ny+Sz*Nz;

            const double b = fmax(SN_dot_product, 0.0);

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
