**Monte Carlo Ray Tracing Renderer**

This project implements a Monte Carlo ray tracing engine for image generation, simulating global illumination through large-scale stochastic light sampling.

**Tech Stack**
C / C++ for core implementation
CUDA for GPU-accelerated ray tracing
MPI for distributed multi-GPU execution
OpenMP for shared-memory CPU parallelism
Overview

The renderer traces a large number of randomly sampled light rays (about 6 Billion) to model realistic lighting effects. It supports execution across serial CPU, multicore CPU, and single/multi-GPU environments, enabling flexible experimentation with parallel and distributed computing strategies.
