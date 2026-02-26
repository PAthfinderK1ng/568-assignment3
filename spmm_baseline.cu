
// spmm_baseline.cu — STUDENT SKELETON
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>

extern void generate_random_csr(int M, int K, double density,
                                std::vector<int>& row_ptr,
                                std::vector<int>& col_idx,
                                std::vector<float>& vals,
                                unsigned seed);

extern void spmm_cpu(int M, int K, int N,
              const std::vector<int>& row_ptr,
              const std::vector<int>& col_idx,
              const std::vector<float>& vals,
              const std::vector<float>& B,
              std::vector<float>& C);

extern float max_abs_err(const std::vector<float>& A, const std::vector<float>& B);

using float_t = float;

/*
===============================================================
 BASELINE KERNEL — one thread processes ONE ROW of A
 STUDENT TODO: 
   - Fill missing loops
   - Compute C[row, j] += value * B[k, j]
===============================================================
*/
__global__ void spmm_csr_row_kernel(
    int M, int N,
    const int* __restrict__ d_row_ptr,
    const int* __restrict__ d_col_idx,
    const float_t* __restrict__ d_vals,
    const float_t* __restrict__ d_B,
    float_t* __restrict__ d_C)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    int c_base = row * N;
    for (int j = 0; j < N; ++j) {
        d_C[c_base + j] = (float_t)0;
    }

    int start = d_row_ptr[row];
    int end   = d_row_ptr[row + 1];
    for (int idx = start; idx < end; ++idx) {
        int k = d_col_idx[idx];    
        float_t v = d_vals[idx];    

        int b_base = k * N;
        for (int j = 0; j < N; ++j) {
            d_C[c_base + j] += v * d_B[b_base + j];  
        }
    }
}

/*
===============================================================
 MAIN PROGRAM
===============================================================
*/
int main(int argc, char** argv) {
    int M = 512, K = 512, N = 64;
    double density = 0.01;
    unsigned seed = 1234;

    std::vector<int> row_ptr, col_idx;
    std::vector<float> vals;
    generate_random_csr(M, K, density, row_ptr, col_idx, vals, seed);
    int nnz = row_ptr.back();
    std::cout << "nnz = " << nnz << "\\n";

    // Create B
    std::vector<float> B((size_t)K * N);
    for (size_t i = 0; i < B.size(); i++) B[i] = float(rand()) / RAND_MAX;

    // CPU reference
    std::vector<float> C_ref;
    spmm_cpu(M, K, N, row_ptr, col_idx, vals, B, C_ref);

    // Copy to device
    int *d_row_ptr, *d_col_idx;
    float *d_vals, *d_B, *d_C;
    cudaMalloc(&d_row_ptr, (M+1)*sizeof(int));
    cudaMalloc(&d_col_idx, nnz*sizeof(int));
    cudaMalloc(&d_vals, nnz*sizeof(float));
    cudaMalloc(&d_B, (size_t)K*N*sizeof(float));
    cudaMalloc(&d_C, (size_t)M*N*sizeof(float));

    cudaMemcpy(d_row_ptr, row_ptr.data(), (M+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx.data(), nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, vals.data(), nnz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), (size_t)K*N*sizeof(float), cudaMemcpyHostToDevice);

    // Launch incomplete student kernel
    int block = 256;
    int grid = (M + block - 1) / block;

    spmm_csr_row_kernel<<<grid, block>>>(M, N, d_row_ptr, d_col_idx, d_vals, d_B, d_C);
    cudaDeviceSynchronize();

// Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iters = 100;

    cudaEventRecord(start);

    for (int it = 0; it < iters; ++it) {
        spmm_csr_row_kernel<<<grid, block>>>(
            M, N,
            d_row_ptr,
            d_col_idx,
            d_vals,
            d_B,
            d_C
        );
    }   

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iters;

    std::cout << "Baseline Kernel time (avg over "
          << iters << ") = " << ms << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

// Optional error check
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        std::cout << "Kernel launch error: "
                << cudaGetErrorString(e) << "\n";
        cudaDeviceSynchronize();

    // Copy back
    std::vector<float> C((size_t)M*N);
    cudaMemcpy(C.data(), d_C, (size_t)M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Compare (will be wrong until students complete TODOs)
    float err = max_abs_err(C_ref, C);
    std::cout << "Max error = " << err << "\\n";

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_vals);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
