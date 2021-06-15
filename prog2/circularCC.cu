//
// Assignment 3 - The threads in a warp compute cross-correlation points separated by a fixed distance
// Rafael Sá, 104552, rafael.sa@ua.pt
// Luís Laranjeira, 81526, lclm@ua.pt
//

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// program configuration
//

static void compute_CC_cpu_kernel(int n, double *x, double *y, double *results);
__global__ static void computeCC_cuda_kernel(int n, double *x_h, double *h_y, double *results);
static double get_delta_time(void);
static void generate_samples(double *m, int N);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main program
//

int main(int argc, char **argv)
{

  // set up device
  int dev = 0;

  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Using Device %d: %s\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

  // create memory areas in host and device memory where the disk sectors data and sector numbers will be stored
  int n = 1 << 16; 

  double *h_x, *h_y, *result_cuda, *result_cpu;
  int nBytes;

  nBytes = n * sizeof(double); //Storage space in bytes

  h_x = (double *)malloc(nBytes);
  h_y = (double *)malloc(nBytes);
  result_cuda = (double *)malloc(nBytes);
  result_cpu = (double *)malloc(nBytes);

  //generate samples for x and y
  (void)get_delta_time();
  generate_samples(h_x, n);
  generate_samples(h_y, n);
  printf("Samples for signals x and y generated on %.3e seconds\n", get_delta_time());

  //reserve memory for gpu
  double *d_x, *d_y, *d_results;
  CHECK(cudaMalloc((void **)&d_x, nBytes));
  CHECK(cudaMalloc((void **)&d_y, nBytes));
  CHECK(cudaMalloc((void **)&d_results, nBytes));

  // copy the host data to the device memory
  (void)get_delta_time();
  //copy to gpu
  CHECK(cudaMemcpy(d_x, h_x, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_y, h_y, nBytes, cudaMemcpyHostToDevice));
  printf("The transfer of %d bytes from the host to the device took %.3e seconds\n",
         3 * nBytes, get_delta_time());

  // run the computational kernel
  // as an example, n threads are launched where each thread deals with one point
  unsigned int gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ;

    blockDimX = 1;    // optimize! // 1 thread
    blockDimY = 1;    // optimize!
    blockDimZ = 1;    // do not change! // sempre 1
    gridDimX = n;     // optimize!
    gridDimY = 1;     // optimize!
    gridDimZ = 1;     // do not change! // sempre 1

  dim3 grid(gridDimX, gridDimY, gridDimZ);
  dim3 block(blockDimX, blockDimY, blockDimZ);

  if ((gridDimX * gridDimY * gridDimZ * blockDimX * blockDimY * blockDimZ) != n)
  {
    printf("%d\n",(gridDimX * gridDimY * gridDimZ * blockDimX * blockDimY * blockDimZ));
    printf("%d\n",n);
    printf("Wrong configuration!\n");
    return 1;
  }
  (void)get_delta_time();
  computeCC_cuda_kernel<<<grid, block>>>(n, d_x, d_y, d_results);
  CHECK(cudaDeviceSynchronize()); // wait for kernel to finish
  CHECK(cudaGetLastError());      // check for kernel errors
  printf("The CUDA kernel <<<(%d,%d,%d), (%d,%d,%d)>>> took %.3e seconds to run\n",
         gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, get_delta_time());

  // copy kernel result back to host side
  CHECK(cudaMemcpy(result_cuda, d_results, nBytes, cudaMemcpyDeviceToHost));
  printf("The transfer of %d bytes from the device to the host took %.3e seconds\n",
         nBytes, get_delta_time());

  // free device global memory
  CHECK(cudaFree(d_x)); //gpu
  CHECK(cudaFree(d_y));
  CHECK(cudaFree(d_results)); //gpu

  // reset device
  CHECK(cudaDeviceReset());

  // compute the CC on the CPU
  (void)get_delta_time();
  compute_CC_cpu_kernel(n, h_x, h_y, result_cpu);
  printf("The cpu kernel took %.3e seconds to run (single core)\n", get_delta_time());

  // compare
  size_t i;
  for (i = 0; i < n; i++)
    if((fabs(result_cuda[i] - result_cpu[i]) >= 1e-6) && 
      ((fabs(result_cuda[i]) < 1e-6 ) || (((result_cuda[i] - result_cpu[i]) / result_cuda[i]) >= 1e-6)))
    {
      printf("Mismatch in point %zu, expected %f.\n", i, result_cpu[i]);
      exit(1);
    }
  printf("All is well!\n");

  // free host memory
  free(h_x); //cpu
  free(h_y);
  free(result_cuda);
  free(result_cpu);

  return 0;
}

static void compute_CC_cpu_kernel(int n, double *x, double *y, double *results)
{
  unsigned int point, i;
  double sum;
  for (point = 0; point < n; point++)
  {
    sum = 0.0;
    for (i = 0; i < n; i++)
    {
      sum += x[i] * y[(point + i) % n];
    }
    results[point] = sum;
  }
}

__global__ static void computeCC_cuda_kernel(int n, double *x_h, double *h_y, double *results)
{
  int distance = 5; // optimize!

  unsigned int x, y, idx, i;
  double sum;

  // compute the thread number
  x = (unsigned int)threadIdx.x + (unsigned int)blockDim.x * (unsigned int)blockIdx.x;
  y = (unsigned int)threadIdx.y + (unsigned int)blockDim.y * (unsigned int)blockIdx.y;
  idx = (unsigned int)blockDim.x * (unsigned int)gridDim.x * y + x;
  idx = (idx + distance) % n;

  sum = 0.0;
  for (i = 0; i < n; i++)
  {
    sum += x_h[i] * h_y[(idx + i) % n];
  }
  results[idx] = sum;
}

static double get_delta_time(void)
{
  static struct timespec t0, t1;

  t0 = t1;
  if (clock_gettime(CLOCK_MONOTONIC, &t1) != 0)
  {
    perror("clock_gettime");
    exit(1);
  }
  return (double)(t1.tv_sec - t0.tv_sec) + 1.0e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
}

static void generate_samples(double *m, int N)
{
  size_t i;
  double lower = -0.5;
  double upper = 0.5;
  for (i = 0; i < N; i++)
  {
    m[i] = ((double)rand() * (upper - lower)) / (double)RAND_MAX + lower;
  }
}
