//
// Tomás Oliveira e Silva, November 2017
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

static void compute_CC_cpu_kernel(int n, double *x, double *y, double *points, double *results);
__global__ static void computeCC_cuda_kernel(int n, double *x_h, double *h_y, double *points, double *results);
static double get_delta_time(void);
static void generate_samples(double *m, int N);
static void generate_points(double *m, int N);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main program
//

int main(int argc, char **argv)
{

  if (argc < 2)
  {
    perror("Please insert the number of samples!");
    ;
    exit(EXIT_FAILURE);
  }

  printf("%s Starting...\n", argv[0]);
  if (sizeof(unsigned int) != (size_t)4)
    return 1; // fail with prejudice if an integer does not have 4 bytes

  // set up device
  int dev = 0;

  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Using Device %d: %s\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

  // create memory areas in host and device memory where the disk sectors data and sector numbers will be stored
  int n; 
  sscanf (argv[1],"%d",&n);; //Number of Samples

  double *h_x, *h_y, *h_points;

  int nBytes = n * sizeof(double); //Storage space in bytes
  double *result_cuda = (double *)malloc(nBytes);
  double *result_cpu = (double *)malloc(nBytes);

  h_x = (double *)malloc(nBytes);
  h_y = (double *)malloc(nBytes);
  h_points = (double *)malloc(nBytes);

  //generate samples for x and y
  (void)get_delta_time();
  generate_samples(h_x, n);
  generate_samples(h_y, n);
  printf("Samples for signals x and y generated on %.3e seconds\n", get_delta_time());

  //generate points
  (void)get_delta_time();
  generate_points(h_points, n);
  printf("Samples for points generated on %.3e seconds\n", get_delta_time());

  //reserve memory for gpu
  double *d_x, *d_y, *d_points, *d_results;
  CHECK(cudaMalloc((void **)&d_x, nBytes));
  CHECK(cudaMalloc((void **)&d_y, nBytes));
  CHECK(cudaMalloc((void **)&d_points, nBytes));
  CHECK(cudaMalloc((void **)&d_results, nBytes));

  // copy the host data to the device memory
  (void)get_delta_time();
  //copy matrix to gpu
  CHECK(cudaMemcpy(d_x, h_x, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_y, h_y, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_points, h_points, nBytes, cudaMemcpyHostToDevice));
  printf("The transfer of %d bytes from the host to the device took %.3e seconds\n",
         3 * nBytes, get_delta_time());

  // run the computational kernel
  // as an example, n threads are launched where each thread deals with one point
  unsigned int gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ;

  blockDimX = 1 << 0; // optimize!
  blockDimY = 1 << 0; // optimize!
  blockDimZ = 1 << 0; // do not change!
  gridDimX = 1 << n;  // optimize!
  gridDimY = 1 << 0;  // optimize!
  gridDimZ = 1 << 0;  // do not change!

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
  computeCC_cuda_kernel<<<grid, block>>>(n, d_x, d_y, d_points, d_results);
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
  CHECK(cudaFree(d_points));  //gpu
  CHECK(cudaFree(d_results)); //gpu

  // reset device
  CHECK(cudaDeviceReset());

  // compute the modified sector data on the CPU
  (void)get_delta_time();
  compute_CC_cpu_kernel(n, h_x, h_y, h_points, result_cpu);
  printf("The cpu kernel took %.3e seconds to run (single core)\n", get_delta_time());

  // compare
  size_t i;
  for (i = 0; i < n; i++)
    if (result_cpu[i] != result_cuda[i])
    {
      printf("Mismatch in point %f, expected %f.\n", h_points[i], result_cpu[i]);
      exit(1);
    }
  printf("All is well!\n");

  // free host memory
  free(h_x); //cpu
  free(h_y);
  free(h_points);
  free(result_cuda);
  free(result_cpu);

  return 0;
}

static void compute_CC_cpu_kernel(int n, double *x, double *y, double *points, double *results)
{
  unsigned int k, i;
  size_t point;
  for (k = 0; k < n; k++)
  {
    point = points[k];
    for (i = 0; i < n; i++)
    {
      results[k] += x[i] * y[(point + i) % n];
    }
  }
}

__global__ static void computeCC_cuda_kernel(int n, double *x_h, double *h_y, double *points, double *results)
{
  unsigned int x, y, idx, i;

  // compute the thread number
  x = (unsigned int)threadIdx.x + (unsigned int)blockDim.x * (unsigned int)blockIdx.x;
  y = (unsigned int)threadIdx.y + (unsigned int)blockDim.y * (unsigned int)blockIdx.y;
  idx = (unsigned int)blockDim.x * (unsigned int)gridDim.x * y + x;

  size_t point = points[idx];

  for (i = 0; i < n; i++)
  {
    results[idx] += x_h[i] * h_y[(point + i) % n];
  }
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

  for (i = 0; i < N; i++)
  {
    m[i] = i;
  }
}

static void generate_points(double *m, int N)
{
  size_t i;

  for (i = 0; i < N; i++)
  {
    m[i] = i;
  }
}