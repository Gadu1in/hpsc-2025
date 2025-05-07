#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void bucket_make_kernel(int* key, int* bucket, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < n) atomicAdd(&bucket[key[idx]], 1);
}

__global__ void bucket_sort_kernel(int* key, int* bucket, int range, int n) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx >= n) return; // We let the same amount of threads as the size of the list key
                        // divide the work between them.

  int count = 0;
  key[idx] = 0;
  for (int i = 0; i < range; i++) {
          count += bucket[i];
          if (idx < count) {
                  key[idx] = i;
                  break;
          }
  }
}

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int *d_key, *d_bucket;
  cudaMalloc(&d_key, n * sizeof(int));
  cudaMalloc(&d_bucket, range * sizeof(int));

  cudaMemcpy(d_key, key.data(), n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(d_bucket, 0, range * sizeof(int));

  bucket_make_kernel<<<1, n>>>(d_key, d_bucket, n);
  cudaDeviceSynchronize();

  bucket_sort_kernel<<<1, n>>>(d_key, d_bucket, range, n);
  cudaDeviceSynchronize();

  cudaMemcpy(key.data(), d_key, n * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_key);
  cudaFree(d_bucket);

  /*
  std::vector<int> bucket(range); 
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }*/

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
