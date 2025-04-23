#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <omp.h>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0);

  auto tic = std::chrono::steady_clock::now();

  for (int i=0; i<n; i++)
    bucket[key[i]]++; // Accumulates how many entries we have in each bucket
  std::vector<int> offset(range,0);
  for (int i=1; i<range; i++)
    offset[i] = offset[i-1] + bucket[i-1]; // The total number of elements to get from the first
                                           // element to the entry after the last element in bucket i
                                           // bucket    = [2] [1] [4] [0] [2] [3]  [2]  [4]
                                           // -> offset = [0] [2] [3] [7] [7] [9] [12] [14]
  for (int i=0; i<range; i++) {
    int j = offset[i];
#pragma omp parallel for
    for (int bucket_n=0; bucket_n<bucket[i]; bucket_n++) {
      key[j + bucket_n] = i;
      printf("Thread %d computing key[j=%d, bucket=%d] = i=%d \n", omp_get_thread_num(), j, bucket_n, i);
    }
  }

  auto toc = std::chrono::steady_clock::now();
  double time = std::chrono::duration<double>(toc - tic).count();

  printf("n=%d, t=%f\n", n, time);

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}