#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <omp.h>

void merge(std::vector<int>& vec, int begin, int mid, int end) {
  std::vector<int> tmp(end-begin+1);
  int left = begin;
  int right = mid+1;
  for (int i=0; i<tmp.size(); i++) {
    if (left > mid)
      tmp[i] = vec[right++];
    else if (right > end)
      tmp[i] = vec[left++];
    else if (vec[left] <= vec[right])
      tmp[i] = vec[left++];
    else
      tmp[i] = vec[right++];
  }
  for (int i=0; i<tmp.size(); i++)
    vec[begin++] = tmp[i];
}

void merge_sort(std::vector<int>& vec, int begin, int end) {
  if(begin < end) {
    int mid = (begin + end) / 2;

    if(end - begin > 2000) { // Cutoff to not be limited by overhead, at some point its better to run
                             // the code in serial instead.
#pragma omp task shared(vec, begin)
    printf("Thread %d computing merge_sort(vec=vec, begin=%d, mid=%d)\n", omp_get_thread_num(), begin, mid);
    merge_sort(vec, begin, mid);
#pragma omp task shared(vec, end)
    printf("Thread %d computing merge_sort(vec=vec, mid=%d, end=%d)\n", omp_get_thread_num(), mid+1, end);
    merge_sort(vec, mid+1, end);
#pragma omp taskwait
    merge(vec, begin, mid, end);
    }
    else {
        merge_sort(vec, begin, mid);
        merge_sort(vec, mid+1, end);
        merge(vec, begin, mid, end);
    }
  }
}


int main() {
  int n = 20;
  std::vector<int> vec(n);
  for (int i=0; i<n; i++) {
    vec[i] = rand() % (10 * n);
    printf("%d ",vec[i]);
  }
  printf("\n");
  auto tic = std::chrono::steady_clock::now();
#pragma omp parallel
#pragma omp single
  merge_sort(vec, 0, n-1);
  auto toc = std::chrono::steady_clock::now();
  double time = std::chrono::duration<double>(toc - tic).count();
  for (int i=0; i<n; i++) {
    printf("%d ",vec[i]);
  }
  printf("\n t=%f \n", time);
}
