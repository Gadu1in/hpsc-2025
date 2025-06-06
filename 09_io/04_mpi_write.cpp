#include <mpi.h>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <vector>
using namespace std;

int main(int argc, char** argv) {
  const int N = 100000000;
  int mpisize, mpirank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int Nlocal = N / mpisize;
  int offset = Nlocal * mpirank;
  vector<int> buffer(Nlocal,1);
  ofstream file("data.dat", ios::binary);
  file.seekp(offset*sizeof(int));
  auto tic = chrono::steady_clock::now();
  file.write((char*)&buffer[0], Nlocal*sizeof(int));
  auto toc = chrono::steady_clock::now();
  file.close();
  double time = chrono::duration<double>(toc - tic).count();
  if(!mpirank) printf("N=%d: %lf s (%lf GB/s)\n",N,time,4*N/time/1e9);
  MPI_Finalize();
}
