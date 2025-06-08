#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

using namespace std;

struct matrx {
    int nx;
    int ny;
    float* arr;

    __host__ __device__
    float& operator()(int j, int i) {
        return arr[j * nx + i];
    }
    __host__ __device__
    const float& operator()(int j, int i) const {
        return arr[j * nx + i];
    }
};


__global__ void init_matrix_struct(matrx* m, float* data, int nx, int ny) {
    m->nx = nx;
    m->ny = ny;
    m->arr = data;
}


__global__ void apply_pressure_bc_kernel(matrx* p, int nx, int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (j < ny) {
        if (i == 0) (*p)(j, i) = (*p)(j, 1);
        if (i == nx - 1) (*p)(j, i) = (*p)(j, nx - 2);
    }
    if (i < nx) {
        if (j == 0) (*p)(j, i) = (*p)(1, i);
        if (j == ny - 1) (*p)(j, i) = 0.0;
    }
}

__global__ void apply_velocity_bc_kernel(matrx* u, matrx* v, int nx, int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (j < ny) {
        if (i == 0 || i == nx - 1) {
            (*u)(j, i) = 0.0;
            (*v)(j, i) = 0.0;
        }
    }
    if (i < nx) {
        if (j == 0) {
            (*u)(j, i) = 0.0;
            (*v)(j, i) = 0.0;
        } else if (j == ny - 1) {
            (*u)(j, i) = 1.0;
            (*v)(j, i) = 0.0;
        }
    }
}

__global__ void compute_b_kernel(matrx* b, const matrx* u, const matrx* v, int nx, int ny, float dx, float dy, float dt, float rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // inner points only
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= nx-1 || j >= ny-1) return;

    float du_dx = ((*u)(j, i+1) - (*u)(j, i-1)) / (2.0 * dx);
    float dv_dy = ((*v)(j+1, i) - (*v)(j-1, i)) / (2.0 * dy);
    float du_dx_sq = du_dx * du_dx;

    float du_dy = ((*u)(j+1, i) - (*u)(j-1, i)) / (2.0 * dy);
    float dv_dx = ((*v)(j, i+1) - (*v)(j, i-1)) / (2.0 * dx);
    float cross = 2.0 * du_dy * dv_dx;

    float dv_dy_sq = dv_dy * dv_dy;

    (*b)(j, i) = rho * (1.0/dt * (du_dx + dv_dy) - du_dx_sq - cross - dv_dy_sq);
}

__global__ void compute_p_kernel(const matrx* b, matrx* p, const matrx* pn, int nx, int ny, float dx, float dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= nx-1 || j >= ny-1) return;

    float dx2 = dx*dx;
    float dy2 = dy*dy;

    (*p)(j,i) = (dy2 * ((*pn)(j, i+1) + (*pn)(j, i-1))
                     + dx2 * ((*pn)(j+1, i) + (*pn)(j-1, i))
                     - (*b)(j, i) * dx2 * dy2)
                     / (2.0 * (dx2 + dy2));
}

__global__ void compute_u_v_kernel(const matrx* p, matrx* u, matrx* v, const matrx* un, const matrx* vn, int nx, int ny, float dx, float dy, float dt, float rho, float nu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= nx-1 || j >= ny-1) return;

    float un_ji = (*un)(j, i);
    float vn_ji = (*vn)(j, i);

    float du_dx = ((*un)(j, i) - (*un)(j, i - 1)) / dx;
    float du_dy = ((*un)(j, i) - (*un)(j - 1, i)) / dy;

    float dp_dx = ((*p)(j, i + 1) - (*p)(j, i - 1)) / (2.0 * dx);

    float laplacian_u = ((*un)(j, i + 1) - 2.0 * un_ji + (*un)(j, i - 1)) / (dx * dx) +
                         ((*un)(j + 1, i) - 2.0 * un_ji + (*un)(j - 1, i)) / (dy * dy);

    (*u)(j, i) = un_ji
        - un_ji * dt * du_dx
        - un_ji * dt * du_dy
        - dt / (2.0 * rho) * dp_dx
        + nu * dt * laplacian_u;

    float dv_dx = ((*vn)(j, i) - (*vn)(j, i - 1)) / dx;
    float dv_dy = ((*vn)(j, i) - (*vn)(j - 1, i)) / dy;

    float dp_dy = ((*p)(j + 1, i) - (*p)(j - 1, i)) / (2.0 * dx);

    float laplacian_v = ((*vn)(j, i + 1) - 2.0 * vn_ji + (*vn)(j, i - 1)) / (dx * dx) +
                         ((*vn)(j + 1, i) - 2.0 * vn_ji + (*vn)(j - 1, i)) / (dy * dy);

    (*v)(j, i) = vn_ji
        - vn_ji * dt * dv_dx
        - vn_ji * dt * dv_dy
        - dt / (2.0 * rho) * dp_dy
        + nu * dt * laplacian_v;
}


int main() {
    int nx = 41;
    int ny = 41;
    int nt = 500;
    int nit = 50;
    float dx = 2. / (nx - 1);
    float dy = 2. / (ny - 1);
    float dt = .01;
    float rho = 1.;
    float nu = .02;

    int size = nx * ny;

    // allocating the host variables
    matrx u = {nx, ny, new float[size]()};
    matrx v = {nx, ny, new float[size]()};
    matrx p = {nx, ny, new float[size]()};
    matrx b = {nx, ny, new float[size]()};
    matrx un = {nx, ny, new float[size]()};
    matrx vn = {nx, ny, new float[size]()};
    matrx pn = {nx, ny, new float[size]()};

    // allocating the device variables

    float* d_u_arr, *d_v_arr, *d_p_arr, *d_b_arr, *d_un_arr, *d_vn_arr, *d_pn_arr;

    cudaMalloc(&d_u_arr, sizeof(float) * size);
    cudaMalloc(&d_v_arr, sizeof(float) * size);
    cudaMalloc(&d_p_arr, sizeof(float) * size);
    cudaMalloc(&d_b_arr, sizeof(float) * size);
    cudaMalloc(&d_un_arr, sizeof(float) * size);
    cudaMalloc(&d_vn_arr, sizeof(float) * size);
    cudaMalloc(&d_pn_arr, sizeof(float) * size);

    // using matrix pointers (instead of just full matrix instances) to avoid passing entire matrix field to global functions

    matrx *d_u, *d_v, *d_p, *d_b, *d_un, *d_vn, *d_pn;
    cudaMalloc(&d_u, sizeof(matrx));
    cudaMalloc(&d_v, sizeof(matrx));
    cudaMalloc(&d_p, sizeof(matrx));
    cudaMalloc(&d_b, sizeof(matrx));
    cudaMalloc(&d_un, sizeof(matrx));
    cudaMalloc(&d_vn, sizeof(matrx));
    cudaMalloc(&d_pn, sizeof(matrx));

    // attach device arrays to device matricies 
    // felt easiest to just do this directly on the device, 
    // and this code is only run once so it should not impact performance

    init_matrix_struct<<<1,1>>>(d_u, d_u_arr, nx, ny);
    init_matrix_struct<<<1,1>>>(d_v, d_v_arr, nx, ny);
    init_matrix_struct<<<1,1>>>(d_p, d_p_arr, nx, ny);
    init_matrix_struct<<<1,1>>>(d_b, d_b_arr, nx, ny);
    init_matrix_struct<<<1,1>>>(d_un, d_un_arr, nx, ny);
    init_matrix_struct<<<1,1>>>(d_vn, d_vn_arr, nx, ny);
    init_matrix_struct<<<1,1>>>(d_pn, d_pn_arr, nx, ny);
    cudaDeviceSynchronize();

    dim3 threadsPerBlock(32, 32); // 32x32 = 1024, limit on threads / block
    dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    ofstream ufile("u.dat");
    ofstream vfile("v.dat");
    ofstream pfile("p.dat");

    for (int n = 0; n < nt; n++) {
        // named sub-calculations inside my compute kernels to make the expression more readable...
        compute_b_kernel<<<numBlocks, threadsPerBlock>>>(d_b, d_u, d_v, nx, ny, dx, dy, dt, rho);
        cudaDeviceSynchronize();

        for (int it = 0; it < nit; it++) {
            // copy p to pn, running this instead of the old line as below
            // for (int j = 0; j < ny; j++) for (int i = 0; i < nx; i++) pn.arr[j + i * pn.nx] = p.arr[j + i * p.nx];
            cudaMemcpy(d_pn_arr, d_p_arr, sizeof(float)*size, cudaMemcpyDeviceToDevice);

            compute_p_kernel<<<numBlocks, threadsPerBlock>>>(d_b, d_p, d_pn, nx, ny, dx, dy);
            cudaDeviceSynchronize();

            // applying boundary conditions
            apply_pressure_bc_kernel<<<numBlocks, threadsPerBlock>>>(d_p, nx, ny);
            cudaDeviceSynchronize();
        }

        cudaMemcpy(d_un_arr, d_u_arr, sizeof(float)*size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_vn_arr, d_v_arr, sizeof(float)*size, cudaMemcpyDeviceToDevice);

        compute_u_v_kernel<<<numBlocks, threadsPerBlock>>>(d_p, d_u, d_v, d_un, d_vn, nx, ny, dx, dy, dt, rho, nu);
        cudaDeviceSynchronize();

        // applying boundary conditions
        apply_velocity_bc_kernel<<<numBlocks, threadsPerBlock>>>(d_u, d_v, nx, ny);
        cudaDeviceSynchronize();

        // used for logging, writes out array contents every 10 iterations
        if (n % 10 == 0) {
            cudaMemcpy(u.arr, d_u_arr, sizeof(float)*size, cudaMemcpyDeviceToHost);
            cudaMemcpy(v.arr, d_v_arr, sizeof(float)*size, cudaMemcpyDeviceToHost);
            cudaMemcpy(p.arr, d_p_arr, sizeof(float)*size, cudaMemcpyDeviceToHost);

            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++)
                    ufile << u(j, i) << " ";
            ufile << "\n";

            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++)
                    vfile << v(j, i) << " ";
            vfile << "\n";

            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++)
                    pfile << p(j, i) << " ";
            pfile << "\n";
        }
    }

    ufile.close();
    vfile.close();
    pfile.close();

    // free device memory for cleanliness
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_p);
    cudaFree(d_b);
    cudaFree(d_un);
    cudaFree(d_vn);
    cudaFree(d_pn);

    // free host memory for cleanliness
    delete[] u.arr;
    delete[] v.arr;
    delete[] p.arr;
    delete[] b.arr;
    delete[] un.arr;
    delete[] vn.arr;
    delete[] pn.arr;

    return 0;
}
