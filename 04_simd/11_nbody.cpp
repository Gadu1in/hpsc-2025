#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  for(int i=0; i<N; i++) {

    __m512 xi = _mm512_set1_ps(x[i]); // Loading same values into the entire register
    __m512 yi = _mm512_set1_ps(y[i]);
    // __m512 fxi = _mm512_set1_ps(fx[i]);
    // __m512 fyj = _mm512_set1_ps(fy[i]);

    __m512 xj = _mm512_load_ps(x); // Loads all the values from x into the register
    __m512 yj = _mm512_load_ps(x);

    __m512 rx = _mm512_sub_ps(xi, xj); // This includes the x[i] - x[i], -> 0, will check later
    __m512 ry = _mm512_sub_ps(yi, yj);

    
    __m512 rinv = _mm512_add_ps(_mm512_mul_ps(rx, rx), _mm512_mul_ps(ry, ry));

    // Find all elements that are equal to zero in rinv
    __mmask16 rinvmask = _mm512_cmp_ps_mask(ry, _mm512_set1_ps(0.0f), _CMP_GT_OQ);

    // Compute 1/r, except when r = 0. Then just return 0 for that value instead.
    __m512 r = _mm512_mask_rsqrt14_ps(_mm512_set1_ps(0.0f), rinvmask, rinv);

    // Now we have 1/r, so we compute rx * m * 1 / r^3. 
    // We don't have to use mask because those values in r3 are already 0
    __m512 mvec = _mm512_load_ps(m);
    __m512 r3 = _mm512_mul_ps(r, _mm512_mul_ps(r, r));
    __m512 mvecr3 = _mm512_mul_ps(mvec, r3);
    __m512 vfxi = _mm512_mul_ps(rx, mvecr3);
    __m512 vfyi = _mm512_mul_ps(ry, mvecr3);

    // Finally sum all the values and load them to fx[i]
    fx[i] = - _mm512_reduce_add_ps(vfxi);
    fy[i] = - _mm512_reduce_add_ps(vfyi);


    // for(int j=0; j<N; j++) {

    //   __m512 mvec = _mm512_set1_ps(m[j]);

    //   if(i != j) {
    //     float rx = x[i] - x[j];
    //     float ry = y[i] - y[j];
    //     float r = std::sqrt(rx * rx + ry * ry);
    //     fx[i] -= rx * m[j] / (r * r * r);
    //     fy[i] -= ry * m[j] / (r * r * r);
    //   }
    // }

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
