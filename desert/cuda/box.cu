#define THREADS _THREADS_

__global__ void box(const int n,
                    float *rnd,
                    float *xy,
                    const float *s,
                    const float *mid,
                    const int grains) {

  const int i = blockIdx.x*THREADS + threadIdx.x;

  if (i >= n) {
    return;
  }

  const int ii = 2*i;
  const int k = 2*(int)floor((float)i/(float)grains);

  xy[ii] = (1.0 - 2.0*rnd[ii]) * s[0] + mid[k];
  xy[ii+1] = (1.0 - 2.0*rnd[ii+1]) * s[1] + mid[k+1];

}
