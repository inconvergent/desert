#define THREADS _THREADS_

__global__ void stroke(const int n,
                       const float *ab,
                       const float *rnd,
                       float *xy,
                       const int grains) {

  const int i = blockIdx.x*THREADS + threadIdx.x;
  const int ii = 2*i;

  if (i >= n) {
    return;
  }

  const int k = 4*(int)floor((float)i/(float)grains);

  const float dx = ab[k+2] - ab[k];
  const float dy = ab[k+3] - ab[k+1];

  const float r = rnd[i];

  xy[ii] = ab[k] + r*dx;
  xy[ii+1] = ab[k+1] + r*dy;

}

