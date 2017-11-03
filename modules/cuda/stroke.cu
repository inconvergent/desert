#define THREADS _THREADS_

__global__ void stroke(const int n,
                       const float *ab,
                       float *xy,
                       const int grains){

  const int i = blockIdx.x*THREADS + threadIdx.x;

  if (i >= n*grains){
    return;
  }

  const int ii = 2*i;
  const int k = 4*(int)floor((float)i/(float)grains);

  const float dx = ab[k+2] - ab[k];
  const float dy = ab[k+3] - ab[k+1];

  const float rnd = xy[ii];

  xy[ii] = ab[k] + rnd*dx;
  xy[ii+1] = ab[k+1] + rnd*dy;
}

