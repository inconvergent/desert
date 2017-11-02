#define THREADS _THREADS_

__global__ void box(const int n,
                    float *xy,
                    const float *s,
                    const float *mid){

  const int i = blockIdx.x*THREADS + threadIdx.x;

  if (i >= n){
    return;
  }

  const int ii = 2*i;

  xy[ii] = (1.0 - 2.0*xy[ii]) * s[0] + mid[0];
  xy[ii+1] = (1.0 - 2.0*xy[ii+1]) * s[1] + mid[1];
}

