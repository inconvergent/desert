#define THREADS _THREADS_

__global__ void stroke(const int n,
                       const int imsize,
                       const float *ab,
                       const float *rnd,
                       int *ind,
                       const int grains){

  const int i = blockIdx.x*THREADS + threadIdx.x;

  if (i >= n*grains){
    return;
  }

  const int k = 4*(int)floor((float)i/(float)grains);

  const float dx = ab[k+2] - ab[k];
  const float dy = ab[k+3] - ab[k+1];

  const float r = rnd[i];

  const float x = ab[k] + r*dx;
  const float y = ab[k+1] + r*dy;

  if (x < 0.0f || x >= 1.0f || y < 0.0f || y >= 1.0f){
    ind[i] = -1;
    return;
  }

  ind[i] = (int)(x*(float)imsize) + (int)(y*(float)imsize) * imsize;

}

