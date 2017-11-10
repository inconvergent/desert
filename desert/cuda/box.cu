#define THREADS _THREADS_

__global__ void box(const int n,
                    const int imsize,
                    float *rnd,
                    int *ind,
                    const float *s,
                    const float *mid,
                    const int grains){

  const int i = blockIdx.x*THREADS + threadIdx.x;

  if (i >= n){
    return;
  }

  const int ii = 2*i;
  const int k = 2*(int)floor((float)i/(float)grains);

  const float x = (1.0 - 2.0*rnd[ii]) * s[0] + mid[k];
  const float y = (1.0 - 2.0*rnd[ii+1]) * s[1] + mid[k+1];

  if (x < 0.0f || x >= 1.0f || y < 0.0f || y >= 1.0f){
    ind[i] = -1;
    return;
  }

  ind[i] = (int)(x*(float)imsize) + (int)(y*(float)imsize) * imsize;
}
