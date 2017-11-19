#define THREADS _THREADS_

__global__ void agg(const int n,
                    const int imsize,
                    const float *xy,
                    int *inds,
                    int *ind_count){

  const int i = blockIdx.x*THREADS + threadIdx.x;
  const int ii = 2*i;

  if (i >= n){
    return;
  }

  const float x = xy[ii];
  const float y = xy[ii + 1];

  if (x < 0.0f || x >= 1.0f || y < 0.0f || y >= 1.0f){
    inds[i] = -1;
    return;
  }

  const int ind = (int)(x*(float)imsize) + (int)(y*(float)imsize) * imsize;
  inds[i] = ind;
  atomicAdd(&ind_count[ind], 1);

}

