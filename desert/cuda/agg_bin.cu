#define THREADS _THREADS_

__global__ void agg_bin(const int n,
                        int *ind_count,
                        const float *rgba,
                        const int *inds,
                        float *new_rgba){

  const int i = blockIdx.x*THREADS + threadIdx.x;


  if (i >= n){
    return;
  }

  const int ii = 4*i;
  const int r = atomicAdd(&ind_count[4*inds[i]+3], 1);
  const int rr = 4*r;

  new_rgba[rr] = rgba[ii];
  new_rgba[rr+1] = rgba[ii+1];
  new_rgba[rr+2] = rgba[ii+2];
  new_rgba[rr+3] = rgba[ii+3];

}

