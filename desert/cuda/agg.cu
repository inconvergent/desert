#define THREADS _THREADS_

__global__ void agg(const int n,
                    const int imsize,
                    const int *inds,
                    int *ind_count){

  const int i = blockIdx.x*THREADS + threadIdx.x;

  if (i >= n){
    return;
  }

  atomicAdd(&ind_count[inds[i]], 1);

}

