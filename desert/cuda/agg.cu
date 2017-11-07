#define THREADS _THREADS_

__global__ void agg(const int n,
                    const int imsize,
                    const float *xy,
                    int *ind_count){

  const int i = blockIdx.x*THREADS + threadIdx.x;

  if (i >= n){
    return;
  }

  const int ii = 2*i;

  const int x = (int) (xy[ii] * (float)imsize);
  const int y = (int) (xy[ii+1] * (float)imsize);

  const int ind = x + y * imsize ;

  atomicAdd(&ind_count[ind], 1);

}

