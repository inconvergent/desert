#define THREADS _THREADS_

/*__device__ inline void atomicSaxpy(float *adr,*/
/*                            const float ia,*/
/*                            const float k){*/
/*  float val;*/
/*  while (true){*/
/*    val = atomicExch(adr, -1.0f);*/
/*    if (val > -1.0f){*/
/*      val = ia*val + k;*/
/*      atomicExch(adr, val);*/
/*      break;*/
/*    }*/
/*  }*/
/*}*/

/*__device__ inline void atomicSaxpy(float *adr,*/
/*                            const float ia,*/
/*                            const float k){*/
/*  float old = -1.0f;*/
/*  float new_old;*/

/*  do {*/
/*    new_old = atomicExch(adr, -1.0f);*/
/*    new_old = ia*new_old + k;*/
/*  } while ((old = atomicExch(adr, new_old)) != -1.0f);*/
/*}*/

__device__ inline void atomicSaxpy(float *adr,
                            const float ia,
                            const float k){

  float old = atomicExch(adr, -1.0f);
  float new_old = old*ia + k;

  while ((old = atomicExch(adr, new_old)) != -1.0f) {
    new_old = atomicExch(adr, -1.0f);
    new_old = old*ia +k;
  }
}


__global__ void dot(const int n,
                    const int imsize,
                    float *img,
                    const float *xy,
                    const float *rgba){
  const int i = blockIdx.x*THREADS + threadIdx.x;

  if (i >= n){
    return;
  }

  const int ii = 2*i;
  const int x = (int)floor(xy[ii]* (float)imsize);
  const int y = (int)floor(xy[ii+1]* (float)imsize);

  if (x>=imsize || x<0 || y>=imsize || y<0){
    return;
  }

  const float a = rgba[3];
  const float ia = 1.0-a;

  const int ij = 4*(x*imsize+y);

  atomicSaxpy(&img[ij], ia, rgba[0]);
  atomicSaxpy(&img[ij+1], ia, rgba[1]);
  atomicSaxpy(&img[ij+2], ia, rgba[2]);
  atomicSaxpy(&img[ij+3], ia, rgba[3]);
}

