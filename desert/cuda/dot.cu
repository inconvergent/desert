#define THREADS _THREADS_

__device__ inline void blend(float *img,
                             const int r, const int kk,
                             const float *rgba){

  const int c = 4*r;
  const float ia = 1.0f-rgba[c+3];

  img[kk] = img[kk]*ia + rgba[c];
  img[kk+1] = img[kk+1]*ia + rgba[c+1];
  img[kk+2] = img[kk+2]*ia + rgba[c+2];
  img[kk+3] = img[kk+3]*ia + rgba[c+3];
}

__global__ void dot(const int n,
                    float *img,
                    const int *ind_count,
                    const float *rgba){

  const int i = blockIdx.x*THREADS + threadIdx.x;

  if (i >= n){
    return;
  }

  const int ii = 4*i;
  const int kk = 4*ind_count[ii];
  const int reps = ind_count[ii+1];
  const int start = ind_count[ii+2];

  if (reps < 1) {
    return;
  }

  for (int r=start; r<start+reps; r++){
    blend(img, r, kk, rgba);
  }

}

