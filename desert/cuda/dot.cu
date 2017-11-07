#define THREADS _THREADS_

__device__ inline void blend(float *img, const int r, const int kk,
    const int *ind_color, const float *rgba) {

  const int ci = 4*ind_color[r];
  const float ia = 1.0f-rgba[ci+3];

  img[kk] = img[kk]*ia + rgba[ci];
  img[kk+1] = img[kk+1]*ia + rgba[ci+1];
  img[kk+2] = img[kk+2]*ia + rgba[ci+2];
  img[kk+3] = img[kk+3]*ia + rgba[ci+3];
}

__global__ void dot(const int n,
                    float *img,
                    const int *ind_color,
                    const int *ind_count,
                    const float *rgba){

  const int i = blockIdx.x*THREADS + threadIdx.x;

  if (i >= n){
    return;
  }

  const int ii = 3*i;
  const int kk = 4*ind_count[ii];
  const int reps = ind_count[ii+1];
  const int start = ind_count[ii+2];

  for (int r=start; r<start+reps; r++){
    blend(img, r, kk, ind_color, rgba);
  }

}

