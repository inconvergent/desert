#define THREADS _THREADS_

__device__ inline void blend(float *img, const int kk, const int c,
                             const float ia, const float *rgba) {

  for (int i=0; i<c; i++){
    img[kk] = img[kk]*ia + rgba[0];
    img[kk+1] = img[kk+1]*ia + rgba[1];
    img[kk+2] = img[kk+2]*ia + rgba[2];
    img[kk+3] = img[kk+3]*ia + rgba[3];
  }
}

__global__ void dot(const int n,
                    float *img,
                    const int *dots,
                    const float *rgba){

  const int i = blockIdx.x*THREADS + threadIdx.x;

  if (i >= n){
    return;
  }

  const int ii = 2*i;
  const int kk = 4*dots[ii];
  const int c = dots[ii+1];

  const float a = rgba[3];
  const float ia = 1.0-a;
  blend(img, kk, c, ia, rgba);

}

