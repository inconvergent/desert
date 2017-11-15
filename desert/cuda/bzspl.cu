#define THREADS _THREADS_


__device__ __inline__ int get_seg(const int ns, const float x){
  const float s = 1.0f / (float) ns;
  return (int) floor(x / s);
}


__device__ __inline__ float get_t(const int ns, const float x) {
  const float s = 1.0f / (float) ns;
  // TODO: avoid this?
  return fmodf(x, s) / s;
}


__global__ void bzspl(const int n,
                      const int imsize,
                      const float *rnd,
                      const int ns,
                      const float *vpts,
                      int *ind){

  const float M[] = {1, 0, 0, -2, 2, 0, 1, -2, 1};

  const int i = blockIdx.x*THREADS + threadIdx.x;

  if (i >= n){
    return;
  }

  float t = get_t(ns, rnd[i]);
  int seg = get_seg(ns, rnd[i]);

  float x = 0.0f;
  float y = 0.0f;
  float tx;
  float ty;
  float mr;
  int sc;

  for (int row=0; row<3; row++) {
    tx = 0.0f;
    ty = 0.0f;
    for (int col=0; col<3; col++) {
      mr = M[row*3 + col];
      sc = 2*(2*seg+col);
      tx += mr*vpts[sc];
      ty += mr*vpts[sc + 1];
    }
    x += pow(t, row) * tx;
    y += pow(t, row) * ty;
  }

  if (x < 0.0f || x >= 1.0f || y < 0.0f || y >= 1.0f){
    ind[i] = -1;
    return;
  }

  ind[i] = (int)(x*(float)imsize) + (int)(y*(float)imsize) * imsize;
}

