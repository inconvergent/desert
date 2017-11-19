#define THREADS _THREADS_


__device__ __inline__ int get_seg(const int num_segments,
                                  const float x) {
  const float s = 1.0f / (float) num_segments;
  return (int) floor(x / s);
}


__device__ __inline__ float get_t(const int num_segments,
                                  const float x) {
  const float s = 1.0f / (float) num_segments;
  // TODO: avoid this?
  return fmodf(x, s) / s;
}


__global__ void bzspl(const int n,
                      const int grains,
                      const int num_segments,
                      const int nv,
                      const float *rnd,
                      const float *vpts,
                      float *xy) {

  // TODO: shared memory?
  const float M[] = {1, 0, 0, -2, 2, 0, 1, -2, 1};

  const int i = blockIdx.x*THREADS + threadIdx.x;
  const int ii = 2*i;

  const int k = (int)floor((float)i/(float)grains);
  const int skip = k*nv*2;

  if (i >= n) {
    return;
  }

  float t = get_t(num_segments, rnd[i]);
  int seg = get_seg(num_segments, rnd[i]);

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
      sc = skip + 2*(2*seg+col);
      tx += mr*vpts[sc];
      ty += mr*vpts[sc + 1];
    }
    x += pow(t, row) * tx;
    y += pow(t, row) * ty;
  }

  xy[ii] = x;
  xy[ii+1] = y;

}

