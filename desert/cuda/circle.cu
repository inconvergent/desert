#define THREADS _THREADS_
#define PI 3.141592654f

__global__ void circle (const int n,
                        const int imsize,
                        const float *rnd,
                        int *ind,
                        const float rad,
                        const float *mid,
                        const int grains
                        ){

  const int i = blockIdx.x*THREADS + threadIdx.x;

  if (i >= n*grains){
    return;
  }

  const int ii = 3*i;
  const int k = 2*(int)floor((float)i/(float)grains);

  const float t = 2 * PI * rnd[ii];
  const float u = rnd[ii+1] + rnd[ii+2];

  float r;
  if (u>1.0f){
    r = rad*(2.0f-u);
  } else {
    r = rad*u;
  }

  const float x = mid[k] + r * cos(t);
  const float y = mid[k+1] + r *sin(t);

  if (x < 0.0f || x >= 1.0f || y < 0.0f || y >= 1.0f){
    ind[i] = -1;
    return;
  }

  ind[i] = (int)(x*(float)imsize) + (int)(y*(float)imsize) * imsize;

}

