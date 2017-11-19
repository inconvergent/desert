#define THREADS _THREADS_
#define PI 3.141592654f

__global__ void circle(const int n,
                       const float *rnd,
                       float *xy,
                       const float rad,
                       const float *mid,
                       const int grains) {

  const int i = blockIdx.x*THREADS + threadIdx.x;

  if (i >= n) {
    return;
  }

  const int ii = 2*i;
  const int iii = 3*i;
  const int k = 2*(int)floor((float)i/(float)grains);

  const float t = 2 * PI * rnd[iii];
  const float u = rnd[iii+1] + rnd[iii+2];

  float r;
  if (u>1.0f){
    r = rad*(2.0f-u);
  } else {
    r = rad*u;
  }

  xy[ii] = mid[k] + r * cos(t);
  xy[ii+1] = mid[k+1] + r *sin(t);

}

