# -*- coding: utf-8 -*-

import pycuda.autoinit
import pycuda.driver as cuda

from numpy import float32 as npfloat
from numpy import int32 as npint
from numpy import prod
from numpy import reshape
from numpy.random import random

from modules.helpers import load_kernel
from modules.helpers import ind_filter



class box():
  def __init__(self, s, mid, dens):

    try:
      sx, sy = s
    except TypeError:
      sx = s
      sy = s

    self.s = reshape([sx, sy], (1, 2)).astype(npfloat)
    self.mid = reshape(mid, (1, 2)).astype(npfloat)
    self.dens = dens

    self.threads = 512

    self._s = cuda.mem_alloc(self.s.nbytes)
    cuda.memcpy_htod(self._s, self.s)
    self._mid = cuda.mem_alloc(self.mid.nbytes)
    cuda.memcpy_htod(self._mid, self.mid)

    self.cuda_sample = load_kernel(
        'modules/cuda/box.cu',
        'box',
        subs={'_THREADS_': self.threads}
        )

  def __repr__(self):
    return '<box s: ({:f} {:f}) xy: ({:f}, {:f}) d: {:f}>'.format(
        self.s[0, 0], self.s[0, 1],
        self.mid[0, 0], self.mid[0, 1],
        self.dens)

  def _get_n(self, imsize):
    s = self.s
    return int(4*prod(s, axis=1)*self.dens*(imsize**2))

  def sample(self, imsize, verbose=False):

    n = self._get_n(imsize)
    blocks = int(n//self.threads + 1)
    shape = (n, 2)

    xy = random(shape).astype(npfloat)
    self.cuda_sample(npint(n),
                     cuda.InOut(xy),
                     npfloat(imsize),
                     self._s, self._mid,
                     block=(self.threads, 1, 1),
                     grid=(blocks, 1))

    if verbose:
      print(self)

    return ind_filter(xy)

class stroke():
  def __init__(self, a, b, dens):

    self.a = reshape(a, (1, 2)).astype(npfloat)
    self.b = reshape(b, (1, 2)).astype(npfloat)
    self.dens = dens

  def __repr__(self):
    return '<stroke a: ({:f} {:f}) b: ({:f}, {:f}) d: {:f}>'.format(
        self.a[0, 0], self.a[0, 1],
        self.b[0, 0], self.b[0, 1],
        self.dens)

  def get_n(self, imsize):
    return int(self.dens*imsize)

