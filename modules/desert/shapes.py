# -*- coding: utf-8 -*-

import pycuda.driver as cuda

from numpy import float32 as npfloat
from numpy import int32 as npint
from numpy import prod
from numpy import reshape
from numpy import zeros
from numpy import column_stack
from numpy.random import random

from modules.helpers import load_kernel
from modules.helpers import ind_filter


THREADS = 512


# http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels


class box():
  def __init__(self, s, mid, dens, threads=THREADS):

    try:
      sx, sy = s
    except TypeError:
      sx = s
      sy = s

    self.s = reshape([sx, sy], (1, 2)).astype(npfloat)
    self.mid = reshape(mid, (1, 2)).astype(npfloat)
    self.dens = dens

    self.threads = threads

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
    return '<box s: ({:0.4f} {:0.4f}) xy: ({:0.4f}, {:0.4f}) d: {:0.4f}>'\
        .format(self.s[0, 0], self.s[0, 1],
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
                     self._s, self._mid,
                     block=(self.threads, 1, 1),
                     grid=(blocks, 1))

    if verbose:
      print(self)

    return ind_filter(xy)


class strokes():
  def __init__(self, a, b, dens, threads=THREADS):

    a = reshape(a, (-1, 2)).astype(npfloat)
    b = reshape(b, (-1, 2)).astype(npfloat)

    assert a.shape[0] == b.shape[0]

    self.ab = column_stack((a, b))

    self.num = self.ab.shape[0]
    self.dens = dens

    self.threads = threads

    self._ab = cuda.mem_alloc(self.ab.nbytes)
    cuda.memcpy_htod(self._ab, self.ab)

    self.cuda_sample = load_kernel(
        'modules/cuda/strokes.cu',
        'strokes',
        subs={'_THREADS_': self.threads}
        )

  def __repr__(self):
    return '<strokes n: {:d} d: {:0.4f}>'.format(
        self.num,
        self.dens)

  def _get_n(self, imsize):
    return int(self.dens*imsize)

  def sample(self, imsize, verbose=False):

    grains = self._get_n(imsize)
    blocks = int(self.num*grains//self.threads + 1)
    shape = (self.num*grains, 2)

    xy = zeros(shape).astype(npfloat)
    xy[:, 0] = random(self.num*grains).astype(npfloat)

    self.cuda_sample(npint(self.num*grains),
                     self._ab,
                     cuda.InOut(xy),
                     npint(grains),
                     block=(self.threads, 1, 1),
                     grid=(blocks, 1))

    if verbose:
      print(self)

    return ind_filter(xy)

