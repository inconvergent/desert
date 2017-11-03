# -*- coding: utf-8 -*-

from time import time

import pycuda.driver as cuda

from numpy import float32 as npfloat
from numpy import int32 as npint
from numpy import prod
from numpy import reshape
from numpy import pi as PI
from numpy import zeros
from numpy import column_stack
from numpy.random import random

from modules.helpers import load_kernel
from modules.helpers import ind_filter


THREADS = 256
TWOPI = 2.0*PI


# http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels


class box():
  def __init__(self, s, mid, dens, threads=THREADS):

    try:
      sx, sy = s
    except TypeError:
      sx = s
      sy = s

    self.s = reshape([sx, sy], (1, 2)).astype(npfloat)
    self.mid = reshape(mid, (-1, 2)).astype(npfloat)
    self.dens = dens

    self.num = self.mid.shape[0]

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
    return '<box n: {: 8d} d: {:0.3f}>'.format(self.num, self.dens)

  def _get_n(self, imsize):
    s = self.s
    return int(4*prod(s, axis=1)*self.dens*(imsize**2))

  def sample(self, imsize, verbose=False):

    grains = self._get_n(imsize)
    ng = self.num*grains
    blocks = int(ng//self.threads + 1)
    shape = (ng, 2)

    st0 = time()
    xy = random(shape).astype(npfloat)

    self.cuda_sample(npint(ng),
                     cuda.InOut(xy),
                     self._s, self._mid,
                     npint(grains),
                     block=(self.threads, 1, 1),
                     grid=(blocks, 1))

    if verbose is not None:
      print('## {:s}        time: {:0.4f}'.format(str(self), time()-st0))

    return ind_filter(xy)


class circle():
  def __init__(self, rad, mid, dens, threads=THREADS):


    self.rad = rad
    self.mid = reshape(mid, (-1, 2)).astype(npfloat)
    self.dens = dens

    self.num = self.mid.shape[0]

    self.threads = threads

    self._mid = cuda.mem_alloc(self.mid.nbytes)
    cuda.memcpy_htod(self._mid, self.mid)

    self.cuda_sample = load_kernel(
        'modules/cuda/circle.cu',
        'circle',
        subs={'_THREADS_': self.threads}
        )

  def __repr__(self):
    return '<circle n: {: 8d} d: {:0.3f}>'.format(self.num, self.dens)

  def _get_n(self, imsize):
    return int(self.dens*PI*(self.rad*imsize)**2)

  def sample(self, imsize, verbose=False):

    grains = self._get_n(imsize)
    ng = self.num*grains
    blocks = int(ng//self.threads + 1)
    shape = (ng, 3)

    st0 = time()
    xy = random(shape).astype(npfloat)

    self.cuda_sample(npint(ng),
                     cuda.InOut(xy),
                     npfloat(self.rad),
                     self._mid,
                     npint(grains),
                     block=(self.threads, 1, 1),
                     grid=(blocks, 1))

    if verbose is not None:
      print('## {:s}     time: {:0.4f}'.format(str(self), time()-st0))

    return ind_filter(xy[:, :2])


class stroke():
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
        'modules/cuda/stroke.cu',
        'stroke',
        subs={'_THREADS_': self.threads}
        )

  def __repr__(self):
    return '<stroke n: {: 8d} d: {:0.3f}>'.format(self.num, self.dens)

  def _get_n(self, imsize):
    return int(self.dens*imsize)

  def sample(self, imsize, verbose=False):

    grains = self._get_n(imsize)
    ng = self.num*grains
    blocks = int(ng//self.threads + 1)
    shape = (ng, 2)

    st0 = time()
    xy = zeros(shape).astype(npfloat)
    xy[:, 0] = random(ng).astype(npfloat)

    self.cuda_sample(npint(ng),
                     self._ab,
                     cuda.InOut(xy),
                     npint(grains),
                     block=(self.threads, 1, 1),
                     grid=(blocks, 1))

    if verbose is not None:
      print('## {:s}     time: {:0.4f}'.format(str(self), time()-st0))

    return ind_filter(xy)

