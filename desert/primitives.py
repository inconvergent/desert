# -*- coding: utf-8 -*-

from json import dumps
from json import loads

import pycuda.driver as cuda

import pkg_resources

from numpy import float32 as npfloat
from numpy import int32 as npint
from numpy import prod
from numpy import reshape
from numpy import pi as PI
from numpy import zeros
from numpy import column_stack
from numpy.random import random

from .helpers import ind_filter
from .helpers import is_verbose
from .helpers import json_array
from .helpers import load_kernel
from .helpers import pfloat



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
         pkg_resources.resource_filename('desert', 'cuda/box.cu'),
        'box',
        subs={'_THREADS_': self.threads}
        )

  def __repr__(self):
    return '<box n: {:d} d: {:0.3f}>'.format(self.num, self.dens)

  @staticmethod
  def from_json(j):
    if isinstance(j, str):
      j = loads(j)
    data = j['_data']
    return box(data['s'], data['mid'], data['dens'])

  def _get_n(self, imsize):
    s = self.s
    return int(4*prod(s, axis=1)*self.dens*(imsize**2))

  def est(self, imsize):
    return self._get_n(imsize) * self.num

  def json(self):
    return dumps({
        '_type': 'box',
        '_data': {
            'mid': json_array(self.mid),
            's': json_array(self.s).pop(),
            'dens': pfloat(self.dens),
            }
        })

  @is_verbose
  def sample(self, imsize, verbose=False):

    grains = self._get_n(imsize)
    ng = self.num*grains
    blocks = int(ng//self.threads + 1)
    shape = (ng, 2)

    xy = random(shape).astype(npfloat)

    self.cuda_sample(npint(ng),
                     cuda.InOut(xy),
                     self._s, self._mid,
                     npint(grains),
                     block=(self.threads, 1, 1),
                     grid=(blocks, 1))

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
         pkg_resources.resource_filename('desert', 'cuda/circle.cu'),
        'circle',
        subs={'_THREADS_': self.threads}
        )

  def __repr__(self):
    return '<circle n: {:d} d: {:0.3f}>'.format(self.num, self.dens)

  def _get_n(self, imsize):
    return int(self.dens*PI*(self.rad*imsize)**2)

  def est(self, imsize):
    return self._get_n(imsize) * self.num

  def json(self):
    return dumps({
        '_type': 'circle',
        '_data': {
            'rad': pfloat(self.rad),
            'mid': json_array(self.mid),
            'dens': pfloat(self.dens),
            }
        })

  @staticmethod
  def from_json(j):
    if isinstance(j, str):
      j = loads(j)
    data = j['_data']
    return circle(data['rad'], data['mid'], data['dens'])

  @is_verbose
  def sample(self, imsize, verbose=False):

    grains = self._get_n(imsize)
    ng = self.num*grains
    blocks = int(ng//self.threads + 1)
    shape = (ng, 3)

    xy = random(shape).astype(npfloat)

    self.cuda_sample(npint(ng),
                     cuda.InOut(xy),
                     npfloat(self.rad),
                     self._mid,
                     npint(grains),
                     block=(self.threads, 1, 1),
                     grid=(blocks, 1))

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
         pkg_resources.resource_filename('desert', 'cuda/stroke.cu'),
        'stroke',
        subs={'_THREADS_': self.threads}
        )

  def __repr__(self):
    return '<stroke n: {:d} d: {:0.3f}>'.format(self.num, self.dens)

  def _get_n(self, imsize):
    return int(self.dens*imsize)

  def est(self, imsize):
    return self._get_n(imsize) * self.num

  def json(self):
    return dumps({
        '_type': 'stroke',
        '_data': {
            'a': json_array(self.ab[:, :2]),
            'b': json_array(self.ab[:, 2:]),
            'dens': pfloat(self.dens),
            }
        })

  @staticmethod
  def from_json(j):
    if isinstance(j, str):
      j = loads(j)
    data = j['_data']
    return stroke(data['a'], data['b'], data['dens'])

  @is_verbose
  def sample(self, imsize, verbose=False):

    grains = self._get_n(imsize)
    ng = self.num*grains
    blocks = int(ng//self.threads + 1)
    shape = (ng, 2)

    xy = zeros(shape).astype(npfloat)
    xy[:, 0] = random(ng).astype(npfloat)

    self.cuda_sample(npint(ng),
                     self._ab,
                     cuda.InOut(xy),
                     npint(grains),
                     block=(self.threads, 1, 1),
                     grid=(blocks, 1))

    return ind_filter(xy)

