# -*- coding: utf-8 -*-

from json import loads

import pycuda.driver as cuda

import pkg_resources

from numpy import column_stack
from numpy import float32 as npfloat
from numpy import int32 as npint
from numpy import pi as PI
from numpy import prod
from numpy import reshape
from numpy import zeros
from numpy.random import random

from .helpers import ind_filter
from .helpers import is_verbose
from .helpers import json_array
from .helpers import load_kernel
from .helpers import pfloat

from .color import Rgba



THREADS = 256
TWOPI = 2.0*PI


def _load_color(o, data):
  c = data.get('rgba')
  if c is not None:
    return o.rgb(Rgba.from_json(c))
  return o


class basePrimitive():
  def __init__(self, threads):
    self.threads = threads
    self.rgba = None
    self.num = None

  def rgb(self, c):
    assert isinstance(c, Rgba)
    self.rgba = c
    return self

  def est(self, imsize):
    return self._get_n(imsize) * self.num

  def _get_n(self, *arg, **args):
    return NotImplemented

  def sample(self, *arg, **args):
    return NotImplemented

  def json(self, *arg, **args):
    return NotImplemented


class box(basePrimitive):
  def __init__(self, s, mid, dens, threads=THREADS):
    basePrimitive.__init__(self, threads)

    try:
      sx, sy = s
    except TypeError:
      sx = s
      sy = s

    self.s = reshape([sx, sy], (1, 2)).astype(npfloat)
    self.mid = reshape(mid, (-1, 2)).astype(npfloat)
    self.dens = dens

    self.num = self.mid.shape[0]

    self._s = None
    self._mid = None
    self._cuda_sample = None

  def __cuda_init(self):
    self._s = cuda.mem_alloc(self.s.nbytes)
    cuda.memcpy_htod(self._s, self.s)
    self._mid = cuda.mem_alloc(self.mid.nbytes)
    cuda.memcpy_htod(self._mid, self.mid)

    self._cuda_sample = load_kernel(
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
    return _load_color(box(data['s'], data['mid'], data['dens']), data)

  def _get_n(self, imsize):
    s = self.s
    return int(4*prod(s, axis=1)*self.dens*(imsize**2))

  def json(self):
    return {
        '_type': 'box',
        '_data': {
            'mid': json_array(self.mid),
            's': json_array(self.s).pop(),
            'dens': pfloat(self.dens),
            'rgba': self.rgba.json() if self.rgba is not None else None
            }
        }

  @is_verbose
  def sample(self, imsize, verbose=False):
    if self._cuda_sample is None:
      self.__cuda_init()

    grains = self._get_n(imsize)
    ng = self.num*grains
    blocks = int(ng//self.threads + 1)
    shape = (ng, 2)

    xy = random(shape).astype(npfloat)

    self._cuda_sample(npint(ng),
                      cuda.InOut(xy),
                      self._s, self._mid,
                      npint(grains),
                      block=(self.threads, 1, 1),
                      grid=(blocks, 1))

    return ind_filter(xy)


class circle(basePrimitive):
  def __init__(self, rad, mid, dens, threads=THREADS):
    basePrimitive.__init__(self, threads)
    self.rad = rad
    self.mid = reshape(mid, (-1, 2)).astype(npfloat)
    self.dens = dens

    self.num = self.mid.shape[0]

    self.threads = threads

    self._mid = None
    self._cuda_sample = None

  def __cuda_init(self):
    self._mid = cuda.mem_alloc(self.mid.nbytes)
    cuda.memcpy_htod(self._mid, self.mid)

    self._cuda_sample = load_kernel(
         pkg_resources.resource_filename('desert', 'cuda/circle.cu'),
        'circle',
        subs={'_THREADS_': self.threads}
        )

  def __repr__(self):
    return '<circle n: {:d} d: {:0.3f}>'.format(self.num, self.dens)

  def _get_n(self, imsize):
    return int(self.dens*PI*(self.rad*imsize)**2)

  def json(self):
    return {
        '_type': 'circle',
        '_data': {
            'rad': pfloat(self.rad),
            'mid': json_array(self.mid),
            'dens': pfloat(self.dens),
            'rgba': self.rgba.json() if self.rgba is not None else None
            }
        }

  @staticmethod
  def from_json(j):
    if isinstance(j, str):
      j = loads(j)
    data = j['_data']
    return _load_color(circle(data['rad'], data['mid'], data['dens']), data)

  @is_verbose
  def sample(self, imsize, verbose=False):
    if self._cuda_sample is None:
      self.__cuda_init()

    grains = self._get_n(imsize)
    ng = self.num*grains
    blocks = int(ng//self.threads + 1)
    shape = (ng, 3)

    xy = random(shape).astype(npfloat)

    self._cuda_sample(npint(ng),
                      cuda.InOut(xy),
                      npfloat(self.rad),
                      self._mid,
                      npint(grains),
                      block=(self.threads, 1, 1),
                      grid=(blocks, 1))

    return ind_filter(xy[:, :2])


class stroke(basePrimitive):
  def __init__(self, a, b, dens, threads=THREADS):
    basePrimitive.__init__(self, threads)

    a = reshape(a, (-1, 2)).astype(npfloat)
    b = reshape(b, (-1, 2)).astype(npfloat)

    assert a.shape[0] == b.shape[0]

    self.ab = column_stack((a, b))

    self.num = self.ab.shape[0]
    self.dens = dens

    self.threads = threads
    self._ab = None
    self._cuda_sample = None


  def __cuda_init(self):
    self._ab = cuda.mem_alloc(self.ab.nbytes)
    cuda.memcpy_htod(self._ab, self.ab)

    self._cuda_sample = load_kernel(
         pkg_resources.resource_filename('desert', 'cuda/stroke.cu'),
        'stroke',
        subs={'_THREADS_': self.threads}
        )

  def __repr__(self):
    return '<stroke n: {:d} d: {:0.3f}>'.format(self.num, self.dens)

  def _get_n(self, imsize):
    return int(self.dens*imsize)

  def json(self):
    return {
        '_type': 'stroke',
        '_data': {
            'a': json_array(self.ab[:, :2]),
            'b': json_array(self.ab[:, 2:]),
            'dens': pfloat(self.dens),
            'rgba': self.rgba.json() if self.rgba is not None else None
            }
        }

  @staticmethod
  def from_json(j):
    if isinstance(j, str):
      j = loads(j)
    data = j['_data']
    return _load_color(stroke(data['a'], data['b'], data['dens']), data)

  @is_verbose
  def sample(self, imsize, verbose=False):
    if self._cuda_sample is None:
      self.__cuda_init()

    grains = self._get_n(imsize)
    ng = self.num*grains
    blocks = int(ng//self.threads + 1)
    shape = (ng, 2)

    xy = zeros(shape).astype(npfloat)
    xy[:, 0] = random(ng).astype(npfloat)

    self._cuda_sample(npint(ng),
                      self._ab,
                      cuda.InOut(xy),
                      npint(grains),
                      block=(self.threads, 1, 1),
                      grid=(blocks, 1))

    return ind_filter(xy)

