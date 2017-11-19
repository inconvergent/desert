# -*- coding: utf-8 -*-

from os import getenv

from json import loads
from functools import wraps

import pycuda.driver as cuda
from pycuda.curandom import XORWOWRandomNumberGenerator as Rgen

import pkg_resources

from numpy import column_stack
from numpy import dstack
from numpy import float32 as npfloat
from numpy import int32 as npint
from numpy import pi as PI
from numpy import prod
from numpy import reshape
from numpy import roll
from numpy import tile
from numpy import zeros

from .helpers import is_verbose
from .helpers import json_array
from .helpers import load_kernel
from .helpers import pfloat

from .color import Rgba

RGEN = Rgen(offset=0)



THREADS = int(getenv('THREADS', 512))
TWOPI = 2.0*PI


_cuda_sample_box = load_kernel(
    pkg_resources.resource_filename('desert', 'cuda/box.cu'),
    'box',
    subs={'_THREADS_': THREADS}
    )

_cuda_sample_circle = load_kernel(
    pkg_resources.resource_filename('desert', 'cuda/circle.cu'),
    'circle',
    subs={'_THREADS_': THREADS}
    )

_cuda_sample_stroke = load_kernel(
    pkg_resources.resource_filename('desert', 'cuda/stroke.cu'),
    'stroke',
    subs={'_THREADS_': THREADS}
    )

_cuda_sample_bzspl = load_kernel(
    pkg_resources.resource_filename('desert', 'cuda/bzspl.cu'),
    'bzspl',
    subs={'_THREADS_': THREADS}
    )

def add_noise(f):
  @wraps(f)
  def inside(*args, **kwargs):
    res = f(*args, **kwargs)
    self = args[0]
    if self.noise is not None:
      rad = npfloat(self.noise)
      ng = res.shape[0]
      xy = zeros((ng, 2), npfloat)
      mid = zeros((1, 2), npfloat)
      _cuda_sample_circle(npint(ng),
                          RGEN.gen_uniform((ng, 3), npfloat),
                          cuda.Out(xy),
                          rad,
                          cuda.In(mid),
                          npint(ng),
                          block=(THREADS, 1, 1),
                          grid=(int(ng//THREADS + 1), 1))
      return res + xy
    return res
  return inside


def _load_color(o, data):
  cc = data.get('rgba')
  if cc is not None:
    if isinstance(cc, list):
      return o.rgb([Rgba.from_json(c) for c in cc])
    return o.rgb(Rgba.from_json(cc))
  return o


def _export_color(cc):
  if isinstance(cc, list):
    return [c.json() for c in cc]
  return cc.json()


class basePrimitive():
  def __init__(self):
    self.rgba = None
    self.num = None
    self.dens = None
    self.noise = None
    self._cinit = False

  def __repr__(self):
    return '<{:s} n: {:d} d: {:0.3f}{:s}>'\
        .format(self.__class__.__name__, self.num,
                self.dens, ' *' if self.rgba else '')

  def has_rgb(self):
    if self.rgba is None:
      return False
    return True

  def color_sample(self, imsize, cc):
    num = self.num
    grains = self._get_n(imsize)
    ng = grains * num

    cc = self.rgba if self.rgba is not None else cc

    if isinstance(cc, Rgba):
      res = reshape(tile(cc.rgba, ng), (ng, 4))
    else:
      res = reshape([tile(c.rgba, grains) for c in cc], (ng, 4))

    return res.astype(npfloat)

  def rgb(self, cc):
    if self.num == 1:
      assert isinstance(cc, Rgba), 'not an Rgba instance'

    elif self.num > 1:

      if not isinstance(cc, Rgba):
        assert len(cc) == self.num, 'inconsistent number of colors'
        for c in cc:
          assert isinstance(c, Rgba), 'not an Rgba instance'

    self.rgba = cc
    return self

  def est(self, imsize):
    return self._get_n(imsize) * self.num

  def _get_n(self, imsize):
    return NotImplemented

  def sample(self, imsize, verbose=False):
    return NotImplemented

  def json(self):
    return NotImplemented


class box(basePrimitive):
  def __init__(self, s, mid, dens, noise=None):
    basePrimitive.__init__(self)

    try:
      sx, sy = s
    except TypeError:
      sx = s
      sy = s

    self.s = reshape([sx, sy], (1, 2)).astype(npfloat)
    self.mid = reshape(mid, (-1, 2)).astype(npfloat)
    self.dens = dens
    self.noise = noise

    self.num = self.mid.shape[0]

    self._s = None
    self._mid = None

  def __cuda_init(self):
    self._s = cuda.mem_alloc(self.s.nbytes)
    cuda.memcpy_htod(self._s, self.s)
    self._mid = cuda.mem_alloc(self.mid.nbytes)
    cuda.memcpy_htod(self._mid, self.mid)
    self._cinit = True

  @staticmethod
  def from_json(j):
    if isinstance(j, str):
      j = loads(j)
    data = j['_data']
    return _load_color(box(data['s'], data['mid'], dens=data['dens'],
                           noise=data['noise']), data)

  def _get_n(self, imsize):
    s = self.s
    return int(4*prod(s, axis=1)*self.dens*(imsize**2))

  def json(self):
    return {
        '_type': 'box',
        '_data': {
            'num': self.num,
            'mid': json_array(self.mid),
            's': json_array(self.s).pop(),
            'dens': pfloat(self.dens),
            'noise': self.noise,
            'rgba': _export_color(self.rgba) if self.rgba is not None else None
            }
        }

  @add_noise
  @is_verbose
  def sample(self, imsize, verbose=False):
    if not self._cinit:
      self.__cuda_init()

    grains = self._get_n(imsize)
    ng = self.num*grains
    shape = (ng, 2)

    xy = zeros((ng, 2), npfloat)

    _cuda_sample_box(npint(ng),
                     RGEN.gen_uniform(shape, npfloat),
                     cuda.Out(xy),
                     self._s, self._mid,
                     npint(grains),
                     block=(THREADS, 1, 1),
                     grid=(int(ng//THREADS + 1), 1))

    return xy


class circle(basePrimitive):
  def __init__(self, rad, mid, dens, noise=None):
    basePrimitive.__init__(self)
    self.rad = rad
    self.mid = reshape(mid, (-1, 2)).astype(npfloat)
    self.dens = dens
    self.noise = noise

    self.num = self.mid.shape[0]

    self._mid = None
    self._cuda_init = False

  def __cuda_init(self):
    self._mid = cuda.mem_alloc(self.mid.nbytes)
    cuda.memcpy_htod(self._mid, self.mid)

  def _get_n(self, imsize):
    return int(self.dens*PI*(self.rad*imsize)**2)

  def json(self):
    return {
        '_type': 'circle',
        '_data': {
            'num': self.num,
            'rad': pfloat(self.rad),
            'mid': json_array(self.mid),
            'dens': pfloat(self.dens),
            'noise': self.noise,
            'rgba': _export_color(self.rgba) if self.rgba is not None else None
            }
        }

  @staticmethod
  def from_json(j):
    if isinstance(j, str):
      j = loads(j)
    data = j['_data']
    return _load_color(circle(data['rad'], data['mid'], dens=data['dens'],
                              noise=data['noise']), data)

  @add_noise
  @is_verbose
  def sample(self, imsize, verbose=False):
    if not self._cinit:
      self.__cuda_init()

    grains = self._get_n(imsize)
    ng = self.num*grains
    shape = (ng, 3)

    xy = zeros((ng, 2), npfloat)

    _cuda_sample_circle(npint(ng),
                        RGEN.gen_uniform(shape, npfloat),
                        cuda.Out(xy),
                        npfloat(self.rad),
                        self._mid,
                        npint(grains),
                        block=(THREADS, 1, 1),
                        grid=(int(ng//THREADS + 1), 1))

    return xy


class stroke(basePrimitive):
  def __init__(self, a, b, dens, noise=None):
    basePrimitive.__init__(self)

    a = reshape(a, (-1, 2)).astype(npfloat)
    b = reshape(b, (-1, 2)).astype(npfloat)

    assert a.shape[0] == b.shape[0], 'inconsistent number of points in a, b'

    self.ab = column_stack((a, b))

    self.num = self.ab.shape[0]
    self.dens = dens
    self.noise = noise

    self._ab = None

  def __cuda_init(self):
    self._ab = cuda.mem_alloc(self.ab.nbytes)
    cuda.memcpy_htod(self._ab, self.ab)

  def _get_n(self, imsize):
    return int(self.dens*imsize)

  def json(self):
    return {
        '_type': 'stroke',
        '_data': {
            'num': self.num,
            'a': json_array(self.ab[:, :2]),
            'b': json_array(self.ab[:, 2:]),
            'dens': pfloat(self.dens),
            'noise': self.noise,
            'rgba': _export_color(self.rgba) if self.rgba is not None else None
            }
        }

  @staticmethod
  def from_json(j):
    if isinstance(j, str):
      j = loads(j)
    data = j['_data']
    return _load_color(stroke(data['a'], data['b'], data['dens'],
                              noise=data['noise']), data)

  @add_noise
  @is_verbose
  def sample(self, imsize, verbose=False):
    if not self._cinit:
      self.__cuda_init()

    grains = self._get_n(imsize)
    ng = self.num*grains

    xy = zeros((ng, 2), npfloat)

    _cuda_sample_stroke(npint(ng),
                        self._ab,
                        RGEN.gen_uniform(ng, npfloat),
                        cuda.Out(xy),
                        npint(grains),
                        block=(THREADS, 1, 1),
                        grid=(int(ng//THREADS + 1), 1))

    return xy


class bzspl(basePrimitive):
  def __init__(self, pts, dens, closed=False, noise=None):
    basePrimitive.__init__(self)

    self.num = len(pts)

    # pts = reshape(pts, (-1, 2, self.num)).astype(npfloat)
    pts = dstack(pts).astype(npfloat)

    assert pts.shape[0] > 2, 'must have at least 3 points'

    self.pts = pts

    self.closed = closed
    self.dens = dens
    self.noise = noise

    ctrlpts = pts.shape[0]

    if closed:
      self.num_segments = ctrlpts
      self.nv = 2*ctrlpts + 1
    else:
      self.num_segments = ctrlpts-2
      self.nv = 2*ctrlpts-3

    if closed:
      self.vpts = self._get_vpts_closed(pts, self.nv)
    else:
      self.vpts = self._get_vpts_open(pts, self.nv)

  def __cuda_init(self):
    self._vpts = cuda.mem_alloc(self.vpts.nbytes)
    cuda.memcpy_htod(self._vpts, self.vpts)

  def _get_vpts_open(self, pts, nv):
    res = zeros((nv, 2, self.num), npfloat)
    res[:2:, :, :] = pts[:2, :, :]
    res[-2:, :, :] = pts[-2:, :, :]
    res[3:-2:2, :, :] = pts[2:-2, :, :]
    res[2:-2:2, :, :] = (pts[1:-2, :, :] + pts[2:-1, :, :])*0.5
    return reshape(res.swapaxes(1, 2), (-1, 2), 'F').copy()

  def _get_vpts_closed(self, pts, nv):
    res = zeros((nv, 2, self.num), npfloat)
    rolled = roll(pts, -1, axis=0)
    res[1::2, :, :] = rolled
    res[:-2:2, :, :] = (rolled + pts)*0.5
    res[-1, :, :] = res[0, :, :]
    return reshape(res.swapaxes(1, 2), (-1, 2), 'F').copy()

  def _get_n(self, imsize):
    return int(self.dens*imsize)

  def json(self):
    return {
        '_type': 'bzspl',
        '_data': {
            'num': self.num,
            'pts': [json_array(self.pts[:, :, i]) for i in range(self.num)],
            'closed': self.closed,
            'dens': pfloat(self.dens),
            'noise': self.noise,
            'rgba': _export_color(self.rgba) if self.rgba is not None else None
            }
        }

  @staticmethod
  def from_json(j):
    if isinstance(j, str):
      j = loads(j)
    data = j['_data']
    return _load_color(
        bzspl(data['pts'],
              dens=data['dens'],
              closed=data['closed'],
              noise=data['noise']), data)

  @add_noise
  @is_verbose
  def sample(self, imsize, verbose=False):
    if not self._cinit:
      self.__cuda_init()

    grains = self._get_n(imsize)
    ng = self.num*grains
    xy = zeros((ng, 2), npfloat)

    _cuda_sample_bzspl(npint(ng),
                       npint(grains),
                       npint(self.num_segments),
                       npint(self.nv),
                       RGEN.gen_uniform(ng, npfloat),
                       self._vpts,
                       cuda.InOut(xy),
                       block=(THREADS, 1, 1),
                       grid=(int(ng//THREADS + 1), 1))

    return xy

