# -*- coding: utf-8 -*-

from time import time

import matplotlib.pyplot as plt

import pkg_resources

from PIL import Image

from numpy import arange
from numpy import column_stack
from numpy import concatenate
from numpy import cumsum
from numpy import float32 as npfloat
from numpy import argsort
from numpy import int32 as npint
from numpy import pi
from numpy import repeat
from numpy import reshape
from numpy import row_stack
from numpy import zeros

import pycuda.driver as cuda

from .color import Rgba
from .color import black
from .color import white

from .helpers import load_kernel
from .helpers import unpack


TWOPI = pi*2


def build_ind_count(counts):
  ind_count_reduced = column_stack((arange(counts.shape[0]).astype(npint),
                                    counts))

  # ns = counts.nonzero()[0]
  # ind_count_reduced = ind_count_reduced[ns, :]

  cm = concatenate(([0], cumsum(ind_count_reduced[:, 1])[:-1])).astype(npint)
  return column_stack((ind_count_reduced, cm, cm)).astype(npint)



class Desert():

  def __init__(self, imsize, show=True, verbose=False):
    self.imsize = imsize
    self.imsize2 = imsize*imsize
    self.img = zeros((self.imsize2, 4), npfloat)

    self._img = cuda.mem_alloc(self.img.nbytes)

    self.threads = 256


    self.cuda_agg = load_kernel(
         pkg_resources.resource_filename('desert', 'cuda/agg.cu'),
        'agg',
        subs={'_THREADS_': self.threads}
        )

    self.cuda_agg_bin = load_kernel(
         pkg_resources.resource_filename('desert', 'cuda/agg_bin.cu'),
        'agg_bin',
        subs={'_THREADS_': self.threads}
        )

    self.cuda_dot = load_kernel(
         pkg_resources.resource_filename('desert', 'cuda/dot.cu'),
        'dot',
        subs={'_THREADS_': self.threads}
        )

    self.fig = None
    if show:
      self.fig = plt.figure()
      self.fig.patch.set_facecolor('gray')

    self._updated = False
    self.verbose = verbose
    self.__fg_set = set(['Rgba', 'Fg'])

    self.fg = None
    self.bg = None

  def __enter__(self):
    return self

  def __exit__(self, _type, val, tb):
    return

  def init(self, fg=black(0.01), bg=white()):
    self.fg = fg
    self.bg = bg
    self.clear()
    return self

  def set_fg(self, c):
    assert isinstance(c, Rgba)
    self.fg = c

  def set_bg(self, c):
    assert isinstance(c, Rgba)
    self.bg = c

  def clear(self, bg=None):
    if bg:
      self.img[:, :] = bg.rgba
    else:
      self.img[:, :] = self.bg.rgba
    cuda.memcpy_htod(self._img, self.img)
    self._updated = True

  def _draw(self, pts, colors, t0, count):
    if not pts:
      return
    imsize = self.imsize

    ind_count = zeros(self.imsize2, npint)
    colors = row_stack(colors).astype(npfloat)

    inds = concatenate(pts).astype(npint)
    _inds = cuda.mem_alloc(inds.nbytes)
    cuda.memcpy_htod(_inds, inds)

    aggn = inds.shape[0]
    self.cuda_agg(npint(aggn),
                  npint(imsize),
                  _inds,
                  cuda.InOut(ind_count),
                  block=(self.threads, 1, 1),
                  grid=(int(aggn//self.threads) + 1, 1))

    ind_count_map = build_ind_count(ind_count)
    _ind_count_map = cuda.mem_alloc(ind_count_map.nbytes)
    cuda.memcpy_htod(_ind_count_map, ind_count_map)

    sort_colors = zeros((aggn, 4), npfloat)
    _sort_colors = cuda.mem_alloc(sort_colors.nbytes)
    cuda.memcpy_htod(_sort_colors, sort_colors)

    self.cuda_agg_bin(npint(aggn),
                      _ind_count_map,
                      cuda.In(colors),
                      _inds,
                      _sort_colors,
                      block=(self.threads, 1, 1),
                      grid=(int(aggn//self.threads) + 1, 1))

    if self.verbose is not None:
      print('-- sampled primitives: {:d}. time: {:0.4f}'\
          .format(count, time()-t0))

    dotn, _ = ind_count_map.shape
    dt0 = time()
    self.cuda_dot(npint(dotn),
                  self._img,
                  _ind_count_map,
                  _sort_colors,
                  block=(self.threads, 1, 1),
                  grid=(int(dotn//self.threads) + 1, 1))

    if self.verbose is not None:
      print('-- drew dots: {:d}. time: {:0.4f}'.format(colors.shape[0],
                                                       time()-dt0))
    self._updated = True

  def draw(self, primitives):
    imsize = self.imsize
    # TODO: group based on estimated dots to draw?

    pts = []
    color_list = []

    count = 0
    t0 = time()

    sample_verbose = True if self.verbose == 'vv' else None

    for p in primitives:
      count += 1
      inds = p.sample(imsize, verbose=sample_verbose)
      colors = p.color_sample(imsize, self.fg)
      mask = inds > 0
      inds = inds[mask]
      colors = colors[mask, :]

      if inds.shape[0] > 0:
        pts.append(inds)
        color_list.append(colors)

    self._draw(pts, color_list, t0, count)

  def show(self, pause=0.00001):
    if not self.fig:
      print('-- warn: show is not enabled.')
      return

    imsize = self.imsize
    t0 = time()
    if self._updated:
      cuda.memcpy_dtoh(self.img, self._img)
      self._updated = False
    plt.imshow(Image.fromarray(unpack(self.img, imsize)))
    if self.verbose == 'vv':
      print('-- show. time: {:0.4f}'.format(time()-t0))

    plt.pause(pause)

  def save(self, fn):
    imsize = self.imsize
    print('-- wrote:', fn, (imsize, imsize))
    if self._updated:
      cuda.memcpy_dtoh(self.img, self._img)
      self._updated = False
    Image.fromarray(unpack(self.img, imsize)).save(fn)

