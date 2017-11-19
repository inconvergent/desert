# -*- coding: utf-8 -*-

from os import getenv
from time import time

import matplotlib.pyplot as plt

import pkg_resources

from PIL import Image

from numpy import arange
from numpy import argsort
from numpy import column_stack
from numpy import concatenate
from numpy import cumsum
from numpy import vstack
from numpy import float32 as npfloat
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


THREADS = int(getenv('THREADS', 512))
TWOPI = pi*2


def _build_ind_count(counts):
  ind_count_reduced = column_stack((arange(counts.shape[0]).astype(npint),
                                    counts))
  cm = concatenate(([0], cumsum(ind_count_reduced[:, 1])[:-1])).astype(npint)
  return column_stack((ind_count_reduced, cm, cm)).astype(npint)



class Desert():

  def __init__(self, imsize, show=True,
               gsamples=1000000, verbose=False):
    self.imsize = imsize
    self.imsize2 = imsize*imsize
    self.img = zeros((self.imsize2, 4), npfloat)

    self._img = cuda.mem_alloc(self.img.nbytes)
    self.gsamples = gsamples

    assert self.gsamples >= 5000, 'you must set gsamples to at least 5000.'

    self.cuda_agg = load_kernel(
         pkg_resources.resource_filename('desert', 'cuda/agg.cu'),
        'agg',
        subs={'_THREADS_': THREADS}
        )

    self.cuda_agg_bin = load_kernel(
         pkg_resources.resource_filename('desert', 'cuda/agg_bin.cu'),
        'agg_bin',
        subs={'_THREADS_': THREADS}
        )

    self.cuda_dot = load_kernel(
         pkg_resources.resource_filename('desert', 'cuda/dot.cu'),
        'dot',
        subs={'_THREADS_': THREADS}
        )

    self.fig = None
    if show:
      self.fig = plt.figure()
      self.fig.patch.set_facecolor('gray')

    self._updated = False
    self.verbose = verbose

    self.fg = None
    self.bg = None

    self._gdraw_reset()

  def __enter__(self):
    return self

  def __exit__(self, _type, val, tb):
    return

  def _gdraw_reset(self):
    self.est = 0
    self.tdraw = 0
    self.count = 0
    self.color_list = []
    self.pts = []
    self._gupdated = True

  def init(self, fg=black(0.01), bg=white()):
    self.fg = fg
    self.bg = bg
    self.clear()
    return self

  def set_fg(self, c):
    assert isinstance(c, Rgba)
    self.fg = c
    return self

  def set_bg(self, c):
    assert isinstance(c, Rgba)
    self.bg = c
    return self

  def clear(self, bg=None):
    if bg:
      self.img[:, :] = bg.rgba
    else:
      self.img[:, :] = self.bg.rgba
    cuda.memcpy_htod(self._img, self.img)
    self._updated = True
    return self

  def _draw(self, pts, colors):
    if not pts:
      return False
    imsize = self.imsize

    dt0 = time()

    ind_count = zeros(self.imsize2, npint)
    colors = row_stack(colors).astype(npfloat)

    xy = vstack(pts).astype(npfloat)
    inds = zeros(xy.shape[0], npint)

    self.cuda_agg(npint(inds.shape[0]),
                  npint(imsize),
                  cuda.In(xy),
                  cuda.InOut(inds),
                  cuda.InOut(ind_count),
                  block=(THREADS, 1, 1),
                  grid=(int(inds.shape[0]//THREADS) + 1, 1))

    mask = inds > -1

    if not mask.any():
      print('-- no dots to draw. time: {:0.4f}'.format(time()-dt0))
      return False

    # xy = xy[mask, :]
    inds = inds[mask]
    colors = colors[mask]

    ind_count_map = _build_ind_count(ind_count)
    _ind_count_map = cuda.mem_alloc(ind_count_map.nbytes)
    cuda.memcpy_htod(_ind_count_map, ind_count_map)

    sort_colors = zeros((inds.shape[0], 4), npfloat)
    _sort_colors = cuda.mem_alloc(sort_colors.nbytes)
    cuda.memcpy_htod(_sort_colors, sort_colors)

    self.cuda_agg_bin(npint(inds.shape[0]),
                      _ind_count_map,
                      cuda.In(colors),
                      cuda.In(inds),
                      _sort_colors,
                      block=(THREADS, 1, 1),
                      grid=(int(inds.shape[0]//THREADS) + 1, 1))

    dotn, _ = ind_count_map.shape
    self.cuda_dot(npint(dotn),
                  self._img,
                  _ind_count_map,
                  _sort_colors,
                  block=(THREADS, 1, 1),
                  grid=(int(dotn//THREADS) + 1, 1))

    if self.verbose is not None:
      print('-- drew dots: {:d}. time: {:0.4f}'.format(colors.shape[0],
                                                       time()-dt0))
    self._updated = True
    return True

  def draw(self, primitives):
    imsize = self.imsize

    pts = []
    color_list = []
    count = 0
    est = 0
    t0 = time()

    sample_verbose = True if self.verbose == 'vv' else None

    for p in primitives:
      count += p.num
      inds = p.sample(imsize, verbose=sample_verbose)
      colors = p.color_sample(imsize, self.fg)
      est += p.est(imsize)

      if inds.shape[0] > 0:
        pts.append(inds)
        color_list.append(colors)

    if self.verbose is not None:
      print('-- sampled primitives: {:d} ({:d}). time: {:0.4f}'\
          .format(count, est, time()-t0))

    self._draw(pts, color_list)
    return self

  def _gdraw(self, force=False):
    if force or self.est > self.gsamples:
      if force:
        print('.. gsamples force: drawing.')
      else:
        print('.. hit gsamples limit: drawing.')
      self._draw(self.pts, self.color_list)

      if self.verbose is not None:
        print('-- sampled primitives: {:d} ({:d}). time: {:0.4f}'\
            .format(self.count, self.est, self.tdraw))
      self._gdraw_reset()
    else:
      self._gupdated = False

  def gforce(self):
    self._gdraw(force=True)
    return self

  def gdraw(self, primitives, force=False):
    imsize = self.imsize
    sample_verbose = True if self.verbose == 'vv' else None

    t0 = time()

    for p in primitives:
      self.count += p.num
      xy = p.sample(imsize, verbose=sample_verbose)
      colors = p.color_sample(imsize, self.fg)
      self.est += p.est(imsize)
      # mask = inds > 0
      # inds = inds[mask]
      # colors = colors[mask, :]

      if xy.shape[0] > 0:
        self.pts.append(xy)
        self.color_list.append(colors)

    self.tdraw += time()-t0

    self._gdraw(force=force)

    return self

  def show(self, pause=0.1, gamma=1):
    if not self.fig:
      print('-- warn: show is not enabled.')
      return

    self.fig.clear()

    imsize = self.imsize
    t0 = time()
    if self._updated:
      cuda.memcpy_dtoh(self.img, self._img)
      self._updated = False
    plt.imshow(Image.fromarray(unpack(self.img, imsize, gamma=gamma)))
    if self.verbose == 'vv':
      print('-- show. time: {:0.4f}'.format(time()-t0))

    plt.pause(pause)
    return self

  def save(self, fn, gamma=1):
    self._gdraw(force=True)
    imsize = self.imsize
    print('-- wrote:', fn, (imsize, imsize))
    if self._updated:
      cuda.memcpy_dtoh(self.img, self._img)
      self._updated = False
    Image.fromarray(unpack(self.img, imsize, gamma=gamma)).save(fn)

    return self

