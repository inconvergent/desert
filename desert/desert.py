# -*- coding: utf-8 -*-

from time import time

import matplotlib.pyplot as plt

import pkg_resources

from PIL import Image

from numpy import float32 as npfloat
from numpy import int32 as npint
from numpy import pi
from numpy import zeros
from numpy import row_stack
from numpy import column_stack
from numpy import arange
from numpy import repeat
from numpy import concatenate
from numpy import cumsum

import pycuda.driver as cuda

from .color import black
from .color import white

from .helpers import load_kernel
from .helpers import unpack


TWOPI = pi*2


class Desert():

  def __init__(self, imsize, show=True, verbose=False):
    self.imsize = imsize
    self.imsize2 = imsize*imsize
    self.img = zeros((self.imsize2, 4), npfloat)

    self._img = cuda.mem_alloc(self.img.nbytes)

    self.threads = 256

    self.cuda_dot = load_kernel(
         pkg_resources.resource_filename('desert', 'cuda/dot.cu'),
        'dot',
        subs={'_THREADS_': self.threads}
        )

    self.cuda_agg = load_kernel(
         pkg_resources.resource_filename('desert', 'cuda/agg.cu'),
        'agg',
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
    self.fg = c

  def set_bg(self, c):
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
    xy = row_stack(pts).astype(npfloat)

    aggn, _ = xy.shape
    self.cuda_agg(npint(aggn),
                  npint(imsize),
                  cuda.In(xy),
                  cuda.InOut(ind_count),
                  block=(self.threads, 1, 1),
                  grid=(int(aggn//self.threads) + 1, 1))

    ind_count_reduced = column_stack((arange(len(ind_count)).astype(npint),
                                      ind_count))

    ns = ind_count.nonzero()[0]
    ind_count_reduced = ind_count_reduced[ns, :]

    cm = concatenate(([0], cumsum(ind_count_reduced[:, 1])[:-1])).astype(npint)
    ind_count_reduced = column_stack((ind_count_reduced, cm))

    block_count = [len(x) for x in pts]
    ind_color = repeat(arange(len(block_count)), block_count)

    if self.verbose is not None:
      print('-- sampled primitives: {:d}. time: {:0.4f}'\
          .format(count, time()-t0))

    dotn, _ = ind_count_reduced.shape
    dt0 = time()
    self.cuda_dot(npint(dotn),
                  self._img,
                  cuda.In(ind_color),
                  cuda.In(ind_count_reduced),
                  cuda.In(row_stack(colors).astype(npfloat)),
                  block=(self.threads, 1, 1),
                  grid=(int(dotn//self.threads) + 1, 1))

    if self.verbose is not None:
      print('-- drew dots: {:d}. time: {:0.4f}'.format(dotn, time()-dt0))
    self._updated = True

  def draw(self, cmds):
    imsize = self.imsize
    # TODO: group based on estimated dots to draw?

    pts = []
    colors = []

    count = 0
    t0 = time()

    sample_verbose = True if self.verbose == 'vv' else None

    for cmd in cmds:
      count += 1
      xy = cmd.sample(imsize, verbose=sample_verbose)
      cr = cmd.rgba
      rgba = cr.rgba if cr is not None else self.fg.rgba
      pts.append(xy)
      colors.append(rgba)

    self._draw(pts, colors, t0, count)

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

