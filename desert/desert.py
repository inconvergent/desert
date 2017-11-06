# -*- coding: utf-8 -*-

from time import time

import matplotlib.pyplot as plt

import pkg_resources

from PIL import Image

from numpy import float32 as npfloat
from numpy import int32 as npint
from numpy import pi
from numpy import row_stack
from numpy import zeros

import pycuda.driver as cuda

from .color import black
from .color import white

from .helpers import agg
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

  def _draw(self, l, t, c):
    if not l:
      return
    imsize = self.imsize
    dots = agg(row_stack(l), imsize)

    if self.verbose is not None:
      print('-- sampled primitives: {:d}. time: {:0.4f}'.format(c, time()-t))

    n, _ = dots.shape
    blocks = int(n//self.threads + 1)

    dt0 = time()
    self.cuda_dot(npint(n),
                  self._img,
                  cuda.In(dots),
                  cuda.In(self.fg.rgba),
                  block=(self.threads, 1, 1),
                  grid=(blocks, 1))

    if self.verbose is not None:
      print('-- drew dots: {:d}. time: {:0.4f}'.format(n, time()-dt0))
    self._updated = True

  def draw(self, cmds):
    imsize = self.imsize
    # TODO: group based on estimated dots to draw?

    l = []
    c = 0
    pt0 = time()

    sample_verbose = True if self.verbose == 'vv' else None

    for cmd in cmds:
      c += 1
      name = type(cmd).__name__
      if name in self.__fg_set:
        self._draw(l, pt0, c)
        self.set_fg(cmd)
        l = []
        pt0 = 0
        c = 0
      elif name == 'Clear':
        # TODO: color
        self.clear()
        l = []
        c = 0
        pt0 = 0
      else:
        l.append(cmd.sample(imsize, verbose=sample_verbose))

    self._draw(l, pt0, c)

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

