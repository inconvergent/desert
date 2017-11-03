# -*- coding: utf-8 -*-

from time import time

import matplotlib.pyplot as plt

import pkg_resources

from numpy import float32 as npfloat
from numpy import int32 as npint
from numpy import pi
from numpy import row_stack
from numpy import zeros

import pycuda.driver as cuda

from PIL import Image

from .helpers import agg
from .helpers import load_kernel
from .helpers import unpack


TWOPI = pi*2


class Desert():

  def __init__(self, imsize, fg, bg, show=True, verbose=False):
    self.imsize = imsize
    self.imsize2 = imsize*imsize
    self.img = zeros((self.imsize2, 4), npfloat)

    self.fg = fg
    self.bg = bg

    self._img = cuda.mem_alloc(self.img.nbytes)

    self.clear()

    self.threads = 256

    # https://documen.tician.de/pycuda/tutorial.html#executing-a-kernel
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

  def draw(self, shapes):
    imsize = self.imsize

    l = []
    c = 0
    pt0 = time()

    sample_verbose = True if self.verbose == 'vv' else None

    for s in shapes:
      l.append(s.sample(imsize, verbose=sample_verbose))
      c += 1
    dots = agg(row_stack(l), imsize)

    if self.verbose is not None:
      print('-- sampled primitives: {:d}. time: {:0.4f}'.format(
            c, time()-pt0))

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
      print('-- drew dots: {:d}. time: {:0.4f}'.format(
            n, time()-dt0))
    self._updated = True

  def imshow(self, pause=0.00001):
    imsize = self.imsize
    t0 = time()

    if self._updated:
      cuda.memcpy_dtoh(self.img, self._img)
      self._updated = False
    plt.imshow(Image.fromarray(unpack(self.img, imsize)))
    if self.verbose == 'vv':
      print('-- show. time: {:0.4f}'\
          .format(time()-t0))

    plt.pause(pause)

  def imsave(self, fn):
    imsize = self.imsize
    print('.. wrote:', fn, (imsize, imsize))
    if self._updated:
      cuda.memcpy_dtoh(self.img, self._img)
      self._updated = False
    Image.fromarray(unpack(self.img, imsize)).save(fn)

