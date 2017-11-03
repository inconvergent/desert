# -*- coding: utf-8 -*-

from time import time

import matplotlib.pyplot as plt

from numpy import float32 as npfloat
from numpy import int32 as npint
from numpy import pi
from numpy import row_stack
from numpy import zeros

import pycuda.driver as cuda

from PIL import Image

from modules.helpers import agg
from modules.helpers import load_kernel
from modules.helpers import unpack


TWOPI = pi*2


class desert():

  def __init__(self, imsize, fg, bg, show=True):
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
        'modules/cuda/dot.cu',
        'dot',
        subs={'_THREADS_': self.threads}
        )

    self.fig = None
    if show:
      self.fig = plt.figure()
      self.fig.patch.set_facecolor('gray')
    self._updated = False

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

  def draw(self, shapes, verbose=None):
    imsize = self.imsize

    l = []
    c = 0
    pt0 = time()

    sample_verbose = True if verbose == 'vv' else None

    for s in shapes:
      l.append(s.sample(imsize, sample_verbose))
      c += 1
    dots = agg(row_stack(l), imsize)

    if verbose is not None:
      print('-- sampled primitives: {: 12d}. time: {:0.4f}'.format(
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

    if verbose is not None:
      print('-- drew dots:          {: 12d}. time: {:0.4f}'.format(
            n, time()-dt0))
    self._updated = True

  def imshow(self, pause=0):
    imsize = self.imsize

    if self._updated:
      cuda.memcpy_dtoh(self.img, self._img)
      self._updated = False
    im = Image.fromarray(unpack(self.img, imsize))
    plt.imshow(im)
    if pause > 0:
      plt.pause(pause)

  def imsave(self, fn):
    imsize = self.imsize
    print('file:', fn, (imsize, imsize))
    if self._updated:
      cuda.memcpy_dtoh(self.img, self._img)
      self._updated = False
    im = Image.fromarray(unpack(self.img, imsize))
    im.save(fn)

