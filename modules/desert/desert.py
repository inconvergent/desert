# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt

from numpy import float32 as npfloat
from numpy import int32 as npint
from numpy import pi
from numpy import row_stack
from numpy import uint8 as npuint8
from numpy import zeros
from numpy.random import random

import pycuda.driver as cuda


import matplotlib.pyplot as plt
from PIL import Image

from modules.helpers import agg
from modules.helpers import load_kernel
from modules.helpers import unpack


TWOPI = pi*2


class Desert():

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

  def draw(self, shapes, verbose=False):
    imsize = self.imsize

    dots = agg(row_stack([s.sample(imsize, verbose) for s in shapes]),
               imsize)
    n, _ = dots.shape
    blocks = int(n//self.threads + 1)

    if verbose:
      print('drawing {:d} dots'.format(n))
    self.cuda_dot(npint(n),
                  self._img,
                  cuda.In(dots),
                  cuda.In(self.fg.rgba),
                  block=(self.threads, 1, 1),
                  grid=(blocks, 1))

  def imshow(self, pause=0):
    imsize = self.imsize

    cuda.memcpy_dtoh(self.img, self._img)
    im = Image.fromarray(unpack(self.img, imsize))
    plt.imshow(im)
    if pause > 0:
      plt.pause(pause)

