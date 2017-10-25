# -*- coding: utf-8 -*-

import pycuda.autoinit

from numpy import array
from numpy import column_stack
from numpy import float32 as npfloat
from numpy import int32 as npint
from numpy import pi
from numpy import reshape
from numpy import uint8 as npuint8
from numpy import zeros
from numpy.random import random

from pycuda.driver import In as in_
from pycuda.driver import InOut as inout_
from pycuda.driver import Out as out_

import matplotlib.pyplot as plt
from PIL import Image

from modules.helpers import load_kernel
from modules.helpers import pre_alpha
from modules.helpers import unpack

from modules.shapes import Box


TWOPI = pi*2


class Desert():

  def __init__(self, imsize, fg, bg):
    self.imsize = imsize
    self.imsize2 = imsize*imsize
    self.img = zeros((self.imsize2, 4), npfloat)
    # self.img = self.vals.view().reshape(imsize, imsize, 4)

    self.fg = pre_alpha(fg)
    self.bg = pre_alpha(bg)

    self.img[:, :] = self.bg


    self.threads = 512

    self.__init_fxn()
    self.__cuda_init()

  def __cuda_init(self):
    self.cuda_dot = load_kernel(
        'modules/cuda/dot.cu',
        'dot',
        subs={'_THREADS_': self.threads}
        )

  def __init_fxn(self):
    self._fxn = {
        'Box': self._box
        }


  def _dot(self, xy):
    n, _ = xy.shape
    imsize = self.imsize
    blocks = int(n//self.threads + 1)

    self.cuda_dot(
        npint(n),
        npint(imsize),
        inout_(self.img),
        in_(xy.astype(npfloat)),
        in_(self.fg),
        block=(self.threads, 1, 1),
        grid=(blocks, 1)
        )

  def imshow(self):
    imsize = self.imsize
    im = Image.fromarray(unpack(self.img, imsize))
    plt.imshow(im)

  def _box(self, box):
    n = box.get_n(self.imsize)
    print(n)
    xy = (1-2*random((n, 2))) * box.s + box.mid
    self._dot(xy)

  def draw(self, ll):

    for l in ll:
      # print(l.__class__.__name__)
      self._fxn[l.__class__.__name__](l)


  # def circle(self, s, mid, dens):

  #   n = int(4*sx*sy*dens*(self.imsize2))
  #   xy = (1-2*random((n, 2))) * s + mid
  #   self._dot(xy)

