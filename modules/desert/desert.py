# -*- coding: utf-8 -*-



from numpy import pi
from numpy import zeros
from numpy import column_stack
from numpy import float32 as npfloat
from numpy import int32 as npint
from numpy import uint8 as npuint8

from numpy.random import random

from numpy import reshape

import matplotlib.pyplot as plt
from PIL import Image

import pycuda.autoinit
import pycuda.driver as drv

from pycuda.driver import In as in_
from pycuda.driver import Out as out_
from pycuda.driver import InOut as inout_

from modules.helpers import load_kernel


TWOPI = pi*2


def pre_alpha(c):
  r, g, b, a = c
  return reshape([a*r , a*g, a*b, a], (1, 4))


def unpack(v, imsize):
  alpha = v[:, 3:4]
  return reshape(npuint8(column_stack((v[:, :3]/alpha, alpha))*255),
                 (imsize, imsize, 4))



class Desert():

  def __init__(self, imsize, fg, bg):
    self.imsize = imsize
    self.imsize2 = imsize*imsize
    self.vals = zeros((self.imsize2, 4), npfloat)
    self.img = self.vals.view().reshape(imsize, imsize, 4)

    self.fg = pre_alpha(fg)
    self.bg = pre_alpha(bg)

    self.vals[:, :] = self.bg


    self.threads = 512

    self.__cuda_init()

  def __cuda_init(self):
    self.cuda_dot = load_kernel(
        'modules/cuda/dot.cu',
        'dot',
        subs={'_THREADS_': self.threads}
        )

  def _dot(self, inds):
    num, _ = inds.shape
    blocks = num//self.threads + 1

    # self.cuda_dot(
    #     npint(num),
    #     npint(1),
    #     npint(1),
    #     in_(self.vals),
    #     inout_(1),
    #     inout_(1),
    #     block=(self.threads, 1, 1),
    #     grid=(blocks, 1)
    #     )

  def imshow(self):
    imsize = self.imsize
    im = Image.fromarray(unpack(self.vals, imsize))
    plt.imshow(im)

  def box(self, s, xy, dens):

    try:
      sx, sy = s
    except TypeError:
      sx = s
      sy = s

    imsize = self.imsize
    n = int(sx*sy*dens*(imsize**2))

    xy = random((n, 2))
    self._dot(xy)

