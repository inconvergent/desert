# -*- coding: utf-8 -*-


from numpy import array
from numpy import column_stack
from numpy import float32 as npfloat
from numpy import int32 as npint
from numpy import pi
from numpy import reshape
from numpy import row_stack
from numpy import uint8 as npuint8
from numpy import zeros
from numpy.random import random

from pycuda.driver import In as in_
from pycuda.driver import InOut as inout_
from pycuda.driver import Out as out_
import pycuda.driver as cuda


import matplotlib.pyplot as plt
from PIL import Image

from modules.color import Rgba
from modules.helpers import agg
from modules.helpers import load_kernel
from modules.helpers import unpack


TWOPI = pi*2


class Desert():

  def __init__(self, imsize, fg, bg):
    self.imsize = imsize
    self.imsize2 = imsize*imsize
    self.img = zeros((self.imsize2, 4), npfloat)
    # self.img = self.vals.view().reshape(imsize, imsize, 4)

    # self.fg = pre_alpha(fg)
    # self.bg = pre_alpha(bg)

    self.fg = Rgba(fg)
    self.bg = Rgba(bg)

    self.img[:, :] = self.bg.rgba
    self._img = cuda.mem_alloc(self.img.nbytes)
    cuda.memcpy_htod(self._img, self.img)

    self.threads = 256

    # https://documen.tician.de/pycuda/tutorial.html#executing-a-kernel
    self.cuda_dot = load_kernel(
        'modules/cuda/dot.cu',
        'dot',
        subs={'_THREADS_': self.threads}
        )

  def draw(self, shapes):
    imsize = self.imsize

    dots = agg(row_stack([s.sample(imsize) for s in shapes]),
               imsize)
    n, _ = dots.shape
    blocks = int(n//self.threads + 1)
    self.cuda_dot(npint(n),
                  self._img,
                  in_(dots),
                  in_(self.fg.rgba),
                  block=(self.threads, 1, 1),
                  grid=(blocks, 1))

  def imshow(self):
    imsize = self.imsize

    cuda.memcpy_dtoh(self.img, self._img)
    im = Image.fromarray(unpack(self.img, imsize))
    plt.imshow(im)

