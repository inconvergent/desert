#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycuda.compiler import SourceModule
from numpy import reshape
from numpy import column_stack
from numpy import float32 as npfloat
from numpy import uint8 as npuint8


def load_kernel(fn, name, subs=None):
  if not subs:
    subs = {}

  with open(fn, 'r') as f:
    kernel = f.read()

  for k, v in list(subs.items()):
    kernel = kernel.replace(k, str(v))

  mod = SourceModule(kernel)
  return mod.get_function(name)


def pre_alpha(c):
  r, g, b, a = c
  return reshape([a*r , a*g, a*b, a], (1, 4)).astype(npfloat)


def unpack(v, imsize):
  alpha = v[:, 3:4]
  return reshape(npuint8(column_stack((v[:, :3]/alpha, alpha))*255),
                 (imsize, imsize, 4))



