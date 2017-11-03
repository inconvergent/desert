#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter

from pycuda.compiler import SourceModule
from numpy import column_stack
from numpy import dstack
from numpy import float32 as npfloat
from numpy import int32 as npint
from numpy import logical_and
from numpy import reshape
from numpy import row_stack
from numpy import transpose
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


def ind_filter(xy):
  return xy[logical_and(
          logical_and(xy[:, 0]>=0, xy[:,0]<1.0),
          logical_and(xy[:, 1]>=0, xy[:,1]<1.0)), :]



def agg(xy, imsize):
  return row_stack(Counter(imsize * (xy[:, 1]*imsize).astype(npint) +
                           (xy[:, 0]*imsize).astype(npint)).items())\
           .astype(npint)


def unpack(img, imsize, verbose=False):
  alpha = reshape(img[:, 3], (imsize, imsize))

  im = npuint8(
      transpose(
        dstack((
          reshape(img[:, 0], (imsize, imsize))/alpha,
          reshape(img[:, 1], (imsize, imsize))/alpha,
          reshape(img[:, 2], (imsize, imsize))/alpha,
          ))*255 , (0, 1, 2)))

  if verbose:
    print(im.shape)
    print(im[:, :, 0])
    print(im[:, :, 1])
    print(im[:, :, 2])

  return im

