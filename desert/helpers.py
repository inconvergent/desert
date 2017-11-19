#!/usr/bin/python
# -*- coding: utf-8 -*-

from functools import wraps
from time import time
from json import dumps

from pycuda.compiler import SourceModule

from numpy import dstack
from numpy import power
from numpy import reshape
from numpy import transpose
from numpy import uint8 as npuint8



def filename(arg):
  try:
    return arg[1] + '.png'
  except Exception:
    return './tmp.png'


def load_kernel(fn, name, subs=None):
  if not subs:
    subs = {}

  with open(fn, 'r') as f:
    kernel = f.read()

  for k, v in list(subs.items()):
    kernel = kernel.replace(k, str(v))

  mod = SourceModule(kernel)
  return mod.get_function(name)


def unpack(img, imsize, gamma=1):
  alpha = reshape(img[:, 3], (imsize, imsize))

  im = npuint8(
      transpose(
          power(dstack((
              reshape(img[:, 0], (imsize, imsize))/alpha,
              reshape(img[:, 1], (imsize, imsize))/alpha,
              reshape(img[:, 2], (imsize, imsize))/alpha,
              )), gamma)*255, (0, 1, 2)))

  return im


def pprint(j=None):
  if not j:
    print()
    return

  print(dumps(j, indent=2, sort_keys=True))


def pfloat(f):
  # return float('{:0.12f}'.format(f))
  return float(f)


def json_array(aa):
  if aa is None:
    return None

  l = []
  for a in aa:
    try:
      l.append(tuple(pfloat(k) for k in a))
    except Exception:
      l.append((pfloat(a)))
  return l


def is_verbose(f):
  @wraps(f)
  def inside(*args, **kwargs):
    st0 = time()
    res = f(*args, **kwargs)
    if 'verbose' in kwargs and kwargs['verbose'] is not None:
      self = args[0]
      print('.. {:s} time: {:0.4f}'.format(str(self), time()-st0))
    return res
  return inside

