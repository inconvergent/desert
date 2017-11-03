# -*- coding: utf-8 -*-

from numpy.random import random
from numpy.linalg import norm
from numpy import zeros
from numpy import reshape
from numpy import column_stack
from numpy import logical_not
from numpy import logical_and
from numpy import array
from numpy import pi as PI
from numpy import cos
from numpy import sin

TWOPI = 2.0*PI


def unit_vec(num, scale):
  from numpy.random import normal

  rnd = normal(size=(num, 3))
  d = norm(rnd, axis=1)
  rnd[:] /= reshape(d, (num, 1))
  return rnd*scale


def in_circle(n, xx, yy, rr):
  """
  get n random points in a circle.
  """

  rnd = random(size=(n, 3))
  t = TWOPI * rnd[:, 0]
  u = rnd[:, 1:].sum(axis=1)
  r = zeros(n, 'float')
  mask = u > 1.
  xmask = logical_not(mask)
  r[mask] = 2.-u[mask]
  r[xmask] = u[xmask]
  xyp = reshape(rr * r, (n, 1)) * column_stack((cos(t), sin(t)))
  dartsxy = xyp + array([xx, yy])
  return dartsxy

