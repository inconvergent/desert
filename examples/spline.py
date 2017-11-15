#!/usr/bin/python3
# -*- coding: utf-8 -*-


from sys import argv

from desert.helpers import filename
from numpy import arange
from numpy import array
from numpy import cos
from numpy import pi
from numpy import reshape
from numpy import sin
from numpy import zeros
from numpy import linspace
from numpy import column_stack
from numpy.random import random
from numpy.random import randint

from desert import Desert
from desert import bzspl

from desert.rnd import in_circle

from desert.color import rgb
from desert.color import white
from desert.color import black

from time import time

TWOPI = 2.0*pi


VERBOSE = 'v'

def main(arg):

  with Desert(1000, show=True, verbose=VERBOSE)\
      .init(fg=black(0.001),
            bg=white()) as c:

    density = 0.15
    num = 20
    rad = 0.35

    noise = 0.00005
    spl = []

    for num in randint(5, 20, 15):
      a = sorted(random(num)*TWOPI)
      spl.append(0.5 + column_stack((cos(a), sin(a)))*rad)

    res = []
    for i in range(1, 4000000):
      for xy in spl:
        n = xy.shape[0]
        xy += in_circle(n, 0, 0, 1)*reshape(arange(n), (n, 1))*noise
        res.append(bzspl(xy, density))

      if not i%400:
        c.draw(res).show(gamma=1.5)
        res = []
      if not i%(400*20):
        c.save(filename(arg), gamma=1.5)


if __name__ == '__main__':
  main(argv)

