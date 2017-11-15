#!/usr/bin/python3
# -*- coding: utf-8 -*-


from sys import argv

from desert.helpers import filename
from numpy import array
from numpy import cos
from numpy import pi
from numpy import sin
from numpy import zeros
from numpy.random import random

from desert import Desert
from desert import stroke

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

    density = 0.02
    a = random(2)*TWOPI
    acc = zeros(2)
    noise = 0.000000005
    rad = 0.45

    resa = []
    resb = []

    for i in range(4000000):
      a += acc
      acc += (1-2*random(2))*noise

      resa.append((cos(a[0]), sin(a[0])))
      resb.append((cos(a[1]), sin(a[1])))

      if not i%100000:
        c.draw([stroke(0.5 + array(resa)*rad,
                       0.5 + array(resb)*rad,
                       density)]).show()
        resa = []
        resb = []

    c.save(filename(arg), gamma=1.5)


if __name__ == '__main__':
  main(argv)

